/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DISTRIBUTED_DBSCAN_HPP
#define ARBORX_DISTRIBUTED_DBSCAN_HPP

#include "kokkos_ext/ArborX_KokkosExtStdAlgorithms.hpp" // FIXME: remove
#include <ArborX_DBSCAN.hpp>
#include <detail/ArborX_AccessTraits.hpp>
#include <detail/ArborX_DistributedDBSCANHelpers.hpp>

#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <mpi.h>

namespace ArborX::Experimental
{

template <typename ExecutionSpace, typename Primitives, typename Coordinate,
          typename Labels>
void dbscan(MPI_Comm comm, ExecutionSpace const &space,
            Primitives const &primitives, Coordinate eps, int core_min_size,
            Labels &labels,
            DBSCAN::Parameters const &params = DBSCAN::Parameters())
{
  std::string prefix = "ArborX::DistributedDBSCAN";
  Kokkos::Profiling::ScopedRegion guard(prefix);
  prefix += "::";

  namespace KokkosExt = ArborX::Details::KokkosExt;

  using Points = Details::AccessValues<Primitives>;
  using MemorySpace = typename Points::memory_space;

  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value,
      "Primitives must be accessible from the execution space");

  ARBORX_ASSERT(eps > 0);
  ARBORX_ASSERT(core_min_size >= 2);

  using Point = typename Points::value_type;
  static_assert(GeometryTraits::is_point_v<Point>);
  static_assert(
      std::is_same_v<typename GeometryTraits::coordinate_type<Point>::type,
                     Coordinate>);

  bool const is_special_case = (core_min_size == 2);

  Points points{primitives}; // NOLINT
  int const n_local = points.size();

  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  // Step 1: receive ghost neighbors
  Kokkos::View<int *, MemorySpace> ghost_ids(prefix + "ghost_ids", 0);
  Kokkos::View<Point *, MemorySpace> ghost_points(prefix + "ghost_points", 0);
  Kokkos::View<int *, MemorySpace> ghost_ranks(prefix + "ghost_ranks", 0);
  // For minPts=2 case, we only need to fetch the ponts within eps distance.
  // For minPts>2, we need points within 2*eps distance to allow to use the
  // local DBSCAN algorithm to determine core points.
  Details::forwardNeighbors(
      comm, space, points,
      (is_special_case ? eps : std::nextafter(2 * eps, 10 * eps)), ghost_points,
      ghost_ids, ghost_ranks);
  int const n_ghost = ghost_points.size();

  // Step 2: do local DBSCAN
  auto local_labels =
      dbscan(space, Details::UnifiedPoints{points, ghost_points}, eps,
             core_min_size, params);

  // Step 3: convert local labels to global
  Kokkos::View<long long *, MemorySpace> rank_offsets(prefix + "rank_offsets",
                                                      0);
  {
    std::vector<int> counts;
    std::vector<long long> offsets;
    Details::computeCountsAndOffsets(comm, (long long)n_local, counts, offsets);

    Kokkos::resize(Kokkos::view_alloc(space, Kokkos::WithoutInitializing),
                   rank_offsets, offsets.size());
    Kokkos::deep_copy(
        space, rank_offsets,
        Kokkos::View<long long *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
            offsets.data(), offsets.size()));
  }
  Kokkos::resize(Kokkos::view_alloc(space, Kokkos::WithoutInitializing), labels,
                 local_labels.size());
  Kokkos::parallel_for(
      prefix + "convert_labels",
      Kokkos::RangePolicy(space, 0, local_labels.size()), KOKKOS_LAMBDA(int i) {
        auto label = local_labels(i);
        if (label == -1)
        {
          labels(i) = -1;
          return;
        }

        if (label < n_local)
          labels(i) = rank_offsets(comm_rank) + label;
        else
          labels(i) = rank_offsets(ghost_ranks(label - n_local)) +
                      ghost_ids(label - n_local);
      });
  Kokkos::resize(local_labels, 0); // free space

  // Step 4: pack and communicate results back to owning ranks
  Kokkos::View<long long *, MemorySpace> ghost_labels(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         prefix + "ghost_labels"),
      n_ghost);
  Kokkos::deep_copy(
      space, ghost_labels,
      Kokkos::subview(labels, Kokkos::make_pair(n_local, n_local + n_ghost)));
  Kokkos::resize(space, labels, n_local);
  // Avoid communicating noise points
  int num_compressed;
  Kokkos::parallel_scan(
      prefix + "compress", Kokkos::RangePolicy(space, 0, n_ghost),
      KOKKOS_LAMBDA(int i, int &update, bool is_final) {
        if (ghost_labels(i) != -1)
        {
          if (is_final)
          {
            ghost_ranks(update) = ghost_ranks(i);
            ghost_ids(update) = ghost_ids(i);
            ghost_labels(update) = ghost_labels(i);
          }
          ++update;
        }
      },
      num_compressed);
  Kokkos::resize(space, ghost_ranks, num_compressed);
  Kokkos::resize(space, ghost_ids, num_compressed);
  Kokkos::resize(space, ghost_labels, num_compressed);
  Details::communicateNeighborDataBack(comm, space, ghost_ranks, ghost_ids,
                                       ghost_labels);
  Details::KokkosExt::sortByKey(space, ghost_ids, ghost_labels);
  Kokkos::View<int *, MemorySpace> ghost_offsets(
      Kokkos::view_alloc(space, prefix + "ghost_offsets"), 0);
  Details::computeOffsetsInOrderedView(space, ghost_ids, ghost_offsets);
  Kokkos::resize(ghost_ranks, 0); // free space

  // Step 5: process multi-labeled indices
  Kokkos::View<Details::MergePair *, MemorySpace> local_merge_pairs(
      prefix + "local_merge_pairs", 0);
  if (ghost_offsets.size() > 1)
  {
    if (is_special_case)
    {
      Details::computeMergePairs(space, Details::CCSCorePoints{}, labels,
                                 ghost_offsets, ghost_ids, ghost_labels,
                                 local_merge_pairs);
    }
    else
    {
      // As we are treating local DBSCAN as a black box, we always do this,
      // even if it may be unnecessary (e.g., FDBSCAN).
      BoundingVolumeHierarchy bvh(space,
                                  Details::UnifiedPoints{points, ghost_points});

      // Find the number of neighbors only for points that appear in the ghost
      // ids as we only need to resolve the labels of those points. We know
      // that none of them are noise (the received labels are never -1 because
      // we filter those before communicating). Some of them may be border
      // points, but it should be a small fraction, and filtering them out is
      // not efficient.
      Kokkos::View<int *, MemorySpace> num_neigh(prefix + "num_neighbors",
                                                 n_local);
      bvh.query(space,
                Details::PointsRequiringResolution{points, ghost_offsets,
                                                   ghost_ids, eps},
                Details::CountUpToN<MemorySpace>{num_neigh, core_min_size});

      Details::computeMergePairs(
          space,
          Details::DBSCANCorePoints<MemorySpace>{num_neigh, core_min_size},
          labels, ghost_offsets, ghost_ids, ghost_labels, local_merge_pairs);
    }
    sortAndFilterMergePairs(space, local_merge_pairs);
  }
  Kokkos::resize(ghost_points, 0); // free space
  Kokkos::resize(ghost_ids, 0);    // free space
  Kokkos::resize(ghost_labels, 0); // free space

  // Step 6: communicate merge pairs (all-to-all)
  Kokkos::View<Details::MergePair *, MemorySpace> global_merge_pairs(
      prefix + "global_merge_pairs", 0);
  communicateMergePairs(comm, space, local_merge_pairs, global_merge_pairs);
  Details::sortAndFilterMergePairs(space, global_merge_pairs);

  // Step 7: flatten the labels
  Details::relabel(space, global_merge_pairs, labels);
}

} // namespace ArborX::Experimental

#endif
