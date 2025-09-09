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

#ifndef ARBORX_DISTRIBUTED_DBSCAN_HELPERS_HPP
#define ARBORX_DISTRIBUTED_DBSCAN_HELPERS_HPP

#include <ArborX_Box.hpp>
#include <ArborX_LinearBVH.hpp>
#include <detail/ArborX_Distributor.hpp>
#include <detail/ArborX_Predicates.hpp>
#include <detail/ArborX_TreeConstruction.hpp>
#include <kokkos_ext/ArborX_KokkosExtKernelStdAlgorithms.hpp>
#include <kokkos_ext/ArborX_KokkosExtViewHelpers.hpp>
#include <misc/ArborX_SortUtils.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <mpi.h>

namespace ArborX::Details
{

template <typename Points, typename GhostPoints>
struct UnifiedPoints
{
  Points _points;
  GhostPoints _ghost_points;
};

template <typename Points, typename GhostOffsets, typename GhostIds,
          typename Coordinate>
struct PointsRequiringResolution
{
  Points _points;
  GhostOffsets _ghost_offsets;
  GhostIds _ghost_ids;
  Coordinate _eps;
};

// FIXME_CLANG(Clang<=17): Clang 16 and some Clang 17 based compilers do not
// support aggregate initialization type deduction
// https://github.com/llvm/llvm-project/issues/54050
// Reproducer: https://godbolt.org/z/nEdjb5rP4
#if defined(__clang__) && KOKKOS_COMPILER_CLANG < 17001
template <typename Points, typename GhostPoints>
KOKKOS_DEDUCTION_GUIDE UnifiedPoints(Points, GhostPoints)
    -> UnifiedPoints<Points, GhostPoints>;

template <typename Points, typename GhostOffsets, typename GhostIds,
          typename Coordinate>
KOKKOS_DEDUCTION_GUIDE PointsRequiringResolution(Points, GhostOffsets, GhostIds,
                                                 Coordinate)
    -> PointsRequiringResolution<Points, GhostOffsets, GhostIds, Coordinate>;
#endif

} // namespace ArborX::Details

template <typename Points, typename GhostPoints>
struct ArborX::AccessTraits<ArborX::Details::UnifiedPoints<Points, GhostPoints>>
{
  using Self = ArborX::Details::UnifiedPoints<Points, GhostPoints>;
  using memory_space = typename Points::memory_space;

  static KOKKOS_FUNCTION auto size(Self const &self)
  {
    return self._points.size() + self._ghost_points.size();
  }
  static KOKKOS_FUNCTION auto get(Self const &self, size_t i)
  {
    auto const num_local = self._points.size();
    return (i < num_local ? self._points(i)
                          : self._ghost_points(i - num_local));
  }
};

template <typename Points, typename GhostOffsets, typename GhostIds,
          typename Coordinate>
struct ArborX::AccessTraits<ArborX::Details::PointsRequiringResolution<
    Points, GhostOffsets, GhostIds, Coordinate>>
{
  using Self = ArborX::Details::PointsRequiringResolution<Points, GhostOffsets,
                                                          GhostIds, Coordinate>;
  using memory_space = typename Points::memory_space;

  static KOKKOS_FUNCTION auto size(Self const &self)
  {
    return self._ghost_offsets.size() - 1;
  }
  static KOKKOS_FUNCTION auto get(Self const &self, size_t i)
  {
    auto const id = self._ghost_ids(self._ghost_offsets(i));
    return ArborX::attach(
        ArborX::intersects(ArborX::Sphere(self._points(id), self._eps)), id);
  }
};

namespace ArborX::Details
{

template <typename Index>
void computeCountsAndOffsets(MPI_Comm comm, Index k, std::vector<int> &counts,
                             std::vector<Index> &offsets)
{
  Kokkos::Profiling::ScopedRegion guard(
      "ArborX::DistributedDBSCAN::computeRankOffset");

  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  counts.resize(comm_size);
  counts[comm_rank] = k;
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, counts.data(), 1, MPI_INT,
                comm);

  offsets.resize(comm_size + 1);
  offsets[0] = 0;
  for (int i = 0; i < comm_size; ++i)
    offsets[i + 1] = counts[i] + offsets[i];
}

template <typename ExecutionSpace, typename Points>
auto gatherGlobalBoxes(MPI_Comm comm, ExecutionSpace const &space,
                       Points const &points)
{
  using MemorySpace = typename Points::memory_space;

  using Point = typename Points::value_type;
  constexpr int DIM = GeometryTraits::dimension_v<Point>;
  using Coordinate = GeometryTraits::coordinate_type_t<Point>;
  using Box = Box<DIM, Coordinate>;

  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  Box local_box;
  Details::TreeConstruction::calculateBoundingBoxOfTheScene(
      space,
      Details::Indexables{points, Experimental::DefaultIndexableGetter{}},
      local_box);

  Kokkos::View<Box *, MemorySpace> global_boxes(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::DistributedDBSCAN::rank_boxes"),
      comm_size);
#ifdef ARBORX_ENABLE_GPU_AWARE_MPI
  Kokkos::deep_copy(space, Kokkos::subview(global_boxes, comm_rank), local_box);
  space.fence(
      "ArborX::DistributedDBSCAN (fill on device done before MPI_Allgather)");

  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                static_cast<void *>(global_boxes.data()), sizeof(Box), MPI_BYTE,
                comm);
#else
  Kokkos::DefaultHostExecutionSpace host_exec;
  auto global_boxes_host = Kokkos::create_mirror_view(
      Kokkos::view_alloc(host_exec, Kokkos::WithoutInitializing), global_boxes);
  host_exec.fence();
  global_boxes_host(comm_rank) = local_box;

  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                static_cast<void *>(global_boxes_host.data()), sizeof(Box),
                MPI_BYTE, comm);

  Kokkos::deep_copy(space, global_boxes, global_boxes_host);
#endif

  return global_boxes;
}

struct IndexOnlyCallback
{
  template <typename Query, typename Value, typename Index, typename Output>
  KOKKOS_FUNCTION auto operator()(Query const &,
                                  PairValueIndex<Value, Index> const &value,
                                  Output const &out) const
  {
    out(value.index);
  }
};

template <typename ExecutionSpace, typename Points, typename Coordinate,
          typename Offsets, typename RanksTo>
void computeRanksTo(MPI_Comm comm, ExecutionSpace const &space,
                    Points const &points, Coordinate eps, Offsets &offsets,
                    RanksTo &ranks_to)
{
  std::string prefix = "ArborX::DistributedDBSCAN::computeRanksTo";
  Kokkos::Profiling::ScopedRegion guard(prefix);
  prefix += "::";

  using MemorySpace = typename Points::memory_space;

  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  auto global_boxes = gatherGlobalBoxes(comm, space, points);
  using Boxes = decltype(global_boxes);
  using Box = typename Boxes::value_type;

  // Filter out: a) local box, and b) boxes that are not close to the local box
  // This changes the local search to neighbor-to-neighbor rather than global,
  // improving weak scaling.
  Kokkos::View<PairValueIndex<Box, int> *, MemorySpace> primitives(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         prefix + "primitives"),
      comm_size);
  int num_primitives;
  Kokkos::parallel_scan(
      prefix + "filter_out_global_boxes",
      Kokkos::RangePolicy(space, 0, comm_size),
      KOKKOS_LAMBDA(int i, int &update, bool is_final) {
        if (i == comm_rank ||
            distance(global_boxes(i), global_boxes(comm_rank)) > eps)
          return;

        if (is_final)
          primitives(update) = {global_boxes(i), i};
        ++update;
      },
      num_primitives);
  Kokkos::resize(space, primitives, num_primitives);
  Kokkos::resize(space, global_boxes, 0); // free space

  BoundingVolumeHierarchy index(space, primitives);
  index.query(space, Experimental::make_intersects(points, eps),
              IndexOnlyCallback{}, ranks_to, offsets);
}

template <typename ExecutionSpace, typename Points, typename Coordinate,
          typename GhostPoints, typename GhostIds, typename GhostRanks>
void forwardNeighbors(MPI_Comm comm, ExecutionSpace space, Points const &points,
                      Coordinate eps, GhostPoints &ghost_points,
                      GhostIds &ghost_ids, GhostRanks &ghost_ranks)
{
  std::string prefix = "ArborX::DistributedDBSCAN::forwardNeighbors";
  Kokkos::Profiling::ScopedRegion guard(prefix);
  prefix += "::";

  using MemorySpace = typename Points::memory_space;

  using Point = typename Points::value_type;

  Kokkos::View<int *, MemorySpace> offsets(prefix + "offsets", 0);
  Kokkos::View<int *, MemorySpace> ranks_to(prefix + "ranks_to", 0);
  computeRanksTo(comm, space, points, eps, offsets, ranks_to);

  // FIXME: very similar to DistributedTree::forwardQueries
  Distributor<MemorySpace> distributor(comm);

  auto const n_exports = ranks_to.size();
  auto const n_imports = distributor.createFromSends(space, ranks_to);

  // If tree is empty (running on one rank), the offsets.size() will consist of
  // only one element, independent of the number of points.
  // FIXME: not 100% sure what's going on. We construct a tree based on an
  // invalid box. What's supposed to happen in this case?
  auto const n_offset_points = (offsets.size() > 0 ? offsets.size() - 1 : 0);

  {
    Kokkos::View<Point *, MemorySpace> export_points(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           prefix + "::export_points"),
        n_exports);
    Kokkos::parallel_for(
        prefix + "fill_export_points",
        Kokkos::RangePolicy(space, 0, n_offset_points), KOKKOS_LAMBDA(int i) {
          for (int j = offsets(i); j < offsets(i + 1); ++j)
            export_points(j) = points(i);
        });
    KokkosExt::reallocWithoutInitializing(space, ghost_points, n_imports);
    distributor.doPostsAndWaits(space, export_points, ghost_points);
  }

  {
    Kokkos::View<int *, MemorySpace> export_ids(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           prefix + "export_ids"),
        n_exports);
    Kokkos::parallel_for(
        prefix + "fill_export_ids",
        Kokkos::RangePolicy(space, 0, n_offset_points), KOKKOS_LAMBDA(int i) {
          for (int j = offsets(i); j < offsets(i + 1); ++j)
            export_ids(j) = i;
        });
    KokkosExt::reallocWithoutInitializing(space, ghost_ids, n_imports);
    distributor.doPostsAndWaits(space, export_ids, ghost_ids);
  }

  {
    int comm_rank;
    MPI_Comm_rank(comm, &comm_rank);

    Kokkos::View<int *, MemorySpace> export_ranks(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           prefix + "export_ranks"),
        n_exports);
    Kokkos::deep_copy(space, export_ranks, comm_rank);
    KokkosExt::reallocWithoutInitializing(space, ghost_ranks, n_imports);
    distributor.doPostsAndWaits(space, export_ranks, ghost_ranks);
  }
}

// FIXME: similar to DistributedTree::communicateResultsBack
template <typename ExecutionSpace, typename Ranks, typename Ids,
          typename Labels>
void communicateNeighborDataBack(MPI_Comm comm, ExecutionSpace space,
                                 Ranks const &ranks, Ids &ids, Labels &labels)
{
  std::string prefix = "ArborX::DistributedDBSCAN::communicateNeighborDataBack";
  Kokkos::Profiling::ScopedRegion guard(prefix);
  prefix += "::";

  using MemorySpace = typename Labels::memory_space;

  // We are assuming here that if the same rank is related to multiple batches
  // these batches appear consecutively. Hence, no reordering is necessary.
  Distributor<MemorySpace> distributor(comm);
  int const n_imports = distributor.createFromSends(space, ranks);

  {
    Kokkos::View<int *, MemorySpace> import_ids(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing, ids.label()),
        n_imports);
    distributor.doPostsAndWaits(space, ids, import_ids);
    ids = import_ids;
  }

  {
    Labels import_labels(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing, labels.label()),
        n_imports);
    distributor.doPostsAndWaits(space, labels, import_labels);
    labels = import_labels;
  }
}

struct MergePair
{
  long long from;
  long long to;

private:
  // performs lexicographical comparison by comparing first the weights and then
  // the unordered pair of vertices
  friend KOKKOS_FUNCTION constexpr bool operator<(MergePair const &lhs,
                                                  MergePair const &rhs)
  {
    if (lhs.from < rhs.from)
      return true;
    if (lhs.from == rhs.from)
      return lhs.to < rhs.to;
    return false;
  }
  friend KOKKOS_FUNCTION constexpr bool operator==(MergePair const &lhs,
                                                   MergePair const &rhs)
  {
    return lhs.from == rhs.from && lhs.to == rhs.to;
  }
  friend std::ostream &operator<<(std::ostream &os, MergePair const &mp)
  {
    return os << "{" << mp.from << " -> " << mp.to << "}";
  }
};

template <typename ExecutionSpace, typename CorePoints, typename Labels,
          typename GhostOffsets, typename GhostIds, typename GhostLabels,
          typename MergePairs>
void computeMergePairs(ExecutionSpace const &space, CorePoints const &is_core,
                       Labels &local_labels, GhostOffsets const &ghost_offsets,
                       GhostIds const &ghost_ids,
                       GhostLabels const &ghost_labels, MergePairs &merge_pairs)
{
  std::string prefix = "ArborX::DistributedDBSCAN::computeMergePairs";
  Kokkos::Profiling::ScopedRegion guard(prefix);
  prefix += "::";

  auto const num_offsets = ghost_offsets.size();

  Kokkos::resize(Kokkos::view_alloc(space, Kokkos::WithoutInitializing),
                 merge_pairs, ghost_labels.size());
  int num_merge_pairs;
  Kokkos::parallel_scan(
      prefix + "process_labels", Kokkos::RangePolicy(space, 0, num_offsets - 1),
      KOKKOS_LAMBDA(int const i, int &update, bool is_final) {
        auto const begin = ghost_offsets(i);
        auto const end = ghost_offsets(i + 1);
        auto const id = ghost_ids(begin);
        auto local_label = local_labels(id);
        bool const is_local_valid = (local_label != -1);

        int num_valid = (end - begin) + (is_local_valid);
        if (num_valid < 2)
        {
          // A noise point or a point with a single label
          return;
        }

        if (!is_core(id))
        {
          // A border point with multiple labels
          if (is_final && !is_local_valid)
          {
            // Update local label if it is invalid (all imported labels are
            // valid as we filter out noise before communicating)
            local_labels(id) = ghost_labels(begin);
          }

          return;
        }

        // A core point with multiple labels
        auto min_label = (is_local_valid ? local_label : LLONG_MAX);
        for (int j = begin; j < end; ++j)
        {
          auto const label_j = ghost_labels(j);
          KOKKOS_ASSERT(label_j != -1);
          min_label = Kokkos::min(label_j, min_label);
        }

        if (is_local_valid && local_label != min_label)
        {
          if (is_final)
          {
            merge_pairs(update) = {local_label, min_label};
            local_labels(id) = min_label;
          }
          ++update;
        }

        for (int j = begin; j < end; ++j)
        {
          auto label_j = ghost_labels(j);
          if (label_j == min_label)
            continue;

          if (is_final)
            merge_pairs(update) = {label_j, min_label};
          ++update;
        }
      },
      num_merge_pairs);
  Kokkos::resize(space, merge_pairs, num_merge_pairs);
}

template <typename ExecutionSpace, typename MergePairs>
void sortAndFilterMergePairs(ExecutionSpace const &space,
                             MergePairs &merge_pairs)
{
  std::string prefix = "ArborX::DistributedDBSCAN::sortAndFilterMergePairs";
  Kokkos::Profiling::ScopedRegion guard(prefix);
  prefix += "::";

  if (merge_pairs.size() == 0)
    return;

  Kokkos::sort(space, merge_pairs);

  int const n = merge_pairs.size();

  auto new_merge_pairs =
      KokkosExt::cloneWithoutInitializingNorCopying(space, merge_pairs);

  int num_unique;
  Kokkos::parallel_scan(
      prefix + "filter", Kokkos::RangePolicy(space, 0, n),
      KOKKOS_LAMBDA(int i, int &update, bool is_final) {
        // For the same "from" value we are only intested in the lowest "to".
        // As the merge pairs are sorted, we just need to grab the first one
        // from the sequence with the same "from".
        if (i > 0 && merge_pairs(i).from == merge_pairs(i - 1).from)
          return;

        if (is_final)
          new_merge_pairs(update) = merge_pairs(i);
        ++update;

        // Insert other connections that may be required on other ranks
        // FIXME: I need to convince myself that this is truly necessary, or
        // if it's a fix for something that should be addressed in a different
        // place
        auto from_i = merge_pairs(i).from;
        auto to = merge_pairs(i).to;
        int j = i + 1;
        while (j < n && merge_pairs(j).from == from_i)
        {
          if (merge_pairs(j).to != merge_pairs(j - 1).to)
          {
            if (is_final)
            {
              KOKKOS_ASSERT(merge_pairs(j).to != to);
              new_merge_pairs(update) = {merge_pairs(j).to, to};
            }
            ++update;
          }
          ++j;
        }
      },
      num_unique);
  Kokkos::resize(space, new_merge_pairs, num_unique);
  merge_pairs = new_merge_pairs;

  // Re-sort after reinsertion
  // FIXME: I don't know whether we have duplicates here or not. If we do, I
  // don't think it matters
  Kokkos::sort(space, merge_pairs);
}

template <typename ExecutionSpace, typename Pairs>
void communicateMergePairs(MPI_Comm comm, ExecutionSpace const &space,
                           Pairs const &local_merge_pairs,
                           Pairs &global_merge_pairs)
{
  std::string prefix = "ArborX::DistributedDBSCAN::communicateMergePairs";
  Kokkos::Profiling::ScopedRegion guard(prefix);
  prefix += "::";

  std::vector<int> counts;
  std::vector<int> offsets;
  computeCountsAndOffsets(comm, (int)local_merge_pairs.size(), counts, offsets);

  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  auto global_n = offsets.back();
  auto slice = Kokkos::make_pair(offsets[comm_rank], offsets[comm_rank + 1]);

  // Scale counts and offsets by sizeof(pair) as we send in MPI_BYTE
  auto const sc = sizeof(typename Pairs::value_type);
  std::for_each(counts.begin(), counts.end(), [sc](auto &x) { x *= sc; });
  std::for_each(offsets.begin(), offsets.end(), [sc](auto &x) { x *= sc; });

  Kokkos::resize(Kokkos::view_alloc(space, Kokkos::WithoutInitializing),
                 global_merge_pairs, global_n);

#ifdef ARBORX_ENABLE_GPU_AWARE_MPI
  Kokkos::deep_copy(space, Kokkos::subview(global_merge_pairs, slice),
                    local_merge_pairs);
  space.fence(prefix + " (fill on device done before MPI_Allgatherv)");

  MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                 static_cast<void *>(global_merge_pairs.data()), counts.data(),
                 offsets.data(), MPI_BYTE, comm);
#else
  Kokkos::DefaultHostExecutionSpace host_exec;
  auto global_merge_pairs_host = Kokkos::create_mirror_view(
      Kokkos::view_alloc(host_exec, Kokkos::WithoutInitializing),
      global_merge_pairs);
  host_exec.fence();
  Kokkos::deep_copy(space, Kokkos::subview(global_merge_pairs_host, slice),
                    local_merge_pairs);
  space.fence();

  MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                 static_cast<void *>(global_merge_pairs_host.data()),
                 counts.data(), offsets.data(), MPI_BYTE, comm);

  Kokkos::deep_copy(space, global_merge_pairs, global_merge_pairs_host);
#endif
}

template <typename ExecutionSpace, typename MergePairs, typename Labels>
void relabel(ExecutionSpace const &space, MergePairs const &merge_pairs,
             Labels &labels)
{
  std::string prefix = "ArborX::DistributedDBSCAN::relabel";
  Kokkos::Profiling::ScopedRegion guard(prefix);
  prefix += "::";

  using MemorySpace = typename Labels::memory_space;

  int const num_merge_pairs = merge_pairs.size();

  // unzip
  Kokkos::View<long long *, MemorySpace> from(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing, prefix + "from"),
      num_merge_pairs);
  Kokkos::View<long long *, MemorySpace> to(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing, prefix + "to"),
      num_merge_pairs);
  Kokkos::parallel_for(
      prefix + "unzip", Kokkos::RangePolicy(space, 0, num_merge_pairs),
      KOKKOS_LAMBDA(int i) {
        from(i) = merge_pairs(i).from;
        to(i) = merge_pairs(i).to;
      });

  Kokkos::parallel_for(
      prefix + "relabel", Kokkos::RangePolicy(space, 0, labels.size()),
      KOKKOS_LAMBDA(int i) {
        auto label = labels(i);
        int pos = num_merge_pairs;
        while (true)
        {
          int next = KokkosExt::lower_bound(
                         from.data(), from.data() + num_merge_pairs, label) -
                     from.data();
          if (next == num_merge_pairs || from(next) != label)
            break;

          pos = next;
          label = to(pos);
        }
        if (pos != num_merge_pairs)
          labels(i) = label;
      });
}

} // namespace ArborX::Details

#endif
