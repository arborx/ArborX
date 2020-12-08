/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DBSCAN_HPP
#define ARBORX_DBSCAN_HPP

#include <ArborX_DetailsDBSCANCallback.hpp>
#include <ArborX_DetailsDBSCANVerification.hpp>
#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_LinearBVH.hpp>

#include <chrono>

namespace ArborX
{

template <typename View>
struct PrimitivesWithRadius
{
  View _M_view;
  double _r;
};

template <typename View>
auto buildPredicates(View v, double r)
{
  return PrimitivesWithRadius<View>{v, r};
}

template <typename View>
struct AccessTraits<PrimitivesWithRadius<View>, PredicatesTag>
{
  using memory_space = typename View::memory_space;
  using Predicates = PrimitivesWithRadius<View>;
  static size_t size(Predicates const &w) { return w._M_view.extent(0); }
  static KOKKOS_FUNCTION auto get(Predicates const &w, size_t i)
  {
    return attach(intersects(Sphere{w._M_view(i), w._r}), (int)i);
  }
};

namespace DBSCAN
{

template <typename MemorySpace>
struct NumNeighEarlyExitCallback
{
  Kokkos::View<int *, MemorySpace> _num_neigh;
  int _core_min_size = 1;

  template <typename Query>
  KOKKOS_FUNCTION auto operator()(Query const &query, int) const
  {
    auto i = getData(query);
    Kokkos::atomic_fetch_add(&_num_neigh(i), 1);

    if (_num_neigh(i) < _core_min_size)
      return ArborX::CallbackTreeTraversalControl::normal_continuation;

    // Once _core_min_size neighbors are found, it is guaranteed to be a core
    // point, and there is no reason to continue the search.
    return ArborX::CallbackTreeTraversalControl::early_exit;
  }
};

struct CCSCorePoints
{
  KOKKOS_FUNCTION bool operator()(int) const { return true; }
};

template <typename MemorySpace>
struct DBSCANCorePoints
{
  Kokkos::View<int *, MemorySpace> _num_neigh;
  int _core_min_size = 1;

  KOKKOS_FUNCTION bool operator()(int const i) const
  {
    return _num_neigh(i) >= _core_min_size;
  }
};

template <typename ExecutionSpace, typename Primitives,
          typename ClusterIndicesView, typename ClusterOffsetView>
void dbscan(ExecutionSpace exec_space, Primitives const &primitives,
            ClusterIndicesView &cluster_indices,
            ClusterOffsetView &cluster_offset, float eps, int core_min_size = 1,
            int cluster_min_size = 2, bool verbose = false, bool verify = false)
{
  static_assert(Kokkos::is_view<ClusterIndicesView>{}, "");
  static_assert(Kokkos::is_view<ClusterOffsetView>{}, "");
  static_assert(std::is_same<typename ClusterIndicesView::value_type, int>{},
                "");
  static_assert(std::is_same<typename ClusterOffsetView::value_type, int>{},
                "");

  using MemorySpace = typename Primitives::memory_space;
  static_assert(
      std::is_same<typename ClusterIndicesView::memory_space, MemorySpace>{},
      "");
  static_assert(
      std::is_same<typename ClusterOffsetView::memory_space, MemorySpace>{},
      "");

  ARBORX_ASSERT(core_min_size >= 1);
  ARBORX_ASSERT(cluster_min_size >= 2);

  Kokkos::Profiling::pushRegion("ArborX::DBSCAN");

  using clock = std::chrono::high_resolution_clock;

  clock::time_point start_total;
  clock::time_point start;
  clock::time_point start_local;
  std::chrono::duration<double> elapsed_construction;
  std::chrono::duration<double> elapsed_stat;
  std::chrono::duration<double> elapsed_neigh;
  std::chrono::duration<double> elapsed_query;
  std::chrono::duration<double> elapsed_cluster;
  std::chrono::duration<double> elapsed_total;
  std::chrono::duration<double> elapsed_verify = clock::duration::zero();

  start_total = clock::now();

  auto const predicates = buildPredicates(primitives, eps);

  int const n = primitives.extent_int(0);

  // Build the tree
  start = clock::now();
  Kokkos::Profiling::pushRegion("ArborX::DBSCAN::tree_construction");
  ArborX::BVH<MemorySpace> bvh(exec_space, primitives);
  Kokkos::Profiling::popRegion();
  elapsed_construction = clock::now() - start;

  start = clock::now();
  Kokkos::Profiling::pushRegion("ArborX::DBSCAN::clusters");

  Kokkos::View<int *, MemorySpace> stat(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "ArborX::DBSCAN::stat"),
      n);
  ArborX::iota(exec_space, stat);
  if (core_min_size == 1)
  {
    // Perform the queries and build clusters through callback
    using CorePoints = CCSCorePoints;
    CorePoints core_points;
    Kokkos::Profiling::pushRegion("ArborX::DBSCAN::clusters::query");
    bvh.query(
        exec_space, predicates,
        Details::DBSCANCallback<MemorySpace, CorePoints>{stat, core_points});
    Kokkos::Profiling::popRegion();
  }
  else
  {
    // Determine core points
    start_local = clock::now();
    Kokkos::Profiling::pushRegion("ArborX::DBSCAN::clusters::num_neigh");
    Kokkos::View<int *, MemorySpace> num_neigh(
        Kokkos::ViewAllocateWithoutInitializing(
            "ArborX::DBSCAN::num_neighbors"),
        n);
    // Initialize to -1 as we don't want to count ourselves as a neighbor
    Kokkos::deep_copy(num_neigh, -1);
    bvh.query(exec_space, predicates,
              NumNeighEarlyExitCallback<MemorySpace>{num_neigh, core_min_size});
    Kokkos::Profiling::popRegion();
    elapsed_neigh = clock::now() - start_local;

    using CorePoints = DBSCANCorePoints<MemorySpace>;

    start_local = clock::now();
    Kokkos::Profiling::pushRegion("ArborX::DBSCAN::clusters:query");
    bvh.query(exec_space, predicates,
              Details::DBSCANCallback<MemorySpace, CorePoints>{
                  stat, CorePoints{num_neigh, core_min_size}});
    Kokkos::Profiling::popRegion();
    elapsed_query = clock::now() - start_local;
  }

  // Per [1]:
  //
  // ```
  // The finalization kernel will, ultimately, make all parents
  // point directly to the representative.
  // ```
  Kokkos::parallel_for("ArborX::DBSCAN::flatten_stat",
                       Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                       KOKKOS_LAMBDA(int const i) {
                         // ##### ECL license (see LICENSE.ECL) #####
                         int next;
                         int vstat = stat(i);
                         int const old = vstat;
                         while (vstat > (next = stat(vstat)))
                         {
                           vstat = next;
                         }
                         if (vstat != old)
                           stat(i) = vstat;
                       });
  Kokkos::Profiling::popRegion();
  elapsed_stat = clock::now() - start;

  // Use new name to clearly demonstrate the meaning of this view from now on
  auto clusters = stat;

  elapsed_total += clock::now() - start_total;
  if (verify)
  {
    start = clock::now();
    Kokkos::Profiling::pushRegion("ArborX::DBSCAN::verify");

    Kokkos::View<int *, MemorySpace> indices("indices", 0);
    Kokkos::View<int *, MemorySpace> offset("offset", 0);
    bvh.query(exec_space, predicates, indices, offset);

    auto passed = Details::verifyClusters(exec_space, indices, offset, clusters,
                                          core_min_size);
    printf("Verification %s\n", (passed ? "passed" : "failed"));

    Kokkos::Profiling::popRegion();
    elapsed_verify = clock::now() - start;
  }
  start_total = clock::now();

  // find clusters
  start = clock::now();
  Kokkos::Profiling::pushRegion("ArborX::DBSCAN::sort_and_filter_clusters");

  // sort clusters and compute permutation
  auto permute = Details::sortObjects(exec_space, clusters);

  reallocWithoutInitializing(cluster_offset, n + 1);
  Kokkos::View<int *, MemorySpace> cluster_starts(
      Kokkos::ViewAllocateWithoutInitializing("ArborX::DBSCAN::cluster_starts"),
      n);
  int num_clusters = 0;
  // In the following scan, we locate the starting position (stored in
  // cluster_starts) and size (stored in cluster_offset) of each valid cluster
  // (i.e., of size >= cluster_min_size). For every index i, we check whether
  // its CC index is different from the previous one (this indicates a start of
  // connected component) and whether the CC index of i + cluster_min_size is
  // the same (this indicates that this CC is at least of cluster_min_size
  // size). If those are true, we do a linear search from i + cluster_min_size
  // till next CC index change to find the CC size.
  Kokkos::parallel_scan(
      "ArborX::DBSCAN::compute_cluster_starts_and_sizes",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int i, int &update, bool final_pass) {
        bool const is_cluster_first_index =
            (i == 0 || clusters(i) != clusters(i - 1));
        bool const is_cluster_large_enough =
            ((i + cluster_min_size - 1 < n) &&
             (clusters(i + cluster_min_size - 1) == clusters(i)));
        if (is_cluster_first_index && is_cluster_large_enough)
        {
          if (final_pass)
          {
            cluster_starts(update) = i;
            int end = i + cluster_min_size - 1;
            while (++end < n && clusters(end) == clusters(i))
              ; // do nothing
            cluster_offset(update) = end - i;
          }
          ++update;
        }
      },
      num_clusters);
  Kokkos::resize(cluster_offset, num_clusters + 1);
  exclusivePrefixSum(exec_space, cluster_offset);

  // Construct cluster indices
  reallocWithoutInitializing(cluster_indices, lastElement(cluster_offset));
  Kokkos::parallel_for(
      "ArborX::DBSCAN::populate_clusters",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_clusters),
      KOKKOS_LAMBDA(int i) {
        for (int k = cluster_offset(i); k < cluster_offset(i + 1); ++k)
        {
          cluster_indices(k) =
              permute(cluster_starts(i) + (k - cluster_offset(i)));
        }
      });
  Kokkos::Profiling::popRegion();
  elapsed_cluster = clock::now() - start;

  elapsed_total += clock::now() - start_total;

  if (verbose)
  {
    printf("total time          : %10.3f\n", elapsed_total.count());
    printf("-- construction     : %10.3f\n", elapsed_construction.count());
    printf("-- query+cluster    : %10.3f\n", elapsed_stat.count());
    if (core_min_size > 1)
    {
      printf("---- neigh          : %10.3f\n", elapsed_neigh.count());
      printf("---- query          : %10.3f\n", elapsed_query.count());
    }
    printf("-- postprocess      : %10.3f\n", elapsed_cluster.count());
    if (verify)
      printf("verify              : %10.3f\n", elapsed_verify.count());
  }

  Kokkos::Profiling::popRegion();
}

} // namespace DBSCAN
} // namespace ArborX

#endif
