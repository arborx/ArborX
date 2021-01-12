/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
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

#include <map>

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

  Kokkos::Timer timer_total;
  Kokkos::Timer timer;
  std::map<std::string, double> elapsed;

  auto timer_start = [&exec_space, verbose](Kokkos::Timer &timer) {
    if (verbose)
      exec_space.fence();
    timer.reset();
  };
  auto timer_seconds = [&exec_space, verbose](Kokkos::Timer const &timer) {
    if (verbose)
      exec_space.fence();
    return timer.seconds();
  };

  timer_start(timer_total);

  auto const predicates = buildPredicates(primitives, eps);

  int const n = primitives.extent_int(0);

  // Build the tree
  timer_start(timer);
  Kokkos::Profiling::pushRegion("ArborX::DBSCAN::tree_construction");
  ArborX::BVH<MemorySpace> bvh(exec_space, primitives);
  Kokkos::Profiling::popRegion();
  elapsed["construction"] = timer_seconds(timer);

  timer_start(timer);
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
    Kokkos::Timer timer_local;
    timer_start(timer_local);
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
    elapsed["neigh"] = timer_seconds(timer_local);

    using CorePoints = DBSCANCorePoints<MemorySpace>;

    // Perform the queries and build clusters through callback
    timer_start(timer_local);
    Kokkos::Profiling::pushRegion("ArborX::DBSCAN::clusters:query");
    bvh.query(exec_space, predicates,
              Details::DBSCANCallback<MemorySpace, CorePoints>{
                  stat, CorePoints{num_neigh, core_min_size}});
    Kokkos::Profiling::popRegion();
    elapsed["query"] = timer_seconds(timer_local);
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
  elapsed["query+cluster"] = timer_seconds(timer);

  // Use new name to clearly demonstrate the meaning of this view from now on
  auto const &clusters = stat;

  if (verify)
  {
    timer_start(timer);
    Kokkos::Profiling::pushRegion("ArborX::DBSCAN::verify");

    Kokkos::View<int *, MemorySpace> indices("ArborX::DBSCAN::indices", 0);
    Kokkos::View<int *, MemorySpace> offset("ArborX::DBSCAN::offset", 0);
    ArborX::query(bvh, exec_space, predicates, indices, offset);

    auto passed = Details::verifyClusters(exec_space, indices, offset, clusters,
                                          core_min_size);
    printf("Verification %s\n", (passed ? "passed" : "failed"));

    Kokkos::Profiling::popRegion();
    elapsed["verify"] = timer_seconds(timer);
  }

  // find clusters
  timer_start(timer);
  Kokkos::Profiling::pushRegion("ArborX::DBSCAN::sort_and_filter_clusters");

  Kokkos::View<int *, MemorySpace> cluster_sizes(
      "ArborX::DBSCAN::cluster_sizes", n);
  Kokkos::parallel_for("ArborX::DBSCAN::compute_cluster_sizes",
                       Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                       KOKKOS_LAMBDA(int const i) {
                         Kokkos::atomic_fetch_add(&cluster_sizes(clusters(i)),
                                                  1);
                       });

  // This kernel serves dual purpose:
  // - it constructs an offset array through exclusive prefix sum, with a
  //   caveat that small clusters (of size < cluster_min_size) are filtered out
  // - it creates a mapping from a cluster index into the cluster's position in
  //   the offset array
  // We reuse the cluster_sizes array for the second, creating a new alias for
  // it for clarity.
  auto &map_cluster_to_offset_position = cluster_sizes;
  int constexpr IGNORED_CLUSTER = -1;
  int num_clusters;
  reallocWithoutInitializing(cluster_offset, n + 1);
  Kokkos::parallel_scan(
      "ArborX::DBSCAN::compute_cluster_offset_with_filter",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int const i, int &update, bool final_pass) {
        bool is_cluster_too_small = (cluster_sizes(i) < cluster_min_size);
        if (!is_cluster_too_small)
        {
          if (final_pass)
          {
            cluster_offset(update) = cluster_sizes(i);
            map_cluster_to_offset_position(i) = update;
          }
          ++update;
        }
        else
        {
          if (final_pass)
            map_cluster_to_offset_position(i) = IGNORED_CLUSTER;
        }
      },
      num_clusters);
  Kokkos::resize(Kokkos::WithoutInitializing, cluster_offset, num_clusters + 1);
  exclusivePrefixSum(exec_space, cluster_offset);

  auto cluster_starts = clone(exec_space, cluster_offset);
  reallocWithoutInitializing(cluster_indices, lastElement(cluster_offset));
  Kokkos::parallel_for("ArborX::DBSCAN::compute_cluster_indices",
                       Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                       KOKKOS_LAMBDA(int const i) {
                         auto offset_pos =
                             map_cluster_to_offset_position(clusters(i));
                         if (offset_pos != IGNORED_CLUSTER)
                         {
                           auto position = Kokkos::atomic_fetch_add(
                               &cluster_starts(offset_pos), 1);
                           cluster_indices(position) = i;
                         }
                       });

  Kokkos::Profiling::popRegion();
  elapsed["cluster"] = timer_seconds(timer);

  elapsed["total"] = timer_seconds(timer_total) - elapsed["verify"];

  if (verbose)
  {
    printf("total time          : %10.3f\n", elapsed["total"]);
    printf("-- construction     : %10.3f\n", elapsed["construction"]);
    printf("-- query+cluster    : %10.3f\n", elapsed["query+cluster"]);
    if (core_min_size > 1)
    {
      printf("---- neigh          : %10.3f\n", elapsed["neigh"]);
      printf("---- query          : %10.3f\n", elapsed["query"]);
    }
    printf("-- postprocess      : %10.3f\n", elapsed["cluster"]);
    if (verify)
      printf("verify              : %10.3f\n", elapsed["verify"]);
  }

  Kokkos::Profiling::popRegion();
}

} // namespace DBSCAN
} // namespace ArborX

#endif
