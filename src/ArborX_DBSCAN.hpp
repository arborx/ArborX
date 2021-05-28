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

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsFDBSCAN.hpp>
#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_LinearBVH.hpp>

#include <map>

namespace ArborX
{

template <typename Primitives>
struct PrimitivesWithRadius
{
  Primitives _primitives;
  double _r;
};

template <typename Primitives>
auto buildPredicates(Primitives const &v, double r)
{
  return PrimitivesWithRadius<Primitives>{v, r};
}

template <typename Primitives>
struct AccessTraits<PrimitivesWithRadius<Primitives>, PredicatesTag>
{
  using PrimitivesAccess = AccessTraits<Primitives, PrimitivesTag>;

  using memory_space = typename PrimitivesAccess::memory_space;
  using Predicates = PrimitivesWithRadius<Primitives>;

  static size_t size(Predicates const &w)
  {
    return PrimitivesAccess::size(w._primitives);
  }
  static KOKKOS_FUNCTION auto get(Predicates const &w, size_t i)
  {
    return attach(
        intersects(Sphere{PrimitivesAccess::get(w._primitives, i), w._r}),
        (int)i);
  }
};

namespace DBSCAN
{

struct CCSCorePoints
{
  KOKKOS_FUNCTION bool operator()(int) const { return true; }
};

template <typename MemorySpace>
struct DBSCANCorePoints
{
  Kokkos::View<int *, MemorySpace> _num_neigh;
  int _core_min_size;

  KOKKOS_FUNCTION bool operator()(int const i) const
  {
    return _num_neigh(i) >= _core_min_size;
  }
};

struct Parameters
{
  // Print timers to standard output
  bool _print_timers = false;

  Parameters &setPrintTimers(bool print_timers)
  {
    _print_timers = print_timers;
    return *this;
  }
};

} // namespace DBSCAN

template <typename ExecutionSpace, typename Primitives>
Kokkos::View<int *,
             typename AccessTraits<Primitives, PrimitivesTag>::memory_space>
dbscan(ExecutionSpace const &exec_space, Primitives const &primitives,
       float eps, int core_min_size,
       DBSCAN::Parameters const &parameters = DBSCAN::Parameters())
{
  Kokkos::Profiling::pushRegion("ArborX::dbscan");

  using Access = AccessTraits<Primitives, PrimitivesTag>;
  using MemorySpace = typename Access::memory_space;

  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value,
      "Primitives must be accessible from the execution space");

  ARBORX_ASSERT(eps > 0);
  ARBORX_ASSERT(core_min_size >= 2);

  bool const is_special_case = (core_min_size == 2);

  Kokkos::Timer timer;
  std::map<std::string, double> elapsed;

  bool const verbose = parameters._print_timers;
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

  auto const predicates = buildPredicates(primitives, eps);

  auto const n = Access::size(primitives);

  // Build the tree
  timer_start(timer);
  Kokkos::Profiling::pushRegion("ArborX::dbscan::tree_construction");
  ArborX::BVH<MemorySpace> bvh(exec_space, primitives);
  Kokkos::Profiling::popRegion();
  elapsed["construction"] = timer_seconds(timer);

  timer_start(timer);
  Kokkos::Profiling::pushRegion("ArborX::dbscan::clusters");

  Kokkos::View<int *, MemorySpace> num_neigh("ArborX::dbscan::num_neighbors",
                                             0);

  Kokkos::View<int *, MemorySpace> labels(
      Kokkos::ViewAllocateWithoutInitializing("ArborX::DBSCAN::labels"), n);
  ArborX::iota(exec_space, labels);
  if (is_special_case)
  {
    // Perform the queries and build clusters through callback
    using CorePoints = DBSCAN::CCSCorePoints;
    CorePoints core_points;
    Kokkos::Profiling::pushRegion("ArborX::dbscan::clusters::query");
    bvh.query(
        exec_space, predicates,
        Details::FDBSCANCallback<MemorySpace, CorePoints>{labels, core_points});
    Kokkos::Profiling::popRegion();
  }
  else
  {
    // Determine core points
    Kokkos::Timer timer_local;
    timer_start(timer_local);
    Kokkos::Profiling::pushRegion("ArborX::dbscan::clusters::num_neigh");
    Kokkos::resize(num_neigh, n);
    bvh.query(exec_space, predicates,
              Details::CountUpToN<MemorySpace>{num_neigh, core_min_size});
    Kokkos::Profiling::popRegion();
    elapsed["neigh"] = timer_seconds(timer_local);

    using CorePoints = DBSCAN::DBSCANCorePoints<MemorySpace>;

    // Perform the queries and build clusters through callback
    timer_start(timer_local);
    Kokkos::Profiling::pushRegion("ArborX::dbscan::clusters:query");
    bvh.query(exec_space, predicates,
              Details::FDBSCANCallback<MemorySpace, CorePoints>{
                  labels, CorePoints{num_neigh, core_min_size}});
    Kokkos::Profiling::popRegion();
    elapsed["query"] = timer_seconds(timer_local);
  }

  // Per [1]:
  //
  // ```
  // The finalization kernel will, ultimately, make all parents
  // point directly to the representative.
  // ```
  Kokkos::View<int *, MemorySpace> cluster_sizes(
      "ArborX::dbscan::cluster_sizes", n);
  Kokkos::parallel_for("ArborX::dbscan::finalize_labels",
                       Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                       KOKKOS_LAMBDA(int const i) {
                         // ##### ECL license (see LICENSE.ECL) #####
                         int next;
                         int vstat = labels(i);
                         int const old = vstat;
                         while (vstat > (next = labels(vstat)))
                         {
                           vstat = next;
                         }
                         if (vstat != old)
                           labels(i) = vstat;

                         Kokkos::atomic_fetch_add(&cluster_sizes(labels(i)), 1);
                       });
  if (is_special_case)
  {
    // Ideally, this kernel would have had the exactly same form as in the
    // else() clause. But there's no available valid is_core() for use here:
    // - CCSCorePoints cannot be used as it always returns true, which is OK
    //   inside the callback, but not here
    // - DBSCANCorePoints cannot be used either as num_neigh is not initialized
    //   in the special case.
    Kokkos::parallel_for("ArborX::dbscan::mark_noise",
                         Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                         KOKKOS_LAMBDA(int const i) {
                           if (cluster_sizes(labels(i)) == 1)
                             labels(i) = -1;
                         });
  }
  else
  {
    DBSCAN::DBSCANCorePoints<MemorySpace> is_core{num_neigh, core_min_size};
    Kokkos::parallel_for("ArborX::dbscan::mark_noise",
                         Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                         KOKKOS_LAMBDA(int const i) {
                           if (cluster_sizes(labels(i)) == 1 && !is_core(i))
                             labels(i) = -1;
                         });
  }
  Kokkos::Profiling::popRegion();
  elapsed["query+cluster"] = timer_seconds(timer);

  if (verbose)
  {
    printf("-- construction     : %10.3f\n", elapsed["construction"]);
    printf("-- query+cluster    : %10.3f\n", elapsed["query+cluster"]);
    if (!is_special_case)
    {
      printf("---- neigh          : %10.3f\n", elapsed["neigh"]);
      printf("---- query          : %10.3f\n", elapsed["query"]);
    }
  }

  Kokkos::Profiling::popRegion();

  return labels;
}

} // namespace ArborX

#endif
