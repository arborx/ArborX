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

#ifndef ARBORX_HDBSCAN_HPP
#define ARBORX_HDBSCAN_HPP

#include <ArborX_DetailsHDBSCAN.hpp>
#include <ArborX_DetailsMutualReachabilityDistance.hpp>
#include <ArborX_DetailsTreeNodeLabeling.hpp>
#include <ArborX_LinearBVH.hpp>

#include <iomanip>
#include <limits>
#include <map>

namespace ArborX
{

namespace HDBSCAN
{

struct Parameters
{
  bool _verbose = false;

  Parameters &setVerbose(bool verbose)
  {
    _verbose = verbose;
    return *this;
  }
};

} // namespace HDBSCAN

template <typename ExecutionSpace, typename Primitives>
void hdbscan(ExecutionSpace const &exec_space, Primitives const &primitives,
             int core_min_size,
             HDBSCAN::Parameters const &parameters = HDBSCAN::Parameters())
{
  using MemorySpace = typename Primitives::memory_space;

  Kokkos::Profiling::pushRegion("ArborX::HDBSCAN");

  // Right now, we use the same minpts for computing core distances as well as
  // minimum cluster size. For the latter, minpts = 2 is special in that it
  // requires introducing a self-loops at the MST level to compute cluster
  // stability. To simplify our life, we disallow this case, and require
  // minpts > 2.
  ARBORX_ASSERT(core_min_size > 2);

  int const n = primitives.extent_int(0);
  bool const verbose = parameters._verbose;

  auto NoInit = [](std::string const &label) {
    return Kokkos::view_alloc(Kokkos::WithoutInitializing, label);
  };

  Kokkos::Profiling::pushRegion("ArborX::HDBSCAN::tree_construction");

  // Build the search index
  ArborX::BVH<MemorySpace> bvh(exec_space, primitives);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::HDBSCAN::core_distances");

  // Compute core distances
  Kokkos::View<float *, MemorySpace> core_distances(
      NoInit("ArborX::HDBSCAN::core_distances"), n);
  constexpr auto inf = KokkosExt::ArithmeticTraits::infinity<float>::value;
  Kokkos::deep_copy(exec_space, core_distances, -inf);
  bvh.query(exec_space,
            Details::NearestK<Primitives>{primitives, core_min_size},
            ArborX::Details::MaxDistance<Primitives, decltype(core_distances)>{
                primitives, core_distances});

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::HDBSCAN::boruvka");

  Kokkos::View<int *, MemorySpace> cached_closest_neighbors(
      NoInit("ArborX::HDBSCAN::neighbors"), n);
  ArborX::iota(exec_space, cached_closest_neighbors);

  Kokkos::View<int *, MemorySpace> labels(NoInit("ArborX::HDBSCAN::labels"), n);
  ArborX::iota(exec_space, labels);

  Kokkos::View<Kokkos::pair<int, int> *, MemorySpace> components_out_edges(
      NoInit("ArborX::HDBSCAN::components_out_edges"), n);

  Kokkos::View<Kokkos::pair<int, int> *, MemorySpace> mst_edges(
      NoInit("ArborX::HDBSCAN::mst_edges"), n - 1);

  // Compute parents
  Kokkos::View<int *, MemorySpace> bvh_parents(
      NoInit("ArborX::HDBSCAN::parents"), 2 * n - 1);
  Details::findParents(exec_space, bvh, bvh_parents);

  using Predicates = Details::NearestK<Primitives>;
  Predicates predicates{primitives, 1};

  Kokkos::Profiling::pushRegion("ArborX::HDBSCAN::compute_permutation");
  // Permute predicates once for all Boruvka iterations
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
  auto permute =
      Details::BatchedQueries<DeviceType>::sortQueriesAlongZOrderCurve(
          exec_space, static_cast<Box>(bvh.bounds()), predicates);
  Kokkos::Profiling::popRegion();

  using PermutedPredicates =
      Details::PermutedData<Predicates, decltype(permute)>;
  PermutedPredicates permuted_predicates{predicates, permute};

  // Boruvka loop
  int it = 0;
  int num_components = n;
  float boruvka_time = 0.f;
  while (num_components > 1)
  {
    Kokkos::Timer timer_boruvka_iteration;
    if (verbose)
    {
      printf("#%4d: [%10d]", ++it, num_components);
      exec_space.fence();
      timer_boruvka_iteration.reset();
    }

    Details::determineComponentEdges(exec_space, permuted_predicates, bvh,
                                     bvh_parents, labels, core_distances,
                                     components_out_edges);

    num_components = Details::updateLabels(
        exec_space, num_components, components_out_edges, labels, mst_edges);

    exec_space.fence();
    if (verbose)
    {
      exec_space.fence();
      auto time_iteration = timer_boruvka_iteration.seconds();
      boruvka_time += time_iteration;
      printf("  time : %10.3f\n", time_iteration);
    }
  }
  if (verbose)
  {
    printf("#%4d: [%10d]\n", ++it, num_components);
    printf("Total Boruvka time  : %10.3f\n", boruvka_time);
  }

  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::popRegion();
}

} // namespace ArborX

#endif
