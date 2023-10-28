/****************************************************************************
 * Copyright (c) 2017-2023 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "ArborXTest_StdVectorToKokkosView.hpp"
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_DetailsMutualReachabilityDistance.hpp>
#include <ArborX_LinearBVH.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

namespace Test
{
using ArborXTest::toView;

template <class ExecutionSpace>
auto compute_core_distances(ExecutionSpace exec_space,
                            std::vector<ArborX::Point> const &points_host,
                            int k)
{
  auto points = toView<ExecutionSpace>(points_host, "Test::points");

  ARBORX_ASSERT(points.extent_int(0) >= k);
  using MemorySpace = typename ExecutionSpace::memory_space;
  ArborX::BoundingVolumeHierarchy<
      MemorySpace, ArborX::Details::PairIndexVolume<ArborX::Point>>
      bvh{exec_space,
          ArborX::Details::LegacyValues<decltype(points), ArborX::Point>{
              points}};
  Kokkos::View<float *, MemorySpace> distances(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Test::core_distances"),
      bvh.size());
  constexpr auto inf = KokkosExt::ArithmeticTraits::infinity<float>::value;
  Kokkos::deep_copy(exec_space, distances, -inf);
  bvh.query(exec_space, ArborX::Details::NearestK<decltype(points)>{points, k},
            ArborX::Details::MaxDistance<decltype(points), decltype(distances)>{
                points, distances});

  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, distances);
}

#define ARBORX_TEST_COMPUTE_CORE_DISTANCES(exec_space, points, k, ref)         \
  BOOST_TEST(Test::compute_core_distances(exec_space, points, k) == ref,       \
             boost::test_tools::per_element())

template <class ExecutionSpace>
auto compute_mutual_reachability_distances(
    ExecutionSpace exec_space, std::vector<float> const &core_distances_host,
    std::vector<Kokkos::pair<int, int>> const &edges_host,
    std::vector<float> const &distances_host)
    -> Kokkos::View<float *, Kokkos::HostSpace>
{
  auto core_distances =
      toView<ExecutionSpace>(core_distances_host, "Test::core_distances");
  auto edges = toView<ExecutionSpace>(edges_host, "Test::edges");
  auto distances = toView<ExecutionSpace>(distances_host, "Test::distances");

  auto const n = edges.size();
  Kokkos::View<float *, ExecutionSpace> mutual_reachability_distances(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "Test::mutual_reachability_distances"),
      n);
  ArborX::Details::MutualReachability<decltype(core_distances)> const
      distance_mutual_reach{core_distances};
  Kokkos::parallel_for(
      "Test::compute_mutual_reachability_distances",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int i) {
        mutual_reachability_distances(i) = distance_mutual_reach(
            edges(i).first, edges(i).second, distances(i));
      });

  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                             mutual_reachability_distances);
}

#define ARBORX_TEST_COMPUTE_MUTUAL_REACHABILITY_DISTANCES(                     \
    exec_space, core_distances, edges, distances, ref)                         \
  BOOST_TEST(Test::compute_mutual_reachability_distances(                      \
                 exec_space, core_distances, edges, distances) == ref,         \
             boost::test_tools::per_element())

} // namespace Test

BOOST_AUTO_TEST_SUITE(MutualReachabilityDistance)

BOOST_AUTO_TEST_CASE_TEMPLATE(compute_core_distances, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace exec_space;

  std::vector<ArborX::Point> points{
      {0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0},
  };
  ARBORX_TEST_COMPUTE_CORE_DISTANCES(exec_space, points, 1,
                                     (std::vector<float>{0, 0, 0, 0, 0}));
  ARBORX_TEST_COMPUTE_CORE_DISTANCES(exec_space, points, 2,
                                     (std::vector<float>{1, 1, 1, 1, 1}));
  ARBORX_TEST_COMPUTE_CORE_DISTANCES(exec_space, points, 3,
                                     (std::vector<float>{2, 1, 1, 1, 2}));
  ARBORX_TEST_COMPUTE_CORE_DISTANCES(exec_space, points, 4,
                                     (std::vector<float>{3, 2, 2, 2, 3}));
  ARBORX_TEST_COMPUTE_CORE_DISTANCES(exec_space, points, 5,
                                     (std::vector<float>{4, 3, 2, 3, 4}));

  std::vector<ArborX::Point> non_equidistant_points{
      {0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {6, 0, 0}, {10, 0, 0},
  };
  ARBORX_TEST_COMPUTE_CORE_DISTANCES(exec_space, non_equidistant_points, 2,
                                     (std::vector<float>{1, 1, 1, 1, 3, 4}));
  ARBORX_TEST_COMPUTE_CORE_DISTANCES(exec_space, non_equidistant_points, 3,
                                     (std::vector<float>{2, 1, 1, 2, 4, 7}));
  ARBORX_TEST_COMPUTE_CORE_DISTANCES(exec_space, non_equidistant_points, 4,
                                     (std::vector<float>{3, 2, 2, 3, 4, 8}));
  ARBORX_TEST_COMPUTE_CORE_DISTANCES(exec_space, non_equidistant_points, 5,
                                     (std::vector<float>{6, 5, 4, 3, 5, 9}));
  ARBORX_TEST_COMPUTE_CORE_DISTANCES(exec_space, non_equidistant_points, 6,
                                     (std::vector<float>{10, 9, 8, 7, 6, 10}));
}
BOOST_AUTO_TEST_CASE_TEMPLATE(compute_mutual_reachability_distances, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace exec_space;

  { // equidistant points
    // 0     1     2     3     4
    //[0]   [1]   [2]   [3]   [4]
    std::vector<Kokkos::pair<int, int>> edges{{0, 1}, {0, 2}, {0, 3}, {0, 4},
                                              {1, 2}, {1, 3}, {1, 4}, {2, 3},
                                              {2, 4}, {3, 4}};
    std::vector<float> distances{1, 2, 3, 4, 1, 2, 3, 1, 2, 1};

    ARBORX_TEST_COMPUTE_MUTUAL_REACHABILITY_DISTANCES(
        exec_space, (std::vector<float>{1, 1, 1, 1, 1}), edges, distances,
        (std::vector<float>{1, 2, 3, 4, 1, 2, 3, 1, 2, 1}));

    ARBORX_TEST_COMPUTE_MUTUAL_REACHABILITY_DISTANCES(
        exec_space, (std::vector<float>{2, 1, 1, 1, 2}), edges, distances,
        (std::vector<float>{2, 2, 3, 4, 1, 2, 3, 1, 2, 2}));

    ARBORX_TEST_COMPUTE_MUTUAL_REACHABILITY_DISTANCES(
        exec_space, (std::vector<float>{3, 2, 2, 2, 3}), edges, distances,
        (std::vector<float>{3, 3, 3, 4, 2, 2, 3, 2, 3, 3}));

    ARBORX_TEST_COMPUTE_MUTUAL_REACHABILITY_DISTANCES(
        exec_space, (std::vector<float>{4, 3, 2, 3, 4}), edges, distances,
        (std::vector<float>{4, 4, 4, 4, 3, 3, 4, 3, 4, 4}));
  }
  { // non-equidistant points
    // 0   1   2   3   4   5   6   7   8   9   10
    //[0] [1] [2] [3]         [4]             [5]
    std::vector<Kokkos::pair<int, int>> edges{
        {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {1, 2}, {1, 3}, {1, 4},
        {1, 5}, {2, 3}, {2, 4}, {2, 5}, {3, 4}, {3, 5}, {4, 5}};
    std::vector<float> distances{1, 2, 3, 6, 10, 1, 2, 5, 9, 1, 4, 8, 3, 7, 4};

    ARBORX_TEST_COMPUTE_MUTUAL_REACHABILITY_DISTANCES(
        exec_space, (std::vector<float>{1, 1, 1, 1, 3, 4}), edges, distances,
        (std::vector<float>{1, 2, 3, 6, 10, 1, 2, 5, 9, 1, 4, 8, 3, 7, 4}));

    ARBORX_TEST_COMPUTE_MUTUAL_REACHABILITY_DISTANCES(
        exec_space, (std::vector<float>{2, 1, 1, 2, 4, 7}), edges, distances,
        (std::vector<float>{2, 2, 3, 6, 10, 1, 2, 5, 9, 2, 4, 8, 4, 7, 7}));

    ARBORX_TEST_COMPUTE_MUTUAL_REACHABILITY_DISTANCES(
        exec_space, (std::vector<float>{3, 2, 2, 3, 4, 8}), edges, distances,
        (std::vector<float>{3, 3, 3, 6, 10, 2, 3, 5, 9, 3, 4, 8, 4, 8, 8}));

    ARBORX_TEST_COMPUTE_MUTUAL_REACHABILITY_DISTANCES(
        exec_space, (std::vector<float>{6, 5, 4, 3, 5, 9}), edges, distances,
        (std::vector<float>{6, 6, 6, 6, 10, 5, 5, 5, 9, 4, 5, 9, 5, 9, 9}));

    ARBORX_TEST_COMPUTE_MUTUAL_REACHABILITY_DISTANCES(
        exec_space, (std::vector<float>{10, 9, 8, 7, 6, 10}), edges, distances,
        (std::vector<float>{10, 10, 10, 10, 10, 9, 9, 9, 10, 8, 8, 10, 7, 10,
                            10}));
  }
}

BOOST_AUTO_TEST_SUITE_END()
