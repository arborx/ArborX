/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
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
#include <ArborX_MinimumSpanningTree.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

namespace ArborX
{
namespace Details
{
// NOTE not sure why but wasn't detected when defined in the global namespace
inline constexpr bool operator==(WeightedEdge const &lhs,
                                 WeightedEdge const &rhs)
{
  return !(lhs < rhs) && !(rhs < lhs);
}
} // namespace Details
} // namespace ArborX

void test_weighted_edges_comparison_compile_only()
{
  using ArborX::Details::WeightedEdge;
  static_assert(WeightedEdge{1, 2, 3} == WeightedEdge{1, 2, 3});
  static_assert(WeightedEdge{1, 2, 3} == WeightedEdge{2, 1, 3});
}

template <>
struct boost::test_tools::tt_detail::print_log_value<
    ArborX::Details::WeightedEdge>
{
  void operator()(std::ostream &os, ArborX::Details::WeightedEdge const &e)
  {
    os << e.source << " -> " << e.target << " [weight=" << e.weight << "]";
  }
};

namespace Test
{
using ArborXTest::toView;

template <class ExecutionSpace>
auto build_minimum_spanning_tree(ExecutionSpace const &exec_space,
                                 std::vector<ArborX::Point> const &points_host,
                                 int k)
{
  auto points = toView<ExecutionSpace>(points_host, "Test::points");

  using MemorySpace = typename ExecutionSpace::memory_space;
  ArborX::Details::MinimumSpanningTree<MemorySpace> mst{exec_space, points, k};

  auto edges_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mst.edges);
  std::sort(edges_host.data(), edges_host.data() + edges_host.size());
  return edges_host;
}

template <class T>
std::vector<T> sorted(std::vector<T> v)
{
  std::sort(v.begin(), v.end());
  return v;
}

#define ARBORX_TEST_MINIMUM_SPANNING_TREE(exec_space, points, k, ref)          \
  BOOST_TEST(Test::build_minimum_spanning_tree(exec_space, points, k) ==       \
                 Test::sorted(ref),                                            \
             boost::test_tools::per_element());

} // namespace Test

BOOST_AUTO_TEST_CASE_TEMPLATE(minimum_spanning_tree, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace exec_space;

  { // equidistant points
    // 0     1     2     3     4
    //[0]   [1]   [2]   [3]   [4]
    std::vector<ArborX::Point> points{
        {0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {4, 0, 0},
    };

    ARBORX_TEST_MINIMUM_SPANNING_TREE(
        exec_space, points, 1,
        (std::vector<ArborX::Details::WeightedEdge>{
            {0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 4, 1}}));

    ARBORX_TEST_MINIMUM_SPANNING_TREE(
        exec_space, points, 2,
        (std::vector<ArborX::Details::WeightedEdge>{
            {0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 4, 1}}));

    ARBORX_TEST_MINIMUM_SPANNING_TREE(
        exec_space, points, 3,
        (std::vector<ArborX::Details::WeightedEdge>{
            {0, 1, 2}, {1, 2, 1}, {2, 3, 1}, {2, 4, 2}}));

    ARBORX_TEST_MINIMUM_SPANNING_TREE(
        exec_space, points, 4,
        (std::vector<ArborX::Details::WeightedEdge>{
            {0, 1, 3}, {1, 2, 2}, {1, 3, 2}, {1, 4, 3}}));

    ARBORX_TEST_MINIMUM_SPANNING_TREE(
        exec_space, points, 5,
        (std::vector<ArborX::Details::WeightedEdge>{
            {0, 1, 4}, {1, 2, 3}, {1, 3, 3}, {0, 4, 4}}));
  }
  { // non-equidistant points
    // 0   1   2   3   4   5   6   7   8   9   10
    //[0] [1] [2] [3]         [4]             [5]
    std::vector<ArborX::Point> non_equidistant_points{
        {0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {3, 0, 0}, {6, 0, 0}, {10, 0, 0},
    };

    ARBORX_TEST_MINIMUM_SPANNING_TREE(
        exec_space, non_equidistant_points, 1,
        (std::vector<ArborX::Details::WeightedEdge>{
            {0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 4, 3}, {4, 5, 4}}));

    ARBORX_TEST_MINIMUM_SPANNING_TREE(
        exec_space, non_equidistant_points, 2,
        (std::vector<ArborX::Details::WeightedEdge>{
            {0, 1, 1}, {1, 2, 1}, {2, 3, 1}, {3, 4, 3}, {4, 5, 4}}));

    ARBORX_TEST_MINIMUM_SPANNING_TREE(
        exec_space, non_equidistant_points, 3,
        (std::vector<ArborX::Details::WeightedEdge>{
            {0, 1, 2}, {1, 2, 1}, {1, 3, 2}, {2, 4, 4}, {3, 5, 7}}));

    ARBORX_TEST_MINIMUM_SPANNING_TREE(
        exec_space, non_equidistant_points, 4,
        (std::vector<ArborX::Details::WeightedEdge>{
            {0, 1, 3}, {1, 2, 2}, {0, 3, 3}, {2, 4, 4}, {2, 5, 8}}));

    ARBORX_TEST_MINIMUM_SPANNING_TREE(
        exec_space, non_equidistant_points, 5,
        (std::vector<ArborX::Details::WeightedEdge>{
            {0, 1, 6}, {1, 2, 5}, {2, 3, 4}, {1, 4, 5}, {1, 5, 9}}));

    ARBORX_TEST_MINIMUM_SPANNING_TREE(
        exec_space, non_equidistant_points, 6,
        (std::vector<ArborX::Details::WeightedEdge>{
            {0, 1, 10}, {1, 2, 9}, {2, 3, 8}, {3, 4, 7}, {0, 5, 10}}));
  }
}
