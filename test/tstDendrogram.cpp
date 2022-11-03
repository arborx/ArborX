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
#include <ArborX_Dendrogram.hpp>
#include <ArborX_DetailsWeightedEdge.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(Dendrogram)

using ArborX::Details::WeightedEdge;

namespace Test
{

template <class ExecutionSpace>
auto build_dendrogram(ExecutionSpace const &exec_space,
                      std::vector<WeightedEdge> const &edges_host)
{
  using ArborXTest::toView;
  auto edges = toView<ExecutionSpace>(edges_host, "Test::edges");

  using MemorySpace = typename ExecutionSpace::memory_space;
  ArborX::Experimental::Dendrogram<MemorySpace> dendrogram{exec_space, edges};

  auto parents_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          dendrogram._parents);
  return parents_host;
}

} // namespace Test

#define ARBORX_TEST_DENDROGRAM(exec_space, edges, ref)                         \
  BOOST_TEST(Test::build_dendrogram(exec_space, edges) == ref,                 \
             boost::test_tools::per_element());

BOOST_AUTO_TEST_CASE_TEMPLATE(dendrogram_union_find, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using ArborX::Details::WeightedEdge;

  ExecutionSpace space;

  // --0--
  // |   |
  // 0   1
  ARBORX_TEST_DENDROGRAM(space, (std::vector<WeightedEdge>{{0, 1, 3.f}}),
                         (std::vector<int>{-1, 0, 0}));

  //      ----0---
  //      |      |
  //   ---1---   |
  //   |     |   |
  // --2--   |   |
  // |   |   |   |
  // 0   1   2   3
  ARBORX_TEST_DENDROGRAM(
      space, (std::vector<WeightedEdge>{{0, 3, 7.f}, {1, 2, 3.f}, {0, 1, 2.f}}),
      (std::vector<int>{-1, 0, 1, 2, 2, 1, 0}));

  //   ----1----
  //   |       |
  // --2--   --0--
  // |   |   |   |
  // 0   1   2   3
  ARBORX_TEST_DENDROGRAM(
      space, (std::vector<WeightedEdge>{{2, 3, 2.f}, {2, 0, 9.f}, {0, 1, 3.f}}),
      (std::vector<int>{1, -1, 1, 2, 2, 0, 0}));
}

BOOST_AUTO_TEST_SUITE_END()
