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

#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include <ArborX_KDOP.hpp>
#include <ArborX_LinearBVH.hpp>

#include <Kokkos_CopyViews.hpp>

#include <boost/test/unit_test.hpp>

#include <vector>

#include "Search_UnitTestHelpers.hpp"

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_kdop, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  using Tree = ArborX::BVH<MemorySpace>;

  using ArborX::Point;

  std::vector<ArborX::Point> primitives = {
      {{0, 0, 0}}, // 0
      {{1, 1, 1}}, // 1
      {{2, 2, 2}}, // 2
      {{3, 3, 3}}, // 3
      {{1, 0, 0}}, // 4
      {{2, 0, 0}}, // 5
      {{3, 0, 0}}, // 6
      {{0, 1, 0}}, // 7
      {{0, 2, 0}}, // 8
      {{0, 3, 0}}, // 9
      {{0, 0, 1}}, // 10
      {{0, 0, 2}}, // 11
      {{0, 0, 3}}, // 12
  };
  Tree const tree(
      ExecutionSpace{},
      Kokkos::create_mirror_view_and_copy(
          MemorySpace{},
          Kokkos::View<Point *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
              primitives.data(), primitives.size())));

  // (0,0,0)->(1,2,3) box with (0,0,0)--(0,0,3) edge cut off
  ArborX::Experimental::KDOP<18> x;
  // bottom
  x += ArborX::Point{0.25, 0, 0};
  x += ArborX::Point{1, 0, 0};
  x += ArborX::Point{1, 2, 0};
  x += ArborX::Point{0, 2, 0};
  x += ArborX::Point{0, 0.25, 0};
  // top
  x += ArborX::Point{0.25, 0, 3};
  x += ArborX::Point{1, 0, 3};
  x += ArborX::Point{1, 2, 3};
  x += ArborX::Point{0, 2, 3};
  x += ArborX::Point{0, 0.25, 3};

  using IntersectsKDop = decltype(ArborX::intersects(x));
  std::vector<IntersectsKDop> predicates = {ArborX::intersects(x)};

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeIntersectsBoxQueries<DeviceType>({}),
                         make_reference_solution<int>({}, {0}));
  ARBORX_TEST_QUERY_TREE(
      ExecutionSpace{}, tree,
      Kokkos::create_mirror_view_and_copy(
          MemorySpace{}, Kokkos::View<IntersectsKDop *, Kokkos::HostSpace,
                                      Kokkos::MemoryUnmanaged>(
                             predicates.data(), predicates.size())),
      make_reference_solution<int>({1, 4, 7, 8}, {0, 4}));
}
