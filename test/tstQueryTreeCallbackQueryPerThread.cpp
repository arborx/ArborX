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

#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include <ArborX.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

#include <numeric>
#include <vector>

BOOST_AUTO_TEST_SUITE(PerThread)

struct IntersectionCallback
{
  int query_index;
  bool &success;

  template <typename Query, typename Value>
  KOKKOS_FUNCTION void operator()(Query const &, Value const &value) const
  {
    success = (query_index == (int)value.index);
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(callback_intersects, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  using Point = ArborX::Point<3>;

  int const n = 10;
  Kokkos::View<Point *, DeviceType> points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "points"), n);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
        points(i) = {{(float)i, (float)i, (float)i}};
      });

  ArborX::BoundingVolumeHierarchy const tree(
      ExecutionSpace{}, ArborX::Experimental::attach_indices(points));

  bool success;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<ExecutionSpace>(0, n),
      KOKKOS_LAMBDA(int i, bool &update) {
        float center = i;
        ArborX::Box box{{center - .5f, center - .5f, center - .5f},
                        {center + .5f, center + .5f, center + .5f}};
        tree.query(ArborX::Experimental::PerThread{}, ArborX::intersects(box),
                   IntersectionCallback{i, update});
      },
      Kokkos::LAnd<bool, Kokkos::HostSpace>(success));

  BOOST_TEST(success);
}

struct OrderedIntersectionCallback
{
  int query_index;
  bool &success;

  template <typename Query, typename Value>
  KOKKOS_FUNCTION auto operator()(Query const &, Value const &value) const
  {
    success = (query_index == (int)value.index);
    return ArborX::CallbackTreeTraversalControl::early_exit;
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(callback_ordered_intersects, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  using Point = ArborX::Point<3>;

  int const n = 10;
  Kokkos::View<Point *, DeviceType> points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "points"), n);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
        points(i) = {{(float)i, (float)i, (float)i}};
      });

  ArborX::BoundingVolumeHierarchy const tree(
      ExecutionSpace{}, ArborX::Experimental::attach_indices(points));

  bool success;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<ExecutionSpace>(0, n),
      KOKKOS_LAMBDA(int i, bool &update) {
        float center = i;
        ArborX::Box box{{center - .5f, center - .5f, center - .5f},
                        {center + .5f, center + .5f, center + .5f}};
        tree.query(ArborX::Experimental::PerThread{},
                   ArborX::Experimental::ordered_intersects(box),
                   OrderedIntersectionCallback{i, update});
      },
      Kokkos::LAnd<bool, Kokkos::HostSpace>(success));

  BOOST_TEST(success);
}

BOOST_AUTO_TEST_SUITE_END()
