/****************************************************************************
 * Copyright (c) 2023 by the ArborX authors                                 *
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

#include <boost/test/unit_test.hpp>

#include <numeric>
#include <vector>

BOOST_AUTO_TEST_SUITE(KerneliQueryCallbacks)

struct IntersectionCallback
{
  int query_index;
  bool &success;

  template <typename Query, typename Value>
  KOKKOS_FUNCTION void operator()(Query const &, Value const &value) const
  {
    ArborX::Box actual_box = value.bounding_volume;
    ArborX::Point min = actual_box.minCorner();
    if (query_index != min[0] || query_index != min[1] || query_index != min[2])
      success = false;
    else
      success = true;
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(callback_intersects, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using MemorySpace = typename DeviceType::memory_space;
  using ExecutionSpace = typename DeviceType::execution_space;
  using Tree = ArborX::BVH<MemorySpace>;

  int const n = 10;
  Kokkos::View<ArborX::Point *, DeviceType> points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "points"), n);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
        points(i) = {{(double)i, (double)i, (double)i}};
      });

  Tree const tree(ExecutionSpace{}, points);

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

BOOST_AUTO_TEST_SUITE_END()
