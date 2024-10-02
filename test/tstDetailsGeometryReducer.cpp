/****************************************************************************
 * Copyright (c) 2024 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include <ArborX_Box.hpp>
#include <details/ArborX_Algorithms.hpp>

#include <Kokkos_Core.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

#include <vector>

template <typename DeviceType, typename Geometry>
Geometry reduce(std::vector<Geometry> const &g)
{
  using MemorySpace = typename DeviceType::memory_space;
  using ExecutionSpace = typename DeviceType::execution_space;

  auto const n = g.size();
  Kokkos::View<Geometry *, MemorySpace> geometries("Testing::geometries", n);
  Kokkos::deep_copy(geometries,
                    Kokkos::View<Geometry const *, Kokkos::HostSpace,
                                 Kokkos::MemoryUnmanaged>(g.data(), n));

  Geometry result;
  Kokkos::parallel_reduce(
      "Testing::reduce_geometries",
      Kokkos::RangePolicy<ExecutionSpace>(ExecutionSpace{}, 0, n),
      KOKKOS_LAMBDA(int i, Geometry &update) {
        using ArborX::Details::expand;
        expand(update, geometries(i));
      },
      ArborX::Details::GeometryReducer<Geometry>(result));
  return result;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(reducer, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ArborX::Details::equals;

  using Box = ArborX::Box<2>;
  BOOST_TEST(equals(reduce<DeviceType, Box>({{{{0.f, 0.f}}, {{1.f, 1.f}}},
                                             {{{2.f, 2.f}}, {{3.f, 3.f}}}}),
                    Box{{{0.f, 0.f}}, {{3.f, 3.f}}}));
}
