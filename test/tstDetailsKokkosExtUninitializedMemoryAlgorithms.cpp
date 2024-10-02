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

#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include <ArborX_Exception.hpp>
#include <kokkos_ext/ArborX_KokkosExtUninitializedMemoryAlgorithms.hpp>

#include <Kokkos_Core.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE KokkosExtUninitializedMemoryAlgorithms

namespace tt = boost::test_tools;

struct NoDefaultConstructor
{
  int value;
  KOKKOS_FUNCTION NoDefaultConstructor(int x)
      : value(x)
  {}
  KOKKOS_FUNCTION ~NoDefaultConstructor()
  {
    // Make sure compiler does not optimize out the write
    *((int volatile *)&value) = -1;
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(construct_destroy_at, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ArborX::Details::KokkosExt::construct_at;
  using ArborX::Details::KokkosExt::destroy_at;

  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace exec;

  int const n = 2;
  Kokkos::View<NoDefaultConstructor *, DeviceType> view(
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "Test::view"), n);
  Kokkos::parallel_for(
      "Test::construct", Kokkos::RangePolicy(exec, 0, n),
      KOKKOS_LAMBDA(int i) { construct_at(&view(i), i); });

  auto view_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  BOOST_TEST(view_host(0).value == 0);
  BOOST_TEST(view_host(1).value == 1);

  Kokkos::parallel_for(
      "Test::destroy", Kokkos::RangePolicy(exec, 0, n),
      KOKKOS_LAMBDA(int i) { destroy_at(&view(i)); });
  Kokkos::deep_copy(view_host, view);
  BOOST_TEST(view_host(0).value == -1);
  BOOST_TEST(view_host(1).value == -1);
}
