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
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE KokkosExtViewHelpers

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE_TEMPLATE(last_element, DeviceType, ARBORX_DEVICE_TYPES)
{
  using KokkosExt::lastElement;
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace execution_space;
  Kokkos::View<int *, DeviceType> v("v", 2);
  auto v_host = Kokkos::create_mirror_view(v);
  v_host(0) = 33;
  v_host(1) = 24;
  Kokkos::deep_copy(v, v_host);
  BOOST_TEST(lastElement(execution_space, v) == 24);
  Kokkos::View<int *, DeviceType> w("w", 0);
  BOOST_CHECK_THROW(lastElement(execution_space, w), ArborX::SearchException);
  Kokkos::View<double[1], DeviceType> u("u");
  Kokkos::deep_copy(u, 3.14);
  BOOST_TEST(lastElement(execution_space, u) == 3.14);
}
