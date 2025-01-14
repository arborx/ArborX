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

#include "ArborXTest_StdVectorToKokkosView.hpp"
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include "ArborX_EnableViewComparison.hpp"
#include <kokkos_ext/ArborX_KokkosExtMinMaxReduce.hpp>
#include <kokkos_ext/ArborX_KokkosExtStdAlgorithms.hpp>

#include <Kokkos_Core.hpp>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

#define BOOST_TEST_MODULE MinMaxReduce

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE_TEMPLATE(minmax_reduce, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace space{};

  namespace KokkosExt = ArborX::Details::KokkosExt;

  Kokkos::View<double[4], DeviceType> v("v");
  auto v_host = Kokkos::create_mirror_view(v);
  v_host(0) = 3.14;
  v_host(1) = 1.41;
  v_host(2) = 2.71;
  v_host(3) = 1.62;
  Kokkos::deep_copy(v, v_host);

  auto const result_float = KokkosExt::minmax_reduce(space, v);
  BOOST_TEST(std::get<0>(result_float) == 1.41);
  BOOST_TEST(std::get<1>(result_float) == 3.14);
  Kokkos::View<int *, DeviceType> w("w", 0);
  BOOST_CHECK_THROW(KokkosExt::minmax_reduce(space, w),
                    ArborX::SearchException);
  Kokkos::resize(w, 1);
  Kokkos::deep_copy(w, 255);
  auto const result_int = KokkosExt::minmax_reduce(space, w);
  BOOST_TEST(std::get<0>(result_int) == 255);
  BOOST_TEST(std::get<1>(result_int) == 255);

  // testing use case in ORNL-CEES/DataTransferKit#336
  Kokkos::View<int[2][3], DeviceType> u("u");
  auto u_host = Kokkos::create_mirror_view(u);
  u_host(0, 0) = 1; // x
  u_host(0, 1) = 2; // y
  u_host(0, 2) = 3; // z
  u_host(1, 0) = 4; // x
  u_host(1, 1) = 5; // y
  u_host(1, 2) = 6; // Z
  Kokkos::deep_copy(u, u_host);
  auto const minmax_x =
      KokkosExt::minmax_reduce(space, Kokkos::subview(u, Kokkos::ALL, 0));
  BOOST_TEST(std::get<0>(minmax_x) == 1);
  BOOST_TEST(std::get<1>(minmax_x) == 4);
  auto const minmax_y =
      KokkosExt::minmax_reduce(space, Kokkos::subview(u, Kokkos::ALL, 1));
  BOOST_TEST(std::get<0>(minmax_y) == 2);
  BOOST_TEST(std::get<1>(minmax_y) == 5);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(min_and_max, DeviceType, ARBORX_DEVICE_TYPES)
{
  namespace KokkosExt = ArborX::Details::KokkosExt;

  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace space{};

  Kokkos::View<int[4], DeviceType> v("v");
  KokkosExt::iota(space, v);
  BOOST_TEST(KokkosExt::min_reduce(space, v) == 0);
  BOOST_TEST(KokkosExt::max_reduce(space, v) == 3);

  Kokkos::View<int *, DeviceType> w("w", 7);
  KokkosExt::iota(space, w, 2);
  BOOST_TEST(KokkosExt::min_reduce(space, w) == 2);
  BOOST_TEST(KokkosExt::max_reduce(space, w) == 8);

  Kokkos::View<float *, DeviceType> x("x", 3);
  Kokkos::deep_copy(x, 3.14f);
  BOOST_TEST(KokkosExt::min_reduce(space, x) == 3.14f);
  BOOST_TEST(KokkosExt::max_reduce(space, x) == 3.14f);
}
