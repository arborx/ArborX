/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

#define BOOST_TEST_MODULE StandardAlgorithms

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE_TEMPLATE(iota, DeviceType, ARBORX_DEVICE_TYPES)
{
  int const n = 10;
  double const val = 3.;
  Kokkos::View<double *, DeviceType> v("v", n);
  ArborX::iota(v, val);
  std::vector<double> v_ref(n);
  std::iota(v_ref.begin(), v_ref.end(), val);
  auto v_host = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(v_host, v);
  BOOST_TEST(v_ref == v_host, tt::per_element());

  Kokkos::View<int[3], DeviceType> w("w");
  ArborX::iota(w);
  std::vector<int> w_ref = {0, 1, 2};
  auto w_host = Kokkos::create_mirror_view(w);
  Kokkos::deep_copy(w_host, w);
  BOOST_TEST(w_ref == w_host, tt::per_element());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(prefix_sum, DeviceType, ARBORX_DEVICE_TYPES)
{
  int const n = 10;
  Kokkos::View<int *, DeviceType> x("x", n);
  std::vector<int> x_ref(n, 1);
  x_ref.back() = 0;
  auto x_host = Kokkos::create_mirror_view(x);
  for (int i = 0; i < n; ++i)
    x_host(i) = x_ref[i];
  Kokkos::deep_copy(x, x_host);
  Kokkos::View<int *, DeviceType> y("y", n);
  ArborX::exclusivePrefixSum(x, y);
  std::vector<int> y_ref(n);
  std::iota(y_ref.begin(), y_ref.end(), 0);
  auto y_host = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(y_host, y);
  Kokkos::deep_copy(x_host, x);
  BOOST_TEST(y_host == y_ref, tt::per_element());
  BOOST_TEST(x_host == x_ref, tt::per_element());
  // in-place
  ArborX::exclusivePrefixSum(x, x);
  Kokkos::deep_copy(x_host, x);
  BOOST_TEST(x_host == y_ref, tt::per_element());
  int const m = 11;
  BOOST_TEST(n != m);
  Kokkos::View<int *, DeviceType> z("z", m);
  BOOST_CHECK_THROW(ArborX::exclusivePrefixSum(x, z), ArborX::SearchException);
  Kokkos::View<double[3], DeviceType> v("v");
  auto v_host = Kokkos::create_mirror_view(v);
  v_host(0) = 1.;
  v_host(1) = 1.;
  v_host(2) = 0.;
  Kokkos::deep_copy(v, v_host);
  ArborX::exclusivePrefixSum(v);
  Kokkos::deep_copy(v_host, v);
  std::vector<double> v_ref = {0., 1., 2.};
  BOOST_TEST(v_host == v_ref, tt::per_element());
  Kokkos::View<double *, DeviceType> w("w", 4);
  BOOST_CHECK_THROW(ArborX::exclusivePrefixSum(v, w), ArborX::SearchException);
  v_host(0) = 1.;
  v_host(1) = 0.;
  v_host(2) = 0.;
  Kokkos::deep_copy(v, v_host);
  Kokkos::resize(w, 3);
  ArborX::exclusivePrefixSum(v, w);
  auto w_host = Kokkos::create_mirror_view(w);
  Kokkos::deep_copy(w_host, w);
  std::vector<double> w_ref = {0., 1., 1.};
  BOOST_TEST(w_host == w_ref, tt::per_element());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(last_element, DeviceType, ARBORX_DEVICE_TYPES)
{
  Kokkos::View<int *, DeviceType> v("v", 2);
  auto v_host = Kokkos::create_mirror_view(v);
  v_host(0) = 33;
  v_host(1) = 24;
  Kokkos::deep_copy(v, v_host);
  BOOST_TEST(ArborX::lastElement(v) == 24);
  Kokkos::View<int *, DeviceType> w("w", 0);
  BOOST_CHECK_THROW(ArborX::lastElement(w), ArborX::SearchException);
  Kokkos::View<double[1], DeviceType> u("u");
  Kokkos::deep_copy(u, 3.14);
  BOOST_TEST(ArborX::lastElement(u) == 3.14);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(minmax, DeviceType, ARBORX_DEVICE_TYPES)
{
  Kokkos::View<double[4], DeviceType> v("v");
  auto v_host = Kokkos::create_mirror_view(v);
  v_host(0) = 3.14;
  v_host(1) = 1.41;
  v_host(2) = 2.71;
  v_host(3) = 1.62;
  Kokkos::deep_copy(v, v_host);

  auto const result_float = ArborX::minMax(v);
  BOOST_TEST(std::get<0>(result_float) == 1.41);
  BOOST_TEST(std::get<1>(result_float) == 3.14);
  Kokkos::View<int *, DeviceType> w("w", 0);
  BOOST_CHECK_THROW(ArborX::minMax(w), ArborX::SearchException);
  Kokkos::resize(w, 1);
  Kokkos::deep_copy(w, 255);
  auto const result_int = ArborX::minMax(w);
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
#if 0
    // FIXME might be an issue with CUDA
    auto const minmax_x =
        ArborX::minMax( Kokkos::subview( u, Kokkos::ALL, 0 ) );
    BOOST_TEST( std::get<0>( minmax_x ) == 1 );
    BOOST_TEST( std::get<1>( minmax_x ) == 4 );
    auto const minmax_y =
        ArborX::minMax( Kokkos::subview( u, Kokkos::ALL, 1 ) );
    BOOST_TEST( std::get<0>( minmax_y ) == 2 );
    BOOST_TEST( std::get<1>( minmax_y ) == 5 );
#endif
}

BOOST_AUTO_TEST_CASE_TEMPLATE(accumulate, DeviceType, ARBORX_DEVICE_TYPES)
{
  Kokkos::View<int[6], DeviceType> v("v");
  Kokkos::deep_copy(v, 5);
  BOOST_TEST(ArborX::accumulate(v, 3) == 33);

  Kokkos::View<int *, DeviceType> w("w", 5);
  ArborX::iota(w, 2);
  BOOST_TEST(ArborX::accumulate(w, 4) == 24);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(adjacent_difference, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  Kokkos::View<int[5], DeviceType> v("v");
  auto v_host = Kokkos::create_mirror_view(v);
  v_host(0) = 2;
  v_host(1) = 4;
  v_host(2) = 6;
  v_host(3) = 8;
  v_host(4) = 10;
  Kokkos::deep_copy(v, v_host);
  // In-place operation is not allowed
  BOOST_CHECK_THROW(ArborX::adjacentDifference(v, v), ArborX::SearchException);
  auto w = Kokkos::create_mirror(DeviceType(), v);
  BOOST_CHECK_NO_THROW(ArborX::adjacentDifference(v, w));
  auto w_host = Kokkos::create_mirror_view(w);
  Kokkos::deep_copy(w_host, w);
  std::vector<int> w_ref(5, 2);
  BOOST_TEST(w_host == w_ref, tt::per_element());

  Kokkos::View<float *, DeviceType> x("x", 10);
  Kokkos::deep_copy(x, 3.14);
  BOOST_CHECK_THROW(ArborX::adjacentDifference(x, x), ArborX::SearchException);
  Kokkos::View<float[10], DeviceType> y("y");
  BOOST_CHECK_NO_THROW(ArborX::adjacentDifference(x, y));
  std::vector<float> y_ref(10);
  y_ref[0] = 3.14;
  auto y_host = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(y_host, y);
  BOOST_TEST(y_host == y_ref, tt::per_element());

  Kokkos::resize(x, 5);
  BOOST_CHECK_THROW(ArborX::adjacentDifference(y, x), ArborX::SearchException);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(min_and_max, DeviceType, ARBORX_DEVICE_TYPES)
{
  Kokkos::View<int[4], DeviceType> v("v");
  ArborX::iota(v);
  BOOST_TEST(ArborX::min(v) == 0);
  BOOST_TEST(ArborX::max(v) == 3);

  Kokkos::View<int *, DeviceType> w("w", 7);
  ArborX::iota(w, 2);
  BOOST_TEST(ArborX::min(w) == 2);
  BOOST_TEST(ArborX::max(w) == 8);
}
