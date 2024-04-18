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
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_DetailsKokkosExtStdAlgorithms.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

#define BOOST_TEST_MODULE KokkosExtStdAlgorithms

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE_TEMPLATE(iota, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  namespace KokkosExt = ArborX::Details::KokkosExt;

  ExecutionSpace space{};
  int const n = 10;
  double const val = 3.;

  Kokkos::View<double *, DeviceType> v("v", n);
  KokkosExt::iota(space, v, val);
  std::vector<double> v_ref(n);
  std::iota(v_ref.begin(), v_ref.end(), val);
  auto v_host = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(v_host, v);
  BOOST_TEST(v_ref == v_host, tt::per_element());

  Kokkos::View<int[3], DeviceType> w("w");
  KokkosExt::iota(space, w);
  std::vector<int> w_ref = {0, 1, 2};
  auto w_host = Kokkos::create_mirror_view(w);
  Kokkos::deep_copy(w_host, w);
  BOOST_TEST(w_ref == w_host, tt::per_element());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(exclusive_scan, DeviceType, ARBORX_DEVICE_TYPES)
{
  namespace KokkosExt = ArborX::Details::KokkosExt;
  using ExecutionSpace = typename DeviceType::execution_space;

  ExecutionSpace space{};

  int const n = 10;
  Kokkos::View<int *, DeviceType> x("x", n);
  std::vector<int> x_ref(n, 1);
  x_ref.back() = 0;
  auto x_host = Kokkos::create_mirror_view(x);
  for (int i = 0; i < n; ++i)
    x_host(i) = x_ref[i];
  Kokkos::deep_copy(x, x_host);

  Kokkos::View<int *, DeviceType> y("y", n);
  KokkosExt::exclusive_scan(space, x, y, 0);

  std::vector<int> y_ref(n);
  std::iota(y_ref.begin(), y_ref.end(), 0);
  auto y_host = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(y_host, y);
  Kokkos::deep_copy(x_host, x);
  BOOST_TEST(y_host == y_ref, tt::per_element());
  BOOST_TEST(x_host == x_ref, tt::per_element());
  // in-place
  KokkosExt::exclusive_scan(space, x, x, 0);
  Kokkos::deep_copy(x_host, x);
  BOOST_TEST(x_host == y_ref, tt::per_element());
  int const m = 11;
  BOOST_TEST(n != m);
  Kokkos::View<int *, DeviceType> z("z", m);
  BOOST_CHECK_THROW(KokkosExt::exclusive_scan(space, x, z, 0),
                    ArborX::SearchException);
  Kokkos::View<double[3], DeviceType> v("v");
  auto v_host = Kokkos::create_mirror_view(v);
  v_host(0) = 1.;
  v_host(1) = 1.;
  v_host(2) = 0.;
  Kokkos::deep_copy(v, v_host);
  // in-place with init value
  KokkosExt::exclusive_scan(space, v, v, 5.);
  Kokkos::deep_copy(v_host, v);
  std::vector<double> v_ref = {5., 6., 7.};
  BOOST_TEST(v_host == v_ref, tt::per_element());
  Kokkos::View<double *, DeviceType> w("w", 4);
  BOOST_CHECK_THROW(KokkosExt::exclusive_scan(space, v, w, 0),
                    ArborX::SearchException);
  v_host(0) = 1.;
  v_host(1) = 0.;
  v_host(2) = 0.;
  Kokkos::deep_copy(v, v_host);
  Kokkos::resize(w, 3);
  KokkosExt::exclusive_scan(space, v, w, 0);
  auto w_host = Kokkos::create_mirror_view(w);
  Kokkos::deep_copy(w_host, w);
  std::vector<double> w_ref = {0., 1., 1.};
  BOOST_TEST(w_host == w_ref, tt::per_element());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(reduce, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace space{};

  namespace KokkosExt = ArborX::Details::KokkosExt;

  Kokkos::View<int[6], DeviceType> v("v");
  Kokkos::deep_copy(v, 5);
  BOOST_TEST(KokkosExt::reduce(space, v, 3) == 33);

  Kokkos::View<int *, DeviceType> w("w", 5);
  KokkosExt::iota(space, w, 2);
  BOOST_TEST(KokkosExt::reduce(space, w, 4) == 24);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(adjacent_difference, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace space{};

  using ArborX::Details::KokkosExt::adjacent_difference;

  Kokkos::View<int[5], DeviceType> v("v");
  auto v_host = Kokkos::create_mirror_view(v);
  v_host(0) = 2;
  v_host(1) = 4;
  v_host(2) = 6;
  v_host(3) = 8;
  v_host(4) = 10;
  Kokkos::deep_copy(v, v_host);
  // In-place operation is not allowed
  BOOST_CHECK_THROW(adjacent_difference(space, v, v), ArborX::SearchException);
  auto w = Kokkos::create_mirror(DeviceType(), v);
  BOOST_CHECK_NO_THROW(adjacent_difference(space, v, w));
  auto w_host = Kokkos::create_mirror_view(w);
  Kokkos::deep_copy(w_host, w);
  std::vector<int> w_ref(5, 2);
  BOOST_TEST(w_host == w_ref, tt::per_element());

  Kokkos::View<float *, DeviceType> x("x", 10);
  Kokkos::deep_copy(x, 3.14);
  BOOST_CHECK_THROW(adjacent_difference(space, x, x), ArborX::SearchException);
  Kokkos::View<float[10], DeviceType> y("y");
  BOOST_CHECK_NO_THROW(adjacent_difference(space, x, y));
  std::vector<float> y_ref(10);
  y_ref[0] = 3.14;
  auto y_host = Kokkos::create_mirror_view(y);
  Kokkos::deep_copy(y_host, y);
  BOOST_TEST(y_host == y_ref, tt::per_element());

  Kokkos::resize(x, 5);
  BOOST_CHECK_THROW(adjacent_difference(space, y, x), ArborX::SearchException);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(nth_element, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace space;

  using ArborX::Details::KokkosExt::nth_element;

  for (auto v_ref : {std::vector<float>{}, std::vector<float>{0.5f},
                     std::vector<float>{0.1f, 0.1f, 0.1f},
                     std::vector<float>{0.1f, 0.2f, 0.3f},
                     std::vector<float>{0.1f, 0.3f, -0.5f, 1.0f, -0.9f, -1.2f}})
  {
    int const n = v_ref.size();

    Kokkos::View<float *, DeviceType> v("v", n);
    Kokkos::deep_copy(
        space, v,
        Kokkos::View<float *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(v_ref.data(), n));

    Kokkos::View<float *, DeviceType> nth("nth", n);
    for (int i = 0; i < n; ++i)
    {
      auto v_copy = ArborX::Details::KokkosExt::clone(space, v);
      Kokkos::parallel_for(
          Kokkos::RangePolicy<ExecutionSpace>(space, 0, 1), KOKKOS_LAMBDA(int) {
            nth_element(v_copy.data(), v_copy.data() + i, v_copy.data() + n);
            nth(i) = v_copy(i);
          });
    }
    space.fence();

    auto nth_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, nth);
    std::sort(v_ref.begin(), v_ref.end());
    BOOST_TEST(nth_host == v_ref, tt::per_element());
  }
}
