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
#include "ArborX_EnableViewComparison.hpp"
#include <kokkos_ext/ArborX_KokkosExtKernelStdAlgorithms.hpp>
#include <kokkos_ext/ArborX_KokkosExtViewHelpers.hpp>
#include <misc/ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

#define BOOST_TEST_MODULE KokkosExtKernelStdAlgorithms

namespace tt = boost::test_tools;

template <typename T>
using UnmanagedHostView = Kokkos::View<T *, Kokkos::HostSpace,
                                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

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

    Kokkos::View<float *, DeviceType> v("Testing::v", 0);
    v = Kokkos::create_mirror_view_and_copy(
        space, UnmanagedHostView<float>(v_ref.data(), n));

    Kokkos::View<float *, DeviceType> nth("Testing::nth", n);
    for (int i = 0; i < n; ++i)
    {
      auto v_copy = ArborX::Details::KokkosExt::clone(space, v);
      Kokkos::parallel_for(
          "Testing::run_nth_element", Kokkos::RangePolicy(space, 0, 1),
          KOKKOS_LAMBDA(int) {
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

template <typename DeviceType>
int findUpperBound(std::vector<float> const &v_host, float x)
{
  auto const n = v_host.size();
  Kokkos::View<float *, DeviceType> v("Testing::v", n);
  Kokkos::deep_copy(v, UnmanagedHostView<float const>(v_host.data(), n));

  int pos;
  Kokkos::parallel_reduce(
      "Testing::run_upper_bound",
      Kokkos::RangePolicy<typename DeviceType::execution_space>(0, 1),
      KOKKOS_LAMBDA(int, int &update) {
        update =
            ArborX::Details::KokkosExt::upper_bound(v.data(), v.data() + n, x) -
            v.data();
      },
      pos);

  return pos;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(upper_bound, DeviceType, ARBORX_DEVICE_TYPES)
{
  BOOST_TEST(findUpperBound<DeviceType>({}, 0) == 0);
  BOOST_TEST(findUpperBound<DeviceType>({0}, -1) == 0);
  BOOST_TEST(findUpperBound<DeviceType>({0}, 0) == 1);
  BOOST_TEST(findUpperBound<DeviceType>({0}, 1) == 1);
  BOOST_TEST(findUpperBound<DeviceType>({1, 1}, 0) == 0);
  BOOST_TEST(findUpperBound<DeviceType>({1, 1}, 1) == 2);
  BOOST_TEST(findUpperBound<DeviceType>({1, 3, 5}, 1) == 1);
  BOOST_TEST(findUpperBound<DeviceType>({1, 3, 5}, 2) == 1);
  BOOST_TEST(findUpperBound<DeviceType>({1, 3, 5}, 3) == 2);
  BOOST_TEST(findUpperBound<DeviceType>({1, 3, 5}, 4) == 2);
  BOOST_TEST(findUpperBound<DeviceType>({1, 3, 5}, 5) == 3);
  BOOST_TEST(findUpperBound<DeviceType>({1, 3, 5, 7, 9, 11, 13}, 8) == 4);
}
