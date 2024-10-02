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
#include <ArborX_Exception.hpp>
#include <ArborX_KokkosExtStdAlgorithms.hpp>

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
