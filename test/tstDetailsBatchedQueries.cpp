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
#include <ArborX_DetailsBatchedQueries.hpp>

#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE DetailsBatchedQueries

namespace tt = boost::test_tools;

template <typename DeviceType, typename ValueType>
Kokkos::View<ValueType *, DeviceType> toView(std::vector<ValueType> const &v)
{
  Kokkos::View<ValueType *, DeviceType> w("whocares", v.size());
  auto w_host = Kokkos::create_mirror_view(w);
  for (int i = 0; i < w.extent_int(0); ++i)
    w_host(i) = v[i];
  Kokkos::deep_copy(w, w_host);
  return w;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(permute_offset_and_indices, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  Kokkos::View<int *, DeviceType> offset("offset", 0);
  Kokkos::View<int *, DeviceType> indices("indices", 0);

  Kokkos::View<size_t *, DeviceType> permute("permute", 0);

  BOOST_CHECK_THROW(
      ArborX::Details::BatchedQueries<DeviceType>::reversePermutation(
          permute, offset, indices),
      ArborX::SearchException);

  Kokkos::resize(offset, 1);
  BOOST_CHECK_NO_THROW(
      ArborX::Details::BatchedQueries<DeviceType>::reversePermutation(
          permute, offset, indices));

  std::vector<int> offset_ = {0, 0, 1, 3, 6, 10};
  std::vector<int> indices_ = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4};
  std::vector<size_t> permute_ = {4, 3, 2, 1, 0};
  std::vector<int> offset_ref = {0, 4, 7, 9, 10, 10};
  std::vector<int> indices_ref = {4, 4, 4, 4, 3, 3, 3, 2, 2, 1};

  std::tie(offset, indices) =
      ArborX::Details::BatchedQueries<DeviceType>::reversePermutation(
          toView<DeviceType>(permute_), toView<DeviceType>(offset_),
          toView<DeviceType>(indices_));
  auto offset_host = Kokkos::create_mirror_view(offset);
  Kokkos::deep_copy(offset_host, offset);
  auto indices_host = Kokkos::create_mirror_view(indices);
  Kokkos::deep_copy(indices_host, indices);
  BOOST_TEST(offset_host == offset_ref, tt::per_element());
  BOOST_TEST(indices_host == indices_ref, tt::per_element());
}
