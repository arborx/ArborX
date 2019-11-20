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
#include <ArborX_DetailsSortUtils.hpp>

#include <boost/test/unit_test.hpp>

#include <set>

#define BOOST_TEST_MODULE DetailsSortUtils

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE_TEMPLATE(sort_results, DeviceType, ARBORX_DEVICE_TYPES)
{
  std::vector<int> ids_ = {4, 3, 2, 1, 4, 3, 2, 4, 3, 4};
  std::vector<int> sorted_ids = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4};
  std::vector<int> offset = {0, 1, 3, 6, 10};
  int const n = 10;
  int const m = 4;
  BOOST_TEST(ids_.size() == n);
  BOOST_TEST(sorted_ids.size() == n);
  BOOST_TEST(offset.size() == m + 1);
  std::vector<int> results_ = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<std::set<int>> sorted_results = {
      {3},
      {6, 2},
      {8, 5, 1},
      {9, 7, 4, 0},
  };
  std::vector<int> ranks_ = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
  std::vector<std::set<int>> sorted_ranks = {
      {13},
      {16, 12},
      {18, 15, 11},
      {19, 17, 14, 10},
  };
  BOOST_TEST(results_.size() == n);
  BOOST_TEST(ranks_.size() == n);

  Kokkos::View<int *, DeviceType> ids("query_ids", n);
  Kokkos::View<int *, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      ids_host(ids_.data(), ids_.size());
  Kokkos::deep_copy(ids, ids_host);

  Kokkos::View<int *, DeviceType> results("results", n);
  Kokkos::View<int *, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      results_host(results_.data(), results_.size());
  Kokkos::deep_copy(results, results_host);

  Kokkos::View<int *, DeviceType> ranks("ranks", n);
  Kokkos::View<int *, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      ranks_host(ranks_.data(), ranks_.size());
  Kokkos::deep_copy(ranks, ranks_host);

  ArborX::Details::sortMultipleViews(ids, results, ranks);

  // COMMENT: ids are untouched
  Kokkos::deep_copy(ids_host, ids);
  BOOST_TEST(ids_host == ids_, tt::per_element());

  Kokkos::deep_copy(results_host, results);
  Kokkos::deep_copy(ranks_host, ranks);
  for (int q = 0; q < m; ++q)
    for (int i = offset[q]; i < offset[q + 1]; ++i)
    {
      BOOST_TEST(sorted_results[q].count(results_host[i]) == 1);
      BOOST_TEST(sorted_ranks[q].count(ranks_host[i]) == 1);
    }

  Kokkos::View<int *, DeviceType> not_sized_properly("", m);
  BOOST_CHECK_THROW(ArborX::Details::sortMultipleViews(ids, not_sized_properly),
                    ArborX::SearchException);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sort_results_2d, DeviceType, ARBORX_DEVICE_TYPES)
{
  std::array<int, 5> ids_{4, 2, 1, 3, 5};
  Kokkos::View<int *, DeviceType> ids("ids", 5);
  Kokkos::View<int *, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      ids_host(ids_.data(), ids_.size());
  Kokkos::deep_copy(ids, ids_host);

  int results_2d_[5][2] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}};
  int sorted_results_2d[5][2] = {{5, 6}, {3, 4}, {7, 8}, {1, 2}, {9, 10}};
  Kokkos::View<int **, DeviceType> results_2d("results", 5, 2);
  auto results_2d_host = Kokkos::create_mirror_view(results_2d);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 2; ++j)
      results_2d_host(i, j) = results_2d_[i][j];
  Kokkos::deep_copy(results_2d, results_2d_host);

  ArborX::Details::sortMultipleViews(ids, results_2d);

  Kokkos::deep_copy(results_2d_host, results_2d);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 2; ++j)
      BOOST_TEST(results_2d_host(i, j) == sorted_results_2d[i][j]);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sort_results_3d, DeviceType, ARBORX_DEVICE_TYPES)
{
  std::array<int, 5> ids_{4, 2, 1, 3, 5};
  Kokkos::View<int *, DeviceType> ids("ids", 5);
  Kokkos::View<int *, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      ids_host(ids_.data(), ids_.size());
  Kokkos::deep_copy(ids, ids_host);

  int results_3d_[5][2][2] = {{{1, 2}, {3, 4}},
                              {{5, 6}, {7, 8}},
                              {{9, 10}, {11, 12}},
                              {{13, 14}, {15, 16}},
                              {{17, 18}, {19, 20}}};
  int sorted_results_3d[5][2][2] = {{{9, 10}, {11, 12}},
                                    {{5, 6}, {7, 8}},
                                    {{13, 14}, {15, 16}},
                                    {{1, 2}, {3, 4}},
                                    {{17, 18}, {19, 20}}};
  Kokkos::View<int ***, DeviceType> results_3d("results", 5, 2, 2);
  auto results_3d_host = Kokkos::create_mirror_view(results_3d);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 2; ++k)
        results_3d_host(i, j, k) = results_3d_[i][j][k];
  Kokkos::deep_copy(results_3d, results_3d_host);

  ArborX::Details::sortMultipleViews(ids, results_3d);

  Kokkos::deep_copy(results_3d_host, results_3d);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 2; ++k)
        BOOST_TEST(results_3d_host(i, j, k) == sorted_results_3d[i][j][k]);
}
