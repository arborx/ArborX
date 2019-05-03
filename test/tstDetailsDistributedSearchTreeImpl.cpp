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
#include <ArborX_DetailsDistributedSearchTreeImpl.hpp>

#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE DetailsDistributedSearchTreeImpl

#include <algorithm> // fill
#include <set>
#include <vector>

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
  auto ids_host = Kokkos::create_mirror_view(ids);
  for (int i = 0; i < n; ++i)
    ids_host(i) = ids_[i];
  Kokkos::deep_copy(ids, ids_host);

  Kokkos::View<int *, DeviceType> results("results", n);
  auto results_host = Kokkos::create_mirror_view(results);
  for (int i = 0; i < n; ++i)
    results_host(i) = results_[i];
  Kokkos::deep_copy(results, results_host);

  Kokkos::View<int *, DeviceType> ranks("ranks", n);
  auto ranks_host = Kokkos::create_mirror_view(ranks);
  for (int i = 0; i < n; ++i)
    ranks_host(i) = ranks_[i];
  Kokkos::deep_copy(ranks, ranks_host);

  ArborX::Details::DistributedSearchTreeImpl<DeviceType>::sortResults(
      ids, results, ranks);

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
  BOOST_CHECK_THROW(
      ArborX::Details::DistributedSearchTreeImpl<DeviceType>::sortResults(
          ids, not_sized_properly),
      ArborX::SearchException);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(count_results, DeviceType, ARBORX_DEVICE_TYPES)
{
  std::vector<int> ids_ref = {4, 3, 2, 1, 4, 3, 2, 4, 3, 4};
  std::vector<int> offset_ref = {
      0, 0, 1, 3, 6, 10,
  };
  int const m = 5;
  int const nnz = 10;
  BOOST_TEST(ids_ref.size() == nnz);
  BOOST_TEST(offset_ref.size() == m + 1);

  Kokkos::View<int *, DeviceType> ids("query_ids", nnz);
  auto ids_host = Kokkos::create_mirror_view(ids);
  for (int i = 0; i < nnz; ++i)
    ids_host(i) = ids_ref[i];
  Kokkos::deep_copy(ids, ids_host);

  Kokkos::View<int *, DeviceType> offset("offset");

  ArborX::Details::DistributedSearchTreeImpl<DeviceType>::countResults(m, ids,
                                                                       offset);

  auto offset_host = Kokkos::create_mirror_view(offset);
  Kokkos::deep_copy(offset_host, offset);
  BOOST_TEST(offset_host == offset_ref, tt::per_element());
}

template <typename View1, typename View2>
inline void checkViewWasNotAllocated(View1 const &v1, View2 const &v2)
{
  // NOTE: cannot use operator== here because array layout may "change" for
  // rank-1 views
  BOOST_TEST(v1.data() == v2.data());
  BOOST_TEST(v1.span() == v2.span());

  BOOST_TEST((int)View1::rank == (int)View2::rank);
  BOOST_TEST((std::is_same<typename View1::const_value_type,
                           typename View2::const_value_type>::value));
  BOOST_TEST((std::is_same<typename View1::memory_space,
                           typename View2::memory_space>::value));

  BOOST_TEST(v1.extent(0) == v2.extent(0));
  BOOST_TEST(v1.extent(1) == v2.extent(1));
  BOOST_TEST(v1.extent(2) == v2.extent(2));
  BOOST_TEST(v1.extent(3) == v2.extent(3));
  BOOST_TEST(v1.extent(4) == v2.extent(4));
  BOOST_TEST(v1.extent(5) == v2.extent(5));
  BOOST_TEST(v1.extent(6) == v2.extent(6));
  BOOST_TEST(v1.extent(7) == v2.extent(7));
}

template <typename View1, typename View2>
inline void checkNewViewWasAllocated(View1 const &v1, View2 const &v2)
{
  BOOST_TEST(v1.data() != v2.data());

  BOOST_TEST((int)View1::rank == (int)View2::rank);
  BOOST_TEST((std::is_same<typename View1::const_value_type,
                           typename View2::const_value_type>::value));

  BOOST_TEST(v1.extent(0) == v2.extent(0));
  BOOST_TEST(v1.extent(1) == v2.extent(1));
  BOOST_TEST(v1.extent(2) == v2.extent(2));
  BOOST_TEST(v1.extent(3) == v2.extent(3));
  BOOST_TEST(v1.extent(4) == v2.extent(4));
  BOOST_TEST(v1.extent(5) == v2.extent(5));
  BOOST_TEST(v1.extent(6) == v2.extent(6));
  BOOST_TEST(v1.extent(7) == v2.extent(7));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(create_layout_right_mirror_view, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ArborX::Details::create_layout_right_mirror_view;
  using Kokkos::ALL;
  using Kokkos::LayoutLeft;
  using Kokkos::LayoutRight;
  using Kokkos::make_pair;
  using Kokkos::subview;
  using Kokkos::View;

  if (!Kokkos::Impl::SpaceAccessibility<
          Kokkos::HostSpace, typename DeviceType::memory_space>::accessible)
    return;

  // rank-1 and not strided -> do not allocate
  View<int *, LayoutLeft, DeviceType> u("u", 255);
  auto u_h = create_layout_right_mirror_view(u);
  checkViewWasNotAllocated(u, u_h);

  // right layout -> do not allocate
  View<int **, LayoutRight, DeviceType> v("v", 2, 3);
  auto v_h = create_layout_right_mirror_view(v);
  checkViewWasNotAllocated(v, v_h);

  // left layout and rank > 1 -> allocate
  View<int **, LayoutLeft, DeviceType> w("w", 4, 5);
  auto w_h = create_layout_right_mirror_view(w);
  checkNewViewWasAllocated(w, w_h);

  // strided layout -> allocate
  auto x = subview(v, ALL, 0);
  auto x_h = create_layout_right_mirror_view(x);
  checkNewViewWasAllocated(x, x_h);

  // subview is rank-1 and not strided -> do not allocate
  auto y = subview(u, make_pair(8, 16));
  auto y_h = create_layout_right_mirror_view(y);
  checkViewWasNotAllocated(y, y_h);
}

void checkBufferLayout(std::vector<int> const &ranks,
                       std::vector<int> const &permute_ref,
                       std::vector<int> const &unique_ref,
                       std::vector<int> const &counts_ref,
                       std::vector<int> const &offsets_ref)
{
  std::vector<int> permute(ranks.size());
  std::vector<int> unique;
  std::vector<int> counts;
  std::vector<int> offsets;
  ArborX::Details::sortAndDetermineBufferLayout(
      Kokkos::View<int const *, Kokkos::HostSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(ranks.data(),
                                                            ranks.size()),
      Kokkos::View<int *, Kokkos::HostSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(permute.data(),
                                                            permute.size()),
      unique, counts, offsets);
  BOOST_TEST(permute_ref == permute, tt::per_element());
  BOOST_TEST(unique_ref == unique, tt::per_element());
  BOOST_TEST(counts_ref == counts, tt::per_element());
  BOOST_TEST(offsets_ref == offsets, tt::per_element());
}

BOOST_AUTO_TEST_CASE(sort_and_determine_buffer_layout)
{
  checkBufferLayout({}, {}, {}, {}, {0});
  checkBufferLayout({2, 2}, {0, 1}, {2}, {2}, {0, 2});
  checkBufferLayout({3, 3, 2, 3, 2, 1}, {0, 1, 3, 2, 4, 5}, {3, 2, 1},
                    {3, 2, 1}, {0, 3, 5, 6});
  checkBufferLayout({1, 2, 3, 2, 3, 3}, {5, 3, 0, 4, 1, 2}, {3, 2, 1},
                    {3, 2, 1}, {0, 3, 5, 6});
  checkBufferLayout({0, 1, 2, 3}, {3, 2, 1, 0}, {3, 2, 1, 0}, {1, 1, 1, 1},
                    {0, 1, 2, 3, 4});
}
