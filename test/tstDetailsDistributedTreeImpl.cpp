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
#include <ArborX_DetailsDistributedTreeImpl.hpp>

#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE DetailsDistributedTreeImpl

#include <set>

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

  using ExecutionSpace = typename DeviceType::execution_space;
  ArborX::Details::DistributedTreeImpl<DeviceType>::sortResults(
      ExecutionSpace{}, ids, results, ranks);

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
      ArborX::Details::DistributedTreeImpl<DeviceType>::sortResults(
          ExecutionSpace{}, ids, not_sized_properly),
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

  using ExecutionSpace = typename DeviceType::execution_space;
  ArborX::Details::DistributedTreeImpl<DeviceType>::sortResults(
      ExecutionSpace{}, ids, results_2d);

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

  using ExecutionSpace = typename DeviceType::execution_space;
  ArborX::Details::DistributedTreeImpl<DeviceType>::sortResults(
      ExecutionSpace{}, ids, results_3d);

  Kokkos::deep_copy(results_3d_host, results_3d);
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 2; ++k)
        BOOST_TEST(results_3d_host(i, j, k) == sorted_results_3d[i][j][k]);
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

  Kokkos::View<int *, DeviceType> offset("offset", m);

  using ExecutionSpace = typename DeviceType::execution_space;
  ArborX::Details::DistributedTreeImpl<DeviceType>::countResults(
      ExecutionSpace{}, m, ids, offset);

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

BOOST_AUTO_TEST_CASE_TEMPLATE(create_layout_right_mirror_view_no_init,
                              DeviceType, ARBORX_DEVICE_TYPES)
{
  using ArborX::Details::create_layout_right_mirror_view_no_init;
  using Kokkos::ALL;
  using Kokkos::LayoutLeft;
  using Kokkos::LayoutRight;
  using Kokkos::make_pair;
  using Kokkos::subview;
  using Kokkos::View;

  if (!Kokkos::SpaceAccessibility<
          Kokkos::HostSpace, typename DeviceType::memory_space>::accessible)
    return;

  // rank-1 and not strided -> do not allocate
  View<int *, LayoutLeft, DeviceType> u("u", 255);
  auto u_h = create_layout_right_mirror_view_no_init(u);
  checkViewWasNotAllocated(u, u_h);

  // right layout -> do not allocate
  View<int **, LayoutRight, DeviceType> v("v", 2, 3);
  auto v_h = create_layout_right_mirror_view_no_init(v);
  checkViewWasNotAllocated(v, v_h);

  // the same with compile time size
  View<int[2][3], LayoutRight, DeviceType> v_c("v");
  auto v_c_h = create_layout_right_mirror_view_no_init(v_c);
  checkViewWasNotAllocated(v_c, v_c_h);

  // left layout and rank > 1 -> allocate
  View<int **, LayoutLeft, DeviceType> w("w", 4, 5);
  auto w_h = create_layout_right_mirror_view_no_init(w);
  checkNewViewWasAllocated(w, w_h);

  // the same with compile time size
  View<int *[5], LayoutLeft, DeviceType> w_c("v", 4);
  auto w_c_h = create_layout_right_mirror_view_no_init(w_c);
  checkNewViewWasAllocated(w_c, w_c_h);

  // strided layout -> allocate
  auto x = subview(v, ALL, 0);
  auto x_h = create_layout_right_mirror_view_no_init(x);
  checkNewViewWasAllocated(x, x_h);

  // subview is rank-1 and not strided -> do not allocate
  auto y = subview(u, make_pair(8, 16));
  auto y_h = create_layout_right_mirror_view_no_init(y);
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
  Kokkos::DefaultHostExecutionSpace space;
  ArborX::Details::sortAndDetermineBufferLayout(
      space,
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

BOOST_AUTO_TEST_CASE(pointer_depth)
{
  static_assert(ArborX::Details::internal::PointerDepth<double>::value == 0,
                "Failing for double");
  static_assert(ArborX::Details::internal::PointerDepth<double *>::value == 1,
                "Failing for double*");
  static_assert(ArborX::Details::internal::PointerDepth<double[2]>::value == 0,
                "Failing for double[2]");
  static_assert(ArborX::Details::internal::PointerDepth<double **>::value == 2,
                "Failing for double**");
  static_assert(ArborX::Details::internal::PointerDepth<double *[2]>::value ==
                    1,
                "Failing for double*[2]");
  static_assert(ArborX::Details::internal::PointerDepth<double[2][3]>::value ==
                    0,
                "Failing for double[2][3]");
  static_assert(ArborX::Details::internal::PointerDepth<double ***>::value == 3,
                "Failing for double***");
  static_assert(ArborX::Details::internal::PointerDepth<double **[2]>::value ==
                    2,
                "Failing for double[2]");
  static_assert(
      ArborX::Details::internal::PointerDepth<double *[2][3]>::value == 1,
      "Failing for double*[2][3]");
  static_assert(
      ArborX::Details::internal::PointerDepth<double[2][3][4]>::value == 0,
      "Failing for double[2][3][4]");
}

template <typename DeviceType>
struct Helper
{
  template <typename View1, typename View2, typename View3>
  static void checkSendAcrossNetwork(MPI_Comm comm, View1 const &ranks,
                                     View2 const &v_exp, View3 const &v_ref)
  {
    ArborX::Details::Distributor<DeviceType> distributor(comm);
    distributor.createFromSends(typename DeviceType::execution_space{}, ranks);

    // NOTE here we assume that the reference solution is sized properly
    auto v_imp = Kokkos::create_mirror(typename View3::memory_space(), v_ref);

    ArborX::Details::DistributedTreeImpl<DeviceType>::sendAcrossNetwork(
        typename DeviceType::execution_space{}, distributor, v_exp, v_imp);

    auto v_imp_host = Kokkos::create_mirror_view(v_imp);
    Kokkos::deep_copy(v_imp_host, v_imp);
    auto v_ref_host = Kokkos::create_mirror_view(v_ref);
    Kokkos::deep_copy(v_ref_host, v_ref);

    BOOST_TEST(v_imp.extent(0) == v_ref.extent(0));
    BOOST_TEST(v_imp.extent(1) == v_ref.extent(1));
    for (unsigned int i = 0; i < v_imp.extent(0); ++i)
    {
      for (unsigned int j = 0; j < v_imp.extent(1); ++j)
      {
        BOOST_TEST(v_imp_host(i, j) == v_ref_host(i, j));
      }
    }
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(send_across_network, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  int const DIM = 3;

  // send 1 packet to rank k
  // receive comm_size packets
  Kokkos::View<int **, DeviceType> u_exp("u_exp", comm_size, DIM);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, comm_size), KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < DIM; ++j)
          u_exp(i, j) = i + j * comm_rank;
      });

  Kokkos::View<int *, DeviceType> ranks_u("", comm_size);
  ArborX::iota(ExecutionSpace{}, ranks_u, 0);

  Kokkos::View<int **, DeviceType> u_ref("u_ref", comm_size, DIM);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, comm_size), KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < DIM; ++j)
          u_ref(i, j) = comm_rank + i * j;
      });

  Helper<DeviceType>::checkSendAcrossNetwork(comm, ranks_u, u_exp, u_ref);

  Kokkos::View<int **, DeviceType, Kokkos::MemoryUnmanaged> u_exp_unmanaged{
      u_exp};
  Helper<DeviceType>::checkSendAcrossNetwork(comm, ranks_u, u_exp_unmanaged,
                                             u_ref);
}
