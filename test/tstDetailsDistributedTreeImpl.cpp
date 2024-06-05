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
#include <ArborX_DetailsDistributedTreeUtils.hpp>
#include <ArborX_DetailsKokkosExtStdAlgorithms.hpp>

#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE DetailsDistributedTree

namespace tt = boost::test_tools;

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
  ArborX::Details::DistributedTree::countResults(ExecutionSpace{}, m, ids,
                                                 offset);

  auto offset_host = Kokkos::create_mirror_view(offset);
  Kokkos::deep_copy(offset_host, offset);
  BOOST_TEST(offset_host == offset_ref, tt::per_element());
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

    ArborX::Details::DistributedTree::sendAcrossNetwork(
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
  ArborX::Details::KokkosExt::iota(ExecutionSpace{}, ranks_u, 0);

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
