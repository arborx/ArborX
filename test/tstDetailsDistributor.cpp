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
#include <ArborX_KokkosExtStdAlgorithms.hpp>
#include <details/ArborX_DistributedTreeUtils.hpp>

#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE DetailsDistributor

namespace tt = boost::test_tools;

template <typename DeviceType>
struct Helper
{
  template <typename View1, typename View2, typename View3>
  static void checkDoPostsAndWaits(MPI_Comm comm, View1 const &ranks,
                                   View2 const &exports,
                                   View3 const &imports_ref)
  {
    ArborX::Details::Distributor<DeviceType> distributor(comm);
    distributor.createFromSends(typename DeviceType::execution_space{}, ranks);

    // NOTE here we assume that the reference solution is sized properly
    auto imports =
        Kokkos::create_mirror(typename View2::memory_space(), imports_ref);

    distributor.doPostsAndWaits(typename DeviceType::execution_space{}, exports,
                                imports);

    auto imports_host = Kokkos::create_mirror_view(imports);
    Kokkos::deep_copy(imports_host, imports);
    auto imports_ref_host = Kokkos::create_mirror_view(imports_ref);
    Kokkos::deep_copy(imports_ref_host, imports_ref);

    BOOST_TEST(imports_host == imports_ref_host, tt::per_element());
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(do_posts_and_waits, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  // send 1 packet to rank k
  // receive comm_size packets
  Kokkos::View<int *, DeviceType> exports("Testing::exports", comm_size);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, comm_size),
      KOKKOS_LAMBDA(int i) { exports(i) = (i < comm_rank ? 0 : comm_rank); });

  Kokkos::View<int *, DeviceType> ranks("Testing::ranks", comm_size);
  ArborX::Details::KokkosExt::iota(ExecutionSpace{}, ranks, 0);

  Kokkos::View<int *, DeviceType> imports_ref("Testing::v_ref", comm_size);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, comm_size), KOKKOS_LAMBDA(int i) {
        // Sum of all smaller ranks including this one
        imports_ref(i) = (i <= comm_rank ? i : 0);
      });

  Helper<DeviceType>::checkDoPostsAndWaits(comm, ranks, exports, imports_ref);

  Kokkos::View<int *, DeviceType, Kokkos::MemoryUnmanaged> exports_unmanaged{
      exports};
  Helper<DeviceType>::checkDoPostsAndWaits(comm, ranks, exports_unmanaged,
                                           imports_ref);
}
