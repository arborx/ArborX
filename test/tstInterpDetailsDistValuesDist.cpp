/****************************************************************************
 * Copyright (c) 2023 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "ArborX_EnableDeviceTypes.hpp"
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_InterpDetailsDistributedValuesDistributor.hpp>

#include <Kokkos_Core.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

#include <mpi.h>

BOOST_AUTO_TEST_CASE_TEMPLATE(distributed_tree_post_query_comms, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  ExecutionSpace space{};

  static constexpr int a = 100;
  static constexpr int b = 200;

  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  int mpi_rank;
  int mpi_size;
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_size);

  // 0: a   b       0: b+1   a+2
  // 1: a+1 b+1 --> 1: b+2   a+3
  // r: a+r b+r     r: b+r+1 a+r+2
  // s: a+s b+s     s: b     a+1
  Kokkos::View<ArborX::PairIndexRank *, MemorySpace> iar0("Testing::iar", 2);
  Kokkos::View<int *, MemorySpace> loc0("Testing::loc", 2);
  Kokkos::View<int *, MemorySpace> ref0("Testing::ref", 2);
  Kokkos::parallel_for(
      "for", Kokkos::RangePolicy<ExecutionSpace>(space, 0, 1),
      KOKKOS_LAMBDA(int const) {
        iar0(0) = {1, (mpi_rank + 1) % mpi_size};
        iar0(1) = {0, (mpi_rank + 2) % mpi_size};

        loc0(0) = a + mpi_rank;
        loc0(1) = b + mpi_rank;

        ref0(0) = b + (mpi_rank + 1) % mpi_size;
        ref0(1) = a + (mpi_rank + 2) % mpi_size;
      });
  ArborX::Interpolation::Details::DistributedValuesDistributor<MemorySpace>
      dtpqc0(mpi_comm, space, iar0);
  dtpqc0.distribute(space, loc0);
  ARBORX_MDVIEW_TEST(ref0, loc0);


  // 0:             0: b+1   a+1
  // 1: a+1 b+1 --> 1: b+2   a+2
  // r: a+r b+r     r: b+r+1 a+r+1
  // s: a+s b+s     s: b+1   a+1
  // Like the first example but "0" has no value / is not requested
  Kokkos::View<ArborX::PairIndexRank *, MemorySpace> iar1("Testing::iar", 2);
  Kokkos::View<int *, MemorySpace> loc1("Testing::loc",
                                        (mpi_rank == 0) ? 0 : 2);
  Kokkos::View<int *, MemorySpace> ref1("Testing::ref", 2);
  Kokkos::parallel_for(
      "for", Kokkos::RangePolicy<ExecutionSpace>(space, 0, 1),
      KOKKOS_LAMBDA(int const) {
        if (mpi_rank != mpi_size - 1)
        {
          iar1(0) = {1, (mpi_rank + 1) % mpi_size};
          iar1(1) = {0, (mpi_rank + 1) % mpi_size};
        }
        else
        {
          iar1(0) = {1, 1};
          iar1(1) = {0, 1};
        }

        if (mpi_rank != 0)
        {
          loc1(0) = a + mpi_rank;
          loc1(1) = b + mpi_rank;
        }

        if (mpi_rank != mpi_size - 1)
        {
          ref1(0) = b + (mpi_rank + 1) % mpi_size;
          ref1(1) = a + (mpi_rank + 1) % mpi_size;
        }
        else
        {
          ref1(0) = b + 1;
          ref1(1) = a + 1;
        }
      });
  ArborX::Interpolation::Details::DistributedValuesDistributor<MemorySpace>
      dtpqc1(mpi_comm, space, iar1);
  dtpqc1.distribute(space, loc1);
  ARBORX_MDVIEW_TEST(ref1, loc1);

  // 0: a   b       0:
  // 1: a+1 b+1 --> 1: b+2   a+3
  // r: a+r b+r     r: b+r+1 a+r+2
  // s: a+s b+s     s: b     a+1
  // Like the first example but "0" requests no value
  Kokkos::View<ArborX::PairIndexRank *, MemorySpace> iar2(
      "Testing::iar", (mpi_rank == 0) ? 0 : 2);
  Kokkos::View<int *, MemorySpace> loc2("Testing::loc", 2);
  Kokkos::View<int *, MemorySpace> ref2("Testing::ref",
                                        (mpi_rank == 0) ? 0 : 2);
  Kokkos::parallel_for(
      "for", Kokkos::RangePolicy<ExecutionSpace>(space, 0, 1),
      KOKKOS_LAMBDA(int const) {
        if (mpi_rank != 0)
        {
          iar2(0) = {1, (mpi_rank + 1) % mpi_size};
          iar2(1) = {0, (mpi_rank + 2) % mpi_size};
        }

        loc2(0) = a + mpi_rank;
        loc2(1) = b + mpi_rank;

        if (mpi_rank != 0)
        {
          ref2(0) = b + (mpi_rank + 1) % mpi_size;
          ref2(1) = a + (mpi_rank + 2) % mpi_size;
        }
      });
  ArborX::Interpolation::Details::DistributedValuesDistributor<MemorySpace>
      dtpqc2(mpi_comm, space, iar2);
  dtpqc2.distribute(space, loc2);
  ARBORX_MDVIEW_TEST(ref2, loc2);
}
