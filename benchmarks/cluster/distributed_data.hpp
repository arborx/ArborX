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

#ifndef ARBORX_DISTRIBUTED_DATA_HPP
#define ARBORX_DISTRIBUTED_DATA_HPP

#include <ArborX_Point.hpp>
#include <detail/ArborX_DistributedUtils.hpp>
#include <misc/ArborX_Exception.hpp>

#include <array>

template <int DIM, typename Coordinate, typename MemorySpace>
Kokkos::View<ArborX::Point<DIM, Coordinate> *, MemorySpace>
generateDistributedData(MPI_Comm comm,
                        ArborXBenchmark::Parameters const &params)
{
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  auto n = params.n;

  auto factors = ArborX::Details::closestFactors<DIM>(comm_size);

  Kokkos::Array<int, DIM> Is;
  for (int d = 0, s = comm_rank; d < DIM; ++d)
  {
    Is[d] = s % factors[d];
    s /= factors[d];
  }

  int nx = std::floor(std::pow(n, 1. / DIM));
  n = std::pow(nx, DIM);

  if (comm_rank == 0)
    printf("n (global)        : %lld\n", ((long long)comm_size) * n);

  // x x   x x   x x   x x
  int num_seq = params.n_seq;
  int spacing = params.spacing;

  typename MemorySpace::execution_space space;
  Kokkos::View<ArborX::Point<DIM, Coordinate> *, MemorySpace> points(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "Benchmark::distributed_data"),
      n);
  if constexpr (DIM == 2)
  {
    Kokkos::parallel_for(
        "Benchmark::generateDistributedData",
        Kokkos::MDRangePolicy(space, {0, 0}, {nx, nx}),
        KOKKOS_LAMBDA(int i, int j) {
          auto pos = [num_seq, spacing](int p) {
            auto tile = num_seq + spacing - 1;
            return static_cast<Coordinate>((p / num_seq) * tile + p % num_seq);
          };
          points(j * nx + i) = {pos(Is[0] * nx + i), pos(Is[1] * nx + j)};
        });
  }
  else if constexpr (DIM == 3)
  {
    Kokkos::parallel_for(
        "Benchmark::generateDistributedData",
        Kokkos::MDRangePolicy(space, {0, 0, 0}, {nx, nx, nx}),
        KOKKOS_LAMBDA(int i, int j, int k) {
          auto pos = [num_seq, spacing](int p) {
            auto tile = num_seq + spacing - 1;
            return static_cast<Coordinate>((p / num_seq) * tile + p % num_seq);
          };
          points(k * nx * nx + j * nx + i) = {
              pos(Is[0] * nx + i), pos(Is[1] * nx + j), pos(Is[2] * nx + k)};
        });
  }
  else
  {
    ARBORX_ASSERT(false);
  }

  return points;
}

#endif
