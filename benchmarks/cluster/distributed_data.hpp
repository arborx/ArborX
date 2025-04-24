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
#include <misc/ArborX_Exception.hpp>

#include <array>
#include <vector>

// Find closest DIM factors for a number. The factors are
// sorted in the descending order.
template <int DIM>
std::array<int, DIM> closestFactors(int n)
{
  static_assert(DIM > 0);

  std::array<int, DIM> result;
  if constexpr (DIM == 1)
  {
    result[0] = n;
    return result;
  }

  std::vector<int> factors;

  // Find all prime factors
  unsigned i = 2;
  while (n > 1)
  {
    if (n % i != 0)
    {
      ++i;
      continue;
    }

    factors.push_back(i);
    n /= i;
  }

  // Reduce the list of factors
  while (factors.size() > DIM)
  {
    // Combine two smallest factors
    factors[1] *= factors[0];
    factors.erase(factors.begin());

    // Re-sort the list
    std::sort(factors.begin(), factors.end());
  }

  int num_factors = factors.size();
  assert(num_factors <= DIM);

  result.fill(1); // for missing factors
  for (int d = 0; d < num_factors; ++d)
    result[d] = factors[num_factors - 1 - d];

  return result;
}

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

  auto factors = closestFactors<DIM>(comm_size);

  std::array<int, DIM> Is;
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
