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

#include <ArborX_InterpMovingLeastSquares.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>

#include <benchmark/benchmark.h>

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
  using MemorySpace = Kokkos::HostSpace;
  ExecutionSpace space{};

  // using Scalar = float;
  using Scalar = double;

  int n_target_1d = 2;
  Kokkos::View<ArborX::Point<2, Scalar> *, Kokkos::HostSpace>
      target_coordinates_host("target_coordinates", 4);
  target_coordinates_host(0) = {.788675, .788675};
  target_coordinates_host(1) = {.211325, .788675};
  target_coordinates_host(2) = {.788675, .211325};
  target_coordinates_host(3) = {.211325, .211325};

  auto target_coordinates = Kokkos::create_mirror_view_and_copy(
      MemorySpace{}, target_coordinates_host);
  Kokkos::View<Scalar *, MemorySpace> target_values(
      "target_values", (n_target_1d) * (n_target_1d));

  Kokkos::parallel_for(
      Kokkos::RangePolicy(space, 0, n_target_1d * n_target_1d),
      KOKKOS_LAMBDA(int const i) {
        target_values(i) = Kokkos::sin(4 * target_coordinates(i)[0]) +
                           Kokkos::sin(2 * target_coordinates(i)[1]);
      });

  for (int i = 1; i < 15; ++i)
  {
    int n_source_1d = (1 << i) + 1;
    Scalar h_source = 2. / (1 << i);

    Kokkos::View<ArborX::Point<2, Scalar> *, MemorySpace> source_coordinates(
        "source_coordinates", n_source_1d * n_source_1d);
    Kokkos::View<Scalar *, MemorySpace> source_values(
        "source_values", n_source_1d * n_source_1d);
    Kokkos::View<Scalar *, MemorySpace> interpolated_values(
        "interpolated_values", (n_target_1d) * (n_target_1d));

    Kokkos::parallel_for(
        Kokkos::MDRangePolicy(space, {0, 0}, {n_source_1d, n_source_1d}),
        KOKKOS_LAMBDA(int const i, int const j) {
          int source_index = i * n_source_1d + j;
          source_coordinates(source_index) = {-1 + i * h_source,
                                              -1 + j * h_source};
          source_values(source_index) = Kokkos::sin(4 * (-1 + i * h_source)) +
                                        Kokkos::sin(2 * (-1 + j * h_source));
        });
    ArborX::Interpolation::MovingLeastSquares<MemorySpace, Scalar> mls(
        space, source_coordinates, target_coordinates);
    mls.interpolate(space, source_values, interpolated_values);
    Scalar max_error;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy(space, 0, (n_target_1d) * (n_target_1d)),
        KOKKOS_LAMBDA(int const i, Scalar &tmp_error) {
          auto error = Kokkos::abs(target_values(i) - interpolated_values(i));
          tmp_error = Kokkos::max(tmp_error, error);
        },
        Kokkos::Max<Scalar, Kokkos::HostSpace>{max_error});
    std::cout << h_source << " max_error: " << max_error << std::endl;
  }
}
