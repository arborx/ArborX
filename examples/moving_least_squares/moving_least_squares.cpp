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

#include <ArborX.hpp>
#include <ArborX_Interp.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>

template <typename T>
KOKKOS_INLINE_FUNCTION double step(T const &p)
{
  return Kokkos::signbit(p[0]) ? 0 : 1;
}

using Point = ArborX::ExperimentalHyperGeometry::Point<1, double>;
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = typename ExecutionSpace::memory_space;

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);
  ExecutionSpace space{};

  static constexpr std::size_t num_points = 1000;

  Kokkos::View<Point *, MemorySpace> source_points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::source_points"),
      num_points);
  Kokkos::View<double *, MemorySpace> source_values(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::source_values"),
      num_points);
  Kokkos::View<Point *, MemorySpace> target_points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::target_points"),
      num_points);
  Kokkos::View<double *, MemorySpace> target_values(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::target_values"),
      num_points);
  Kokkos::parallel_for(
      "Example::fill_views",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_points),
      KOKKOS_LAMBDA(int const i) {
        double loc = i / (num_points - 1.);
        double off = .5 / (num_points - 1.);

        source_points(i)[0] = 2 * loc - 1;
        target_points(i)[0] = 2 * loc - 1 + off;

        source_values(i) = step(source_points(i));
        target_values(i) = step(target_points(i));
      });

  ArborX::Interpolation::MovingLeastSquares<MemorySpace, double> mls(
      space, source_points, target_points);

  auto approx_values = mls.apply(space, source_values);

  double max_error = 0.;
  Kokkos::parallel_reduce(
      "Example::reduce_error",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_points),
      KOKKOS_LAMBDA(int const i, double &loc_error) {
        loc_error = Kokkos::max(
            loc_error, Kokkos::abs(target_values(i) - approx_values(i)));
      },
      Kokkos::Max<double>(max_error));

  std::cout << "Error: " << max_error << '\n';

  return 0;
}