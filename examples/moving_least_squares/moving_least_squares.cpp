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

#include <ArborX.hpp>
#include <ArborX_InterpMovingLeastSquares.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>

using Point = ArborX::Point<2, double>;

KOKKOS_FUNCTION double functionToApproximate(Point const &p)
{
  return Kokkos::cos(p[0] + p[1] / 4);
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;
  ExecutionSpace space{};

  // Source space is a 3x3 grid
  // Target space is 3 off-grid points
  //
  //  ^
  //  |
  //  S     S     S
  //  |
  //  |   T
  //  S     S  T  S
  //  |
  //  | T
  // -S-----S-----S->
  //  |

  int const num_sources = 9;
  int const num_targets = 3;

  // Set up points
  Kokkos::View<Point *, MemorySpace> src_points("Example::src_points",
                                                num_sources);
  Kokkos::View<Point *, MemorySpace> tgt_points("Example::tgt_points",
                                                num_targets);
  Kokkos::parallel_for(
      "Example::make_points", Kokkos::RangePolicy(space, 0, 1),
      KOKKOS_LAMBDA(int const) {
        src_points(0) = {0., 0.};
        src_points(1) = {1., 0.};
        src_points(2) = {2., 0.};
        src_points(3) = {0., 1.};
        src_points(4) = {1., 1.};
        src_points(5) = {2., 1.};
        src_points(6) = {0., 2.};
        src_points(7) = {1., 2.};
        src_points(8) = {2., 2.};
        tgt_points(0) = {4. / 6., 4. / 3.};
        tgt_points(1) = {9. / 6., 3. / 3.};
        tgt_points(2) = {2. / 6., 1. / 3.};
      });

  // Set up values
  Kokkos::View<double *, MemorySpace> src_values("Example::src_values",
                                                 num_sources);
  Kokkos::View<double *, MemorySpace> app_values("Example::app_values",
                                                 num_targets);
  Kokkos::View<double *, MemorySpace> ref_values("Example::ref_values",
                                                 num_targets);
  Kokkos::parallel_for(
      "Example::make_values", Kokkos::RangePolicy(space, 0, num_sources),
      KOKKOS_LAMBDA(int const i) {
        src_values(i) = functionToApproximate(src_points(i));
        if (i < num_targets)
          ref_values(i) = functionToApproximate(tgt_points(i));
      });

  // Build the moving least squares coefficients
  ArborX::Interpolation::MovingLeastSquares<MemorySpace> mls(space, src_points,
                                                             tgt_points);

  // Interpolate
  mls.interpolate(space, src_values, app_values);

  // Show results
  auto app_values_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, app_values);
  auto ref_values_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ref_values);
  auto diff = [=](int const i) {
    return Kokkos::abs(app_values_host(i) - ref_values_host(i));
  };

  std::cout << "Approximated values: " << app_values_host(0) << ' '
            << app_values_host(1) << ' ' << app_values_host(2) << '\n';
  std::cout << "Real values        : " << ref_values_host(0) << ' '
            << ref_values_host(1) << ' ' << ref_values_host(2) << '\n';
  std::cout << "Differences        : " << diff(0) << ' ' << diff(1) << ' '
            << diff(2) << '\n';

  return 0;
}
