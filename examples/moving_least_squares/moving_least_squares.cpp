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

using Point = ArborX::ExperimentalHyperGeometry::Point<2, double>;
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = typename ExecutionSpace::memory_space;

KOKKOS_FUNCTION double functionToApproximate(Point const &p)
{
  return Kokkos::cos(p[0] + p[1] / 4);
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  ExecutionSpace space{};

  // Source space is a 3x3 grid
  // Target space is a single point
  //
  //  ^
  //  |
  //  S     S     S
  //  |
  //  |   T
  //  S     S     S
  //  |
  //  |
  // -S-----S-----S->
  //  |

  // Set up points
  Kokkos::View<Point *, MemorySpace> src_points("Example::src_points", 9);
  Kokkos::View<Point *, MemorySpace> tgt_points("Example::tgt_points", 1);
  Kokkos::parallel_for(
      "Example::make_points", Kokkos::RangePolicy<ExecutionSpace>(space, 0, 1),
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
      });

  // Set up values
  Kokkos::View<double *, MemorySpace> src_values("Example::src_values", 9);
  Kokkos::View<double *, MemorySpace> app_values("Example::app_values", 1);
  Kokkos::parallel_for(
      "Example::make_values", Kokkos::RangePolicy<ExecutionSpace>(space, 0, 9),
      KOKKOS_LAMBDA(int const i) {
        src_values(i) = functionToApproximate(src_points(i));
      });

  // Build the moving least squares coefficients
  ArborX::Interpolation::MovingLeastSquares<MemorySpace> mls(space, src_points,
                                                             tgt_points);

  // Interpolate
  mls.interpolate(space, src_values, app_values);

  // Show results
  auto tgt_points_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, tgt_points);
  auto app_values_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, app_values);
  auto tgt_value = functionToApproximate(tgt_points_host(0));

  std::cout << "Approximated value: " << app_values_host(0) << '\n';
  std::cout << "Real value        : " << tgt_value << '\n';
  std::cout << "Difference        : "
            << Kokkos::abs(app_values_host(0) - tgt_value) << '\n';

  return 0;
}