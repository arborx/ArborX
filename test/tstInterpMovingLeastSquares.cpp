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
#include <ArborX_InterpMovingLeastSquares.hpp>

#include <Kokkos_Core.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE_TEMPLATE(moving_least_squares, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  ExecutionSpace space{};

  // Case 1: f(x) = 3, 2 neighbors, linear
  //      -------0--------------->
  // SRC:        0   2   4   6
  // TGT:          1   3   5
  using point0 = ArborX::ExperimentalHyperGeometry::Point<1, double>;
  Kokkos::View<point0 *, MemorySpace> srcp0("srcp", 4);
  Kokkos::View<point0 *, MemorySpace> tgtp0("tgtp", 3);
  Kokkos::View<double *, MemorySpace> srcv0("srcv", 4);
  Kokkos::View<double *, MemorySpace> tgtv0("tgtv", 3);
  Kokkos::parallel_for(
      "for", Kokkos::RangePolicy<ExecutionSpace>(space, 0, 4),
      KOKKOS_LAMBDA(int const i) {
        auto f = [](const point0 &) { return 3.; };

        srcp0(i) = {{2. * i}};
        srcv0(i) = f(srcp0(i));

        if (i < 3)
        {
          tgtp0(i) = {{2. * i + 1}};
          tgtv0(i) = f(tgtp0(i));
        }
      });
  ArborX::Interpolation::MovingLeastSquares<MemorySpace, double> mls0(
      space, srcp0, tgtp0, 2, ArborX::Interpolation::CRBF::Wendland<0>{},
      ArborX::Interpolation::PolynomialDegree<1>{});
  auto eval0 = mls0.apply(space, srcv0);
  ARBORX_MDVIEW_TEST_TOL(eval0, tgtv0, Kokkos::Experimental::epsilon_v<float>);

  // Case 2: f(x, y) = xy + x, 8 neighbors, quad
  //        ^
  //        |
  //    S   S   S
  //      T | T
  // ---S---S---S--->
  //      T | T
  //    S   S   S
  //        |
  using point1 = ArborX::ExperimentalHyperGeometry::Point<2, double>;
  Kokkos::View<point1 *, MemorySpace> srcp1("srcp", 9);
  Kokkos::View<point1 *, MemorySpace> tgtp1("tgtp", 4);
  Kokkos::View<double *, MemorySpace> srcv1("srcv", 9);
  Kokkos::View<double *, MemorySpace> tgtv1("tgtv", 4);
  Kokkos::parallel_for(
      "for", Kokkos::RangePolicy<ExecutionSpace>(space, 0, 9),
      KOKKOS_LAMBDA(int const i) {
        int u = (i / 2) * 2 - 1;
        int v = (i % 2) * 2 - 1;
        int x = (i / 3) - 1;
        int y = (i % 3) - 1;
        auto f = [](const point1 &p) { return p[0] * p[1] + p[0]; };

        srcp1(i) = {{x * 2., y * 2.}};
        srcv1(i) = f(srcp1(i));
        if (i < 4)
        {
          tgtp1(i) = {{u * 1., v * 1.}};
          tgtv1(i) = f(tgtp1(i));
        }
      });
  ArborX::Interpolation::MovingLeastSquares<MemorySpace, double> mls1(
      space, srcp1, tgtp1, 8, ArborX::Interpolation::CRBF::Wendland<2>{},
      ArborX::Interpolation::PolynomialDegree<2>{});
  auto eval1 = mls1.apply(space, srcv1);
  ARBORX_MDVIEW_TEST_TOL(eval1, tgtv1, Kokkos::Experimental::epsilon_v<float>);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(moving_least_squares_edge_cases, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  ExecutionSpace space{};

  // Case 1: Same as previous case 1, but points are 2D and locked on y=0
  using point0 = ArborX::ExperimentalHyperGeometry::Point<2, double>;
  Kokkos::View<point0 *, MemorySpace> srcp0("srcp", 4);
  Kokkos::View<point0 *, MemorySpace> tgtp0("tgtp", 3);
  Kokkos::View<double *, MemorySpace> srcv0("srcv", 4);
  Kokkos::View<double *, MemorySpace> tgtv0("tgtv", 3);
  Kokkos::parallel_for(
      "for", Kokkos::RangePolicy<ExecutionSpace>(space, 0, 4),
      KOKKOS_LAMBDA(int const i) {
        auto f = [](const point0 &) { return 3.; };

        srcp0(i) = {{2. * i, 0.}};
        srcv0(i) = f(srcp0(i));

        if (i < 3)
        {
          tgtp0(i) = {{2. * i + 1, 0.}};
          tgtv0(i) = f(tgtp0(i));
        }
      });
  ArborX::Interpolation::MovingLeastSquares<MemorySpace, double> mls0(
      space, srcp0, tgtp0, 2, ArborX::Interpolation::CRBF::Wendland<0>{},
      ArborX::Interpolation::PolynomialDegree<1>{});
  auto eval0 = mls0.apply(space, srcv0);
  ARBORX_MDVIEW_TEST_TOL(eval0, tgtv0, Kokkos::Experimental::epsilon_v<float>);

  // Case 2: Same but corner source points are also targets
  using point1 = ArborX::ExperimentalHyperGeometry::Point<2, double>;
  Kokkos::View<point1 *, MemorySpace> srcp1("srcp", 9);
  Kokkos::View<point1 *, MemorySpace> tgtp1("tgtp", 4);
  Kokkos::View<double *, MemorySpace> srcv1("srcv", 9);
  Kokkos::View<double *, MemorySpace> tgtv1("tgtv", 4);
  Kokkos::parallel_for(
      "for", Kokkos::RangePolicy<ExecutionSpace>(space, 0, 9),
      KOKKOS_LAMBDA(int const i) {
        int u = (i / 2) * 2 - 1;
        int v = (i % 2) * 2 - 1;
        int x = (i / 3) - 1;
        int y = (i % 3) - 1;
        auto f = [](const point1 &p) { return p[0] * p[1] + p[0]; };

        srcp1(i) = {{x * 2., y * 2.}};
        srcv1(i) = f(srcp1(i));
        if (i < 4)
        {
          tgtp1(i) = {{u * 2., v * 2.}};
          tgtv1(i) = f(tgtp1(i));
        }
      });
  ArborX::Interpolation::MovingLeastSquares<MemorySpace, double> mls1(
      space, srcp1, tgtp1, 8, ArborX::Interpolation::CRBF::Wendland<2>{},
      ArborX::Interpolation::PolynomialDegree<2>{});
  auto eval1 = mls1.apply(space, srcv1);
  ARBORX_MDVIEW_TEST_TOL(eval1, tgtv1, Kokkos::Experimental::epsilon_v<float>);
}