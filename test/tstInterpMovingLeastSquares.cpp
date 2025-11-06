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

#include "ArborX_EnableDeviceTypes.hpp"
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_InterpMovingLeastSquares.hpp>

#include <Kokkos_Core.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE_TEMPLATE(moving_least_squares, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  // FIXME_HIP: the CI fails with:
  // fatal error: in "mls_coefficients_edge_cases<Kokkos__Device<Kokkos__HIP_
  // Kokkos__HIPSpace>>": std::runtime_error: Kokkos::Impl::ParallelFor/Reduce<
  // HIP > could not find a valid team size.
  // The error seems similar to https://github.com/kokkos/kokkos/issues/6743
#ifdef KOKKOS_ENABLE_HIP
  if (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>)
  {
    return;
  }
#endif

  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  ExecutionSpace space{};

  // Case 1: f(x) = 3, 2 neighbors, linear
  //      -------0--------------->
  // SRC:        0   2   4   6
  // TGT:          1   3   5
  using Point0 = ArborX::Point<1, double>;
  Kokkos::View<Point0 *, MemorySpace> srcp0("Testing::srcp0", 4);
  Kokkos::View<Point0 *, MemorySpace> tgtp0("Testing::tgtp0", 3);
  Kokkos::View<double *, MemorySpace> srcv0("Testing::srcv0", 4);
  Kokkos::View<double *, MemorySpace> tgtv0("Testing::tgtv0", 3);
  Kokkos::View<double *, MemorySpace> eval0("Testing::eval0", 3);
  Kokkos::parallel_for(
      "Testing::moving_least_squares::for0", Kokkos::RangePolicy(space, 0, 4),
      KOKKOS_LAMBDA(int const i) {
        auto f = [](Point0 const &) { return 3.; };

        srcp0(i) = {{2. * i}};
        srcv0(i) = f(srcp0(i));

        if (i < 3)
        {
          tgtp0(i) = {{2. * i + 1}};
          tgtv0(i) = f(tgtp0(i));
        }
      });
  ArborX::Interpolation::MovingLeastSquares<MemorySpace, double> mls0(
      space, srcp0, tgtp0, ArborX::Interpolation::CRBF::Wendland<0>{},
      ArborX::Interpolation::PolynomialDegree<1>{}, 2);
  mls0.interpolate(space, srcv0, eval0);
  ARBORX_MDVIEW_TEST_TOL(eval0, tgtv0, Kokkos::Experimental::epsilon_v<float>);

  // Case 2: f(x, y) = xy + 4x, 8 neighbors, quad
  //        ^
  //        |
  //    S   S   S
  //      T | T
  // ---S---S---S--->
  //      T | T
  //    S   S   S
  //        |
  using Point1 = ArborX::Point<2, double>;
  Kokkos::View<Point1 *, MemorySpace> srcp1("Testing::srcp1", 9);
  Kokkos::View<Point1 *, MemorySpace> tgtp1("Testing::tgtp1", 4);
  Kokkos::View<double *, MemorySpace> srcv1("Testing::srcv1", 9);
  Kokkos::View<double *, MemorySpace> tgtv1("Testing::tgtv1", 4);
  Kokkos::View<double *, MemorySpace> eval1("Testing::eval1", 4);
  Kokkos::parallel_for(
      "Testing::moving_least_squares::for1", Kokkos::RangePolicy(space, 0, 9),
      KOKKOS_LAMBDA(int const i) {
        int u = (i / 2) * 2 - 1;
        int v = (i % 2) * 2 - 1;
        int x = (i / 3) - 1;
        int y = (i % 3) - 1;
        auto f = [](Point1 const &p) { return p[0] * p[1] + 4 * p[0]; };

        srcp1(i) = {{x * 2., y * 2.}};
        srcv1(i) = f(srcp1(i));
        if (i < 4)
        {
          tgtp1(i) = {{u * 1., v * 1.}};
          tgtv1(i) = f(tgtp1(i));
        }
      });
  ArborX::Interpolation::MovingLeastSquares<MemorySpace, double> mls1(
      space, srcp1, tgtp1, ArborX::Interpolation::CRBF::Wendland<2>{},
      ArborX::Interpolation::PolynomialDegree<2>{}, 8);
  mls1.interpolate(space, srcv1, eval1);
  ARBORX_MDVIEW_TEST_TOL(eval1, tgtv1, Kokkos::Experimental::epsilon_v<float>);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(moving_least_squares_edge_cases, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  // FIXME_HIP: the CI fails with:
  // fatal error: in "mls_coefficients_edge_cases<Kokkos__Device<Kokkos__HIP_
  // Kokkos__HIPSpace>>": std::runtime_error: Kokkos::Impl::ParallelFor/Reduce<
  // HIP > could not find a valid team size.
  // The error seems similar to https://github.com/kokkos/kokkos/issues/6743
#ifdef KOKKOS_ENABLE_HIP
  if (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>)
  {
    return;
  }
#endif

  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  ExecutionSpace space{};

  // Case 1: Same as previous case 1, but points are 2D and locked on y=0
  using Point0 = ArborX::Point<2, double>;
  Kokkos::View<Point0 *, MemorySpace> srcp0("Testing::srcp0", 4);
  Kokkos::View<Point0 *, MemorySpace> tgtp0("Testing::tgtp0", 3);
  Kokkos::View<double *, MemorySpace> srcv0("Testing::srcv0", 4);
  Kokkos::View<double *, MemorySpace> tgtv0("Testing::tgtv0", 3);
  Kokkos::View<double *, MemorySpace> eval0("Testing::eval0", 3);
  Kokkos::parallel_for(
      "Testing::moving_least_squares_edge_cases::for0",
      Kokkos::RangePolicy(space, 0, 4), KOKKOS_LAMBDA(int const i) {
        auto f = [](Point0 const &) { return 3.; };

        srcp0(i) = {{2. * i, 0.}};
        srcv0(i) = f(srcp0(i));

        if (i < 3)
        {
          tgtp0(i) = {{2. * i + 1, 0.}};
          tgtv0(i) = f(tgtp0(i));
        }
      });
  ArborX::Interpolation::MovingLeastSquares<MemorySpace, double> mls0(
      space, srcp0, tgtp0, ArborX::Interpolation::CRBF::Wendland<0>{},
      ArborX::Interpolation::PolynomialDegree<1>{}, 2);
  mls0.interpolate(space, srcv0, eval0);
  ARBORX_MDVIEW_TEST_TOL(eval0, tgtv0, Kokkos::Experimental::epsilon_v<float>);

  // Case 2: Same but corner source points are also targets
  using Point1 = ArborX::Point<2, double>;
  Kokkos::View<Point1 *, MemorySpace> srcp1("Testing::srcp1", 9);
  Kokkos::View<Point1 *, MemorySpace> tgtp1("Testing::tgtp1", 4);
  Kokkos::View<double *, MemorySpace> srcv1("Testing::srcv1", 9);
  Kokkos::View<double *, MemorySpace> tgtv1("Testing::tgtv1", 4);
  Kokkos::View<double *, MemorySpace> eval1("Testing::eval1", 4);
  Kokkos::parallel_for(
      "Testing::moving_least_squares_edge_cases::for0",
      Kokkos::RangePolicy(space, 0, 9), KOKKOS_LAMBDA(int const i) {
        int u = (i / 2) * 2 - 1;
        int v = (i % 2) * 2 - 1;
        int x = (i / 3) - 1;
        int y = (i % 3) - 1;
        auto f = [](Point1 const &p) { return p[0] * p[1] + 4 * p[0]; };

        srcp1(i) = {{x * 2., y * 2.}};
        srcv1(i) = f(srcp1(i));
        if (i < 4)
        {
          tgtp1(i) = {{u * 2., v * 2.}};
          tgtv1(i) = f(tgtp1(i));
        }
      });
  ArborX::Interpolation::MovingLeastSquares<MemorySpace, double> mls1(
      space, srcp1, tgtp1, ArborX::Interpolation::CRBF::Wendland<2>{},
      ArborX::Interpolation::PolynomialDegree<2>{}, 8);
  mls1.interpolate(space, srcv1, eval1);
  ARBORX_MDVIEW_TEST_TOL(eval1, tgtv1, Kokkos::Experimental::epsilon_v<float>);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(moving_least_square_cartesian_convergence,
                              DeviceType, ARBORX_DEVICE_TYPES)
{
  // Test interpolation on a cartesian-type grid where using the minimal number
  // of neighbors (6 for quadratic polynomials in 2d) wasn't sufficient for an
  // exact interpolation for a function contained in the ansatz space.
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  ExecutionSpace space{};

  using Point = ArborX::Point<2, double>;

  auto f = [](double x, double y) {
    return Kokkos::sin(4 * x) + Kokkos::sin(2 * y);
  };

  constexpr int num_targets = 4;
  Kokkos::View<Point *, MemorySpace> target_coords("target_coords",
                                                   num_targets);
  Kokkos::View<double *, MemorySpace> target_values("target_values",
                                                    num_targets);
  Kokkos::parallel_for(
      Kokkos::RangePolicy(space, 0, 1), KOKKOS_LAMBDA(int) {
        target_coords(0) = {.788675, .788675};
        target_coords(1) = {.211325, .788675};
        target_coords(2) = {.788675, .211325};
        target_coords(3) = {.211325, .211325};
        for (int i = 0; i < num_targets; ++i)
          target_values(i) = f(target_coords(i)[0], target_coords(i)[1]);
      });

  constexpr int num_refinements = 9;
  constexpr int n = (1 << num_refinements) + 1;
  constexpr double h = 2. / (n - 1);
  constexpr int num_sources = n * n;

  std::cout << "h = " << h << std::endl;

  Kokkos::View<Point *, MemorySpace> source_coords("source_coords",
                                                   num_sources);
  Kokkos::View<double *, MemorySpace> source_values("source_values",
                                                    num_sources);
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy(space, {0, 0}, {n, n}),
      KOKKOS_LAMBDA(int const i, int const j) {
        int index = i * n + j;
        source_coords(index) = {-1 + i * h, -1 + j * h};
        source_values(index) =
            f(source_coords(index)[0], source_coords(index)[1]);
      });

  ArborX::Interpolation::MovingLeastSquares<MemorySpace, double> mls(
      space, source_coords, target_coords);
  Kokkos::View<double *, MemorySpace> interpolated_values("interpolated_values",
                                                          num_targets);
  mls.interpolate(space, source_values, interpolated_values);

  ARBORX_MDVIEW_TEST_TOL(interpolated_values, target_values, 2e-7);
}
