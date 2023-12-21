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

#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_HyperPoint.hpp>
#include <ArborX_InterpDetailsPolynomialBasis.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(polynomial_basis_slice_lengths)
{
  using view = Kokkos::View<std::size_t **, Kokkos::HostSpace>;

  auto [arr0] =
      ArborX::Interpolation::Details::polynomialBasisSliceLengths<5, 3>();
  std::size_t ref0[3][5] = {
      {1, 1, 1, 1, 1}, {1, 2, 3, 4, 5}, {1, 3, 6, 10, 15}};
  view arr0_view(&arr0[0][0], 3, 5);
  view ref0_view(&ref0[0][0], 3, 5);
  ARBORX_MDVIEW_TEST(arr0_view, ref0_view);

  auto [arr1] =
      ArborX::Interpolation::Details::polynomialBasisSliceLengths<2, 3>();
  std::size_t ref1[3][2] = {{1, 1}, {1, 2}, {1, 3}};
  view arr1_view(&arr1[0][0], 3, 2);
  view ref1_view(&ref1[0][0], 3, 2);
  ARBORX_MDVIEW_TEST(arr1_view, ref1_view);
}

BOOST_AUTO_TEST_CASE(polynomial_basis_size)
{
  BOOST_TEST(
      (ArborX::Interpolation::Details::polynomialBasisSize<5, 3>() == 56));
  BOOST_TEST(
      (ArborX::Interpolation::Details::polynomialBasisSize<2, 3>() == 10));
  BOOST_TEST(
      (ArborX::Interpolation::Details::polynomialBasisSize<3, 0>() == 1));
}

BOOST_AUTO_TEST_CASE(polynomial_basis)
{
  using view = Kokkos::View<double *, Kokkos::HostSpace>;

  ArborX::ExperimentalHyperGeometry::Point<5, double> point0 = {1, 2, 3, 4, 5};
  auto arr0 =
      ArborX::Interpolation::Details::evaluatePolynomialBasis<3>(point0);
  double ref0[56] = {1,  1,  2,  3,  4,  5,  1,  2,  4,  3,  6,  9,  4,   8,
                     12, 16, 5,  10, 15, 20, 25, 1,  2,  4,  8,  3,  6,   12,
                     9,  18, 27, 4,  8,  16, 12, 24, 36, 16, 32, 48, 64,  5,
                     10, 20, 15, 30, 45, 20, 40, 60, 80, 25, 50, 75, 100, 125};
  view arr0_view(arr0.data(), 56);
  view ref0_view(&ref0[0], 56);
  ARBORX_MDVIEW_TEST(arr0_view, ref0_view);

  ArborX::ExperimentalHyperGeometry::Point<2, double> point1 = {-2, 0};
  auto arr1 =
      ArborX::Interpolation::Details::evaluatePolynomialBasis<3>(point1);
  double ref1[10] = {1, -2, 0, 4, 0, 0, -8, 0, 0, 0};
  view arr1_view(arr1.data(), 10);
  view ref1_view(&ref1[0], 10);
  ARBORX_MDVIEW_TEST(arr1_view, ref1_view);
}