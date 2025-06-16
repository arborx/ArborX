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

#include "ArborX_EnableArrayComparison.hpp"
#include <ArborX_Box.hpp>
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_Point.hpp>
#include <ArborX_Tetrahedron.hpp>
#include <ArborX_Triangle.hpp>
#include <algorithms/ArborX_BarycentricCoordinates.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

using CoordinatesList = std::tuple<float, double>;

BOOST_AUTO_TEST_CASE_TEMPLATE(barycentric_triangle, Coordinate, CoordinatesList)
{
  using ArborX::Experimental::barycentricCoordinates;
  using Point = ArborX::Point<2, Coordinate>;
  using Triangle = ArborX::Triangle<2, Coordinate>;
  using Array = Kokkos::Array<Coordinate, 3>;

  Triangle tri{{-1, -1}, {1, -1}, {-1, 1}};
  // clang-format off
  // vertices
  BOOST_TEST(barycentricCoordinates(tri, Point{-1, -1}) == (Array{1, 0, 0}), tt::per_element());
  BOOST_TEST(barycentricCoordinates(tri, Point{1, -1}) == (Array{0, 1, 0}), tt::per_element());
  BOOST_TEST(barycentricCoordinates(tri, Point{-1, 1}) == (Array{0, 0, 1}), tt::per_element());
  // mid edges
  BOOST_TEST(barycentricCoordinates(tri, Point{0, -1}) == (Array{0.5, 0.5, 0}), tt::per_element());
  BOOST_TEST(barycentricCoordinates(tri, Point{-1, 0}) == (Array{0.5, 0, 0.5}), tt::per_element());
  BOOST_TEST(barycentricCoordinates(tri, Point{0, 0}) == (Array{0, 0.5, 0.5}), tt::per_element());
  // center
  BOOST_TEST(barycentricCoordinates(tri, Point{-1./3, -1./3}) == (Array{1./3, 1./3, 1./3}), tt::tolerance((Coordinate)1e-7) << tt::per_element());
  // off-center
  BOOST_TEST(barycentricCoordinates(tri, Point{-0.4, 0.2}) == (Array{0.1, 0.3, 0.6}), tt::tolerance((Coordinate)1e-6) << tt::per_element());
  // clang-format on
}

BOOST_AUTO_TEST_CASE_TEMPLATE(barycentric_tetrahedron, Coordinate,
                              CoordinatesList)
{
  using ArborX::Experimental::barycentricCoordinates;
  using Point = ArborX::Point<3, Coordinate>;
  using Tetrahedron =
      ArborX::ExperimentalHyperGeometry::Tetrahedron<Coordinate>;
  using Array = Kokkos::Array<Coordinate, 4>;

  Tetrahedron tet{{0, 0, 0}, {0, 2, 0}, {2, 0, 0}, {0, 0, 2}};
  // clang-format off
  // vertices
  BOOST_TEST(barycentricCoordinates(tet, Point{0, 0, 0}) == (Array{1, 0, 0, 0}), tt::per_element());
  BOOST_TEST(barycentricCoordinates(tet, Point{0, 2, 0}) == (Array{0, 1, 0, 0}), tt::per_element());
  BOOST_TEST(barycentricCoordinates(tet, Point{2, 0, 0}) == (Array{0, 0, 1, 0}), tt::per_element());
  BOOST_TEST(barycentricCoordinates(tet, Point{0, 0, 2}) == (Array{0, 0, 0, 1}), tt::per_element());
  // (some) mid edges
  BOOST_TEST(barycentricCoordinates(tet, Point{0, 1, 0}) == (Array{0.5, 0.5, 0, 0}), tt::per_element());
  BOOST_TEST(barycentricCoordinates(tet, Point{0, 1, 1}) == (Array{0, 0.5, 0, 0.5}), tt::per_element());
  BOOST_TEST(barycentricCoordinates(tet, Point{1, 0, 1}) == (Array{0, 0, 0.5, 0.5}), tt::per_element());
  // (some) mid faces
  BOOST_TEST(barycentricCoordinates(tet, Point{2./3, 2./3, 0}) == (Array{1./3, 1./3, 1./3, 0}), tt::tolerance((Coordinate)1e-6) << tt::per_element());
  BOOST_TEST(barycentricCoordinates(tet, Point{2./3, 2./3, 2./3}) == (Array{0, 1./3, 1./3, 1./3}), tt::tolerance((Coordinate)1e-7) << tt::per_element());
  // center
  BOOST_TEST(barycentricCoordinates(tet, Point{0.5, 0.5, 0.5}) == (Array{0.25, 0.25, 0.25, 0.25}), tt::tolerance((Coordinate)1e-7) << tt::per_element());
  // clang-format on
}
