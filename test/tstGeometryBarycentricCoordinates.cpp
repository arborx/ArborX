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

template <int N>
using Array = Kokkos::Array<float, N>;

BOOST_AUTO_TEST_CASE(barycentric_triangle)
{
  using ArborX::Point;
  using ArborX::Triangle;
  using ArborX::Experimental::barycentricCoordinates;

  Triangle<2> tri{{-1, -1}, {1, -1}, {-1, 1}};
  // clang-format off
  // vertices
  BOOST_TEST(barycentricCoordinates(tri, Point{-1.f, -1.f}) == (Array<3>{1, 0, 0}), tt::per_element());
  BOOST_TEST(barycentricCoordinates(tri, Point{1.f, -1.f}) == (Array<3>{0, 1, 0}), tt::per_element());
  BOOST_TEST(barycentricCoordinates(tri, Point{-1.f, 1.f}) == (Array<3>{0, 0, 1}), tt::per_element());
  // mid edges
  BOOST_TEST(barycentricCoordinates(tri, Point{0.f, -1.f}) == (Array<3>{0.5f, 0.5f, 0}), tt::per_element());
  BOOST_TEST(barycentricCoordinates(tri, Point{-1.f, 0.f}) == (Array<3>{0.5f, 0, 0.5f}), tt::per_element());
  BOOST_TEST(barycentricCoordinates(tri, Point{0.f, 0.f}) == (Array<3>{0, 0.5f, 0.5f}), tt::per_element());
  // center
  BOOST_TEST(barycentricCoordinates(tri, Point{-1.f/3, -1.f/3}) == (Array<3>{1.f/3, 1.f/3, 1.f/3}), tt::tolerance(1e-7f) << tt::per_element());
  // off-center
  BOOST_TEST(barycentricCoordinates(tri, Point{-0.4f, 0.2f}) == (Array<3>{0.1f, 0.3f, 0.6f}), tt::tolerance(1e-6f) << tt::per_element());
  // clang-format on
}

BOOST_AUTO_TEST_CASE(barycentric_tetrahedron)
{
  using ArborX::Point;
  using ArborX::Experimental::barycentricCoordinates;
  using ArborX::ExperimentalHyperGeometry::Tetrahedron;

  Tetrahedron<> tet{{0, 0, 0}, {0, 2, 0}, {2, 0, 0}, {0, 0, 2}};
  // clang-format off
  // vertices
  BOOST_TEST(barycentricCoordinates(tet, Point{0.f, 0.f, 0.f}) == (Array<4>{1, 0, 0, 0}), tt::per_element());
  BOOST_TEST(barycentricCoordinates(tet, Point{0.f, 2.f, 0.f}) == (Array<4>{0, 1, 0, 0}), tt::per_element());
  BOOST_TEST(barycentricCoordinates(tet, Point{2.f, 0.f, 0.f}) == (Array<4>{0, 0, 1, 0}), tt::per_element());
  BOOST_TEST(barycentricCoordinates(tet, Point{0.f, 0.f, 2.f}) == (Array<4>{0, 0, 0, 1}), tt::per_element());
  // (some) mid edges
  BOOST_TEST(barycentricCoordinates(tet, Point{0.f, 1.f, 0.f}) == (Array<4>{0.5f, 0.5f, 0, 0}), tt::per_element());
  BOOST_TEST(barycentricCoordinates(tet, Point{0.f, 1.f, 1.f}) == (Array<4>{0, 0.5f, 0, 0.5f}), tt::per_element());
  BOOST_TEST(barycentricCoordinates(tet, Point{1.f, 0.f, 1.f}) == (Array<4>{0, 0, 0.5f, 0.5f}), tt::per_element());
  // (some) mid faces
  BOOST_TEST(barycentricCoordinates(tet, Point{2.f/3, 2.f/3, 0.f}) == (Array<4>{1.f/3, 1.f/3, 1.f/3, 0}), tt::tolerance(1e-6f) << tt::per_element());
  BOOST_TEST(barycentricCoordinates(tet, Point{2.f/3, 2.f/3, 2.f/3}) == (Array<4>{0, 1.f/3, 1.f/3, 1.f/3}), tt::tolerance(1e-7f) << tt::per_element());
  // center
  BOOST_TEST(barycentricCoordinates(tet, Point{0.5f, 0.5f, 0.5f}) == (Array<4>{0.25f, 0.25f, 0.25f, 0.25f}), tt::tolerance(1e-7f) << tt::per_element());
  // clang-format on
}
