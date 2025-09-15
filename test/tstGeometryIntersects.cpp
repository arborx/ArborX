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

#include <ArborX_Box.hpp>
#include <ArborX_Ellipsoid.hpp>
#include <ArborX_KDOP.hpp>
#include <ArborX_Sphere.hpp>
#include <ArborX_Tetrahedron.hpp>
#include <ArborX_Triangle.hpp>
#include <algorithms/ArborX_Intersects.hpp>

#include <boost/test/unit_test.hpp>

using CoordinatesList = std::tuple<float, double>;

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_point_box, Coordinate, CoordinatesList)
{
  using ArborX::Details::intersects;
  using Point = ArborX::Point<3, Coordinate>;
  using Box = ArborX::Box<3, Coordinate>;

  constexpr Point point{{1.0, 1.0, 1.0}};

  // point is contained in a box
  static_assert(intersects(point, Box{{{0.0, 0.0, 0.0}}, {{2.0, 2.0, 2.0}}}));
  static_assert(
      !intersects(point, Box{{{-1.0, -1.0, -1.0}}, {{0.0, 0.0, 0.0}}}));
  // point is on a side of a box
  static_assert(intersects(point, Box{{{0.0, 0.0, 0.0}}, {{2.0, 2.0, 1.0}}}));
  // point is a corner of a box
  static_assert(intersects(point, Box{{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_box_box, Coordinate, CoordinatesList)
{
  using ArborX::Details::intersects;
  using Box = ArborX::Box<3, Coordinate>;

  // uninitialized box does not intersect with other boxes
  static_assert(!intersects(Box{}, Box{{{1.0, 2.0, 3.0}}, {{4.0, 5.0, 6.0}}}));
  // uninitialized box does not even intersect with itself
  static_assert(!intersects(Box{}, Box{}));
  // box with zero extent does
  static_assert(intersects(Box{{{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}},
                           Box{{{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}}));

  // unit cube
  constexpr Box box{{{0.0, 0.0, 0.0}}, {{1.0, 1.0, 1.0}}};
  static_assert(intersects(box, box));
  static_assert(!intersects(box, Box{}));
  // smaller box inside
  static_assert(
      intersects(box, Box{{{0.25, 0.25, 0.25}}, {{0.75, 0.75, 0.75}}}));
  // bigger box that contains it
  static_assert(intersects(box, Box{{{-1.0, -1.0, -1.0}}, {{2.0, 2.0, 2.0}}}));
  // couple boxes that do intersect
  static_assert(intersects(box, Box{{{0.5, 0.5, 0.5}}, {{1.5, 1.5, 1.5}}}));
  static_assert(intersects(box, Box{{{-0.5, -0.5, -0.5}}, {{0.5, 0.5, 0.5}}}));
  // couple boxes that do not
  static_assert(
      !intersects(box, Box{{{-2.0, -2.0, -2.0}}, {{-1.0, -1.0, -1.0}}}));
  static_assert(!intersects(box, Box{{{0.0, 0.0, 2.0}}, {{1.0, 1.0, 3.0}}}));
  // boxes intersect if faces touch
  static_assert(intersects(box, Box{{{1.0, 0.0, 0.0}}, {{2.0, 1.0, 1.0}}}));
  static_assert(intersects(box, Box{{{-0.5, -0.5, -0.5}}, {{0.5, 0.0, 0.5}}}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_sphere_point, Coordinate,
                              CoordinatesList)
{
  using ArborX::Details::intersects;
  using Point = ArborX::Point<3, Coordinate>;
  using Sphere = ArborX::Sphere<3, Coordinate>;

  // unit sphere
  constexpr Sphere sphere{{{0., 0., 0.}}, 1.};
  BOOST_TEST(intersects(sphere, Point{0., 0.5, 0.5}));
  BOOST_TEST(intersects(sphere, Point{0., 0., 1.0}));
  BOOST_TEST(intersects(Point{-1., 0., 0.}, sphere));
  BOOST_TEST(intersects(Point{-0.6, -0.8, 0.}, sphere));
  BOOST_TEST(!intersects(Point{-0.7, -0.8, 0.}, sphere));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_sphere_box, Coordinate,
                              CoordinatesList)
{
  using ArborX::Details::intersects;
  using Box = ArborX::Box<3, Coordinate>;
  using Sphere = ArborX::Sphere<3, Coordinate>;

  constexpr Sphere sphere{{{0., 0., 0.}}, 1.};
  BOOST_TEST(intersects(sphere, Box{{{0., 0., 0.}}, {{1., 1., 1.}}}));
  BOOST_TEST(!intersects(sphere, Box{{{1., 2., 3.}}, {{4., 5., 6.}}}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_point_triangle, Coordinate,
                              CoordinatesList)
{
  using ArborX::Details::intersects;
  using Point2 = ArborX::Point<2, Coordinate>;

  constexpr ArborX::Triangle<2, Coordinate> triangle{
      {{0, 0}}, {{1, 0}}, {{0, 2}}};

  BOOST_TEST(intersects(Point2{{0, 0}}, triangle));
  BOOST_TEST(intersects(Point2{{1, 0}}, triangle));
  BOOST_TEST(intersects(Point2{{0, 2}}, triangle));
  BOOST_TEST(intersects(Point2{{0.5, 0}}, triangle));
  BOOST_TEST(intersects(Point2{{0.5, 1}}, triangle));
  BOOST_TEST(intersects(Point2{{0, 1}}, triangle));
  BOOST_TEST(intersects(Point2{{0.25, 0.5}}, triangle));
  BOOST_TEST(!intersects(Point2{{1, 1}}, triangle));
  BOOST_TEST(!intersects(Point2{{0.5, 1.1}}, triangle));
  BOOST_TEST(!intersects(Point2{{1.1, 0}}, triangle));
  BOOST_TEST(!intersects(Point2{{-0.1, 0}}, triangle));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_triangle_box, Coordinate,
                              CoordinatesList)
{
  using ArborX::Details::intersects;
  using Box = ArborX::Box<3, Coordinate>;

  constexpr ArborX::Triangle<3, Coordinate> triangle3{
      {{1, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}}};

  BOOST_TEST(intersects(triangle3, Box{{{0., 0., 0.}}, {{1., 1., 1.}}}));
  BOOST_TEST(intersects(triangle3, Box{{{.2, .25, .25}}, {{.4, .3, .5}}}));
  BOOST_TEST(!intersects(triangle3, Box{{{.1, .2, .3}}, {{.2, .3, .4}}}));
  BOOST_TEST(intersects(triangle3, Box{{{0, 0, 0}}, {{.5, .25, .25}}}));

  constexpr Box unit_box{{{0, 0, 0}}, {{1, 1, 1}}};
  BOOST_TEST(intersects(
      ArborX::Triangle<3, Coordinate>{{{0, 0, 0}}, {{0, 1, 0}}, {{1, 0, 0}}},
      unit_box));
  BOOST_TEST(intersects(
      ArborX::Triangle<3, Coordinate>{{{0, 0, 0}}, {{0, 1, 0}}, {{-1, 0, 0}}},
      unit_box));
  BOOST_TEST(intersects(ArborX::Triangle<3, Coordinate>{{{.1, .1, .1}},
                                                        {{.1, .9, .1}},
                                                        {{.9, .1, .1}}},
                        unit_box));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_point_tetrahedron, Coordinate,
                              CoordinatesList)
{
  using ArborX::Details::intersects;
  using Point = ArborX::Point<3, Coordinate>;

  constexpr ArborX::ExperimentalHyperGeometry::Tetrahedron<Coordinate> tet{
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

  BOOST_TEST(intersects(Point{0, 0, 0}, tet));
  BOOST_TEST(intersects(Point{1, 0, 0}, tet));
  BOOST_TEST(intersects(Point{0, 1, 0}, tet));
  BOOST_TEST(intersects(Point{0, 0, 1}, tet));
  BOOST_TEST(intersects(Point{0.2, 0.2, 0.1}, tet));
  BOOST_TEST(!intersects(Point{-0.1, 0, 0}, tet));
  BOOST_TEST(!intersects(Point{0, -0.1, 0}, tet));
  BOOST_TEST(!intersects(Point{0, 0, 1.1}, tet));
  BOOST_TEST(!intersects(Point{0.5, 0.5, 0.5}, tet));
  BOOST_TEST(!intersects(Point{-0.5, 0.5, 0.5}, tet));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_segment_segment, Coordinate,
                              CoordinatesList)
{
  using ArborX::Details::intersects;
  using Segment2 = ArborX::Experimental::Segment<2, Coordinate>;

  constexpr Segment2 seg{{1, 1}, {2, 2}};
  BOOST_TEST(intersects(Segment2{{2, 2}, {3, 3}}, seg));
  BOOST_TEST(intersects(Segment2{{1.5, 1.5}, {1.7, 1.7}}, seg));
  BOOST_TEST(intersects(Segment2{{0, 0}, {1, 1}}, seg));
  BOOST_TEST(intersects(Segment2{{1, 2}, {2, 1}}, seg));
  BOOST_TEST(intersects(Segment2{{2, 0}, {0, 2}}, seg));
  BOOST_TEST(intersects(Segment2{{1, 3}, {3, 1}}, seg));
  BOOST_TEST(!intersects(Segment2{{0, 0}, {0.9, 0.9}}, seg));
  BOOST_TEST(!intersects(Segment2{{1.1, 1}, {2, 1}}, seg));
  BOOST_TEST(!intersects(Segment2{{1, 0}, {2, 1}}, seg));
  BOOST_TEST(!intersects(Segment2{{1, 3}, {3, 1.1}}, seg));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_segment_box, Coordinate,
                              CoordinatesList)
{
  using ArborX::Details::intersects;
  using Segment2 = ArborX::Experimental::Segment<2, Coordinate>;

  constexpr ArborX::Box<2, Coordinate> box2{{{0.0, 0.0}}, {{1.0, 1.0}}};

  BOOST_TEST(intersects(Segment2{{0, 0}, {0, 0}}, box2));
  BOOST_TEST(intersects(Segment2{{-1, 1}, {1, -1}}, box2));
  BOOST_TEST(intersects(Segment2{{-1, 0}, {2, 0}}, box2));
  BOOST_TEST(intersects(Segment2{{-1, 0.5}, {0.5, 0.5}}, box2));
  BOOST_TEST(intersects(Segment2{{0.5, 0.5}, {0.5, 2}}, box2));
  BOOST_TEST(intersects(Segment2{{-1, 2}, {2, -1}}, box2));
  BOOST_TEST(intersects(Segment2{{0.5, 2}, {0.5, -1}}, box2));
  BOOST_TEST(intersects(Segment2{{0.4, 0.4}, {0.6, 0.6}}, box2));
  BOOST_TEST(!intersects(Segment2{{0, -1}, {1, -1}}, box2));
  BOOST_TEST(!intersects(Segment2{{0.5, 1.6}, {2, 0}}, box2));
  BOOST_TEST(intersects(box2, Segment2{{-1, 2}, {2, -1}}));
  BOOST_TEST(!intersects(Segment2{{0.5, 1.6}, {2, 0}}, box2));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_point_ellipse, Coordinate,
                              CoordinatesList)
{
  using ArborX::Details::intersects;
  using Point2 = ArborX::Point<2, Coordinate>;

  // ellipsoid [2x^2 - 3xy + 2y^2 <= 1]
  constexpr ArborX::Experimental::Ellipsoid<2, Coordinate> ellipse{
      {1.f, 1.f}, {{2.f, -1.5f}, {-1.5f, 2.f}}};

  BOOST_TEST(intersects(ellipse, Point2{1.f, 1.f}));
  BOOST_TEST(intersects(ellipse, Point2{0.f, 0.f}));
  BOOST_TEST(!intersects(ellipse, Point2{-0.01f, -0.01f}));
  BOOST_TEST(intersects(ellipse, Point2{0.f, 0.5f}));
  BOOST_TEST(!intersects(ellipse, Point2{0.f, 0.6f}));
  BOOST_TEST(intersects(ellipse, Point2{2.f, 2.f}));
  BOOST_TEST(!intersects(ellipse, Point2{0.5f, 1.5f}));
  BOOST_TEST(intersects(ellipse, Point2{1.f, 0.3f}));
  BOOST_TEST(!intersects(ellipse, Point2{1.f, 0.29f}));
  BOOST_TEST(intersects(ellipse, Point2{1.f, 1.69f}));
  BOOST_TEST(intersects(ellipse, Point2{1.f, 1.70f}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_ellipse_segment, Coordinate,
                              CoordinatesList)
{
  using ArborX::Details::intersects;
  using Segment2 = ArborX::Experimental::Segment<2, Coordinate>;

  // ellipsoid [2x^2 - 3xy + 2y^2 <= 1]
  constexpr ArborX::Experimental::Ellipsoid<2, Coordinate> ellipse{
      {1.f, 1.f}, {{2.f, -1.5f}, {-1.5f, 2.f}}};

  BOOST_TEST(intersects(ellipse, Segment2{{1, 1}, {1, 1}}));
  BOOST_TEST(intersects(ellipse, Segment2{{-1, 1}, {1, -1}}));
  BOOST_TEST(!intersects(ellipse, Segment2{{-1.1, 1}, {1, -1}}));
  BOOST_TEST(intersects(ellipse, Segment2{{0, 0}, {0, 1}}));
  BOOST_TEST(intersects(ellipse, Segment2{{0.5, 0.5}, {1.5, 1.5}}));
  BOOST_TEST(intersects(ellipse, Segment2{{0.0, 1.9}, {3.0, 1.9}}));
  BOOST_TEST(!intersects(ellipse, Segment2{{2.1, 0}, {2.1, 3}}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_ellipsoid_box, Coordinate,
                              CoordinatesList)
{
  using ArborX::Details::intersects;
  using Box2 = ArborX::Box<2, Coordinate>;
  using Box3 = ArborX::Box<3, Coordinate>;

  // ellipsoid [2x^2 - 3xy + 2y^2 <= 1] shifted by (1,1)
  constexpr ArborX::Experimental::Ellipsoid<2, Coordinate> ellipse2{
      {1.f, 1.f}, {{2.f, -1.5f}, {-1.5f, 2.f}}};

  BOOST_TEST(intersects(ellipse2, Box2{{-10, -10}, {10, 10}}));
  BOOST_TEST(intersects(ellipse2, Box2{{0.5, 0.5}, {1.0, 1.0}}));
  BOOST_TEST(intersects(ellipse2, Box2{{-1, -1}, {0, 0}}));
  BOOST_TEST(intersects(ellipse2, Box2{{2, 2}, {3, 3}}));
  BOOST_TEST(intersects(ellipse2, Box2{{-1, -1}, {0, 2}}));
  BOOST_TEST(intersects(ellipse2, Box2{{-1, -1}, {2, 0}}));
  BOOST_TEST(intersects(ellipse2, Box2{{1.05, 1.05}, {1.1, 1.1}}));
  BOOST_TEST(intersects(ellipse2, Box2{{2, 1}, {3, 3}}));
  BOOST_TEST(intersects(ellipse2, Box2{{1, 2}, {3, 3}}));
  BOOST_TEST(!intersects(ellipse2, Box2{{1.5, 0}, {2, 0.5}}));
  BOOST_TEST(!intersects(ellipse2, Box2{{-1, -1}, {-0.1, -0.1}}));
  BOOST_TEST(!intersects(ellipse2, Box2{{0, 1.5}, {0.5, 2}}));
  BOOST_TEST(!intersects(ellipse2, Box2{{2.1, 2.1}, {3, 3}}));

  // (spherical) ellipsoid [x^2 + y^2 + z^2 <= 1] shifted by (1,1,1)
  constexpr ArborX::Experimental::Ellipsoid<3, Coordinate> ellipse3{
      {1, 1, 1}, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  constexpr ArborX::Sphere<3, Coordinate> sphere{{1, 1, 1}, 1};

  BOOST_TEST(intersects(ellipse3, Box3{{-1, -1, -1}, {2, 2, 0}}) ==
             intersects(sphere, Box3{{-1, -1, -1}, {2, 2, 0}}));
  BOOST_TEST(
      intersects(ellipse3, Box3{{0.75, 0.75, 0.75}, {1.25, 1.25, 1.25}}) ==
      intersects(sphere, Box3{{0.75, 0.75, 0.75}, {1.25, 1.25, 1.25}}));
  BOOST_TEST(intersects(ellipse3, Box3{{-1, -1, -1}, {3, 3, 3}}) ==
             intersects(sphere, Box3{{-1, -1, -1}, {3, 3, 3}}));
  BOOST_TEST(!intersects(ellipse3, Box3{{0, 0, 0}, {0.25, 0.25, 0.25}}) ==
             !intersects(sphere, Box3{{0, 0, 0}, {0.25, 0.25, 0.25}}));

  // ellipsoid [4x^2 + 4y^2 + 4z^2 - 3xy - 3xz - 3yz <= 1] shifted by (1,1,1)
  constexpr ArborX::Experimental::Ellipsoid<3, Coordinate> ellipse4{
      {1, 1, 1},
      {{4.f, -1.5f, -1.5f}, {-1.5f, 4.f, -1.5f}, {-1.5f, -1.5f, 4.f}}};
  BOOST_TEST(intersects(ellipse4, Box3{{-1, -1, -1}, {0.45, 0.45, 0.45}}));
  BOOST_TEST(intersects(ellipse4, Box3{{-1, -1, -1}, {3, 3, 3}}));
  BOOST_TEST(intersects(ellipse4, Box3{{1, -2, -2}, {3, 2, 2}}));  // left face
  BOOST_TEST(intersects(ellipse4, Box3{{-3, -2, -2}, {1, 2, 2}})); // right face
  BOOST_TEST(intersects(ellipse4, Box3{{-2, 1, -2}, {2, 3, 2}}));  // front face
  BOOST_TEST(intersects(ellipse4, Box3{{-2, -3, -2}, {2, 1, 2}})); // back face
  BOOST_TEST(intersects(ellipse4, Box3{{-2, -2, 1}, {2, 2, 3}})); // bottom face
  BOOST_TEST(intersects(ellipse4, Box3{{-2, -3, -2}, {2, 1, 2}})); // top face
  BOOST_TEST(!intersects(ellipse4, Box3{{0.2, 0.2, 0.2}, {0.3, 0.3, 0.3}}));
  BOOST_TEST(!intersects(ellipse4, Box3{{-1, -1, -1}, {0.4, 0.4, 0.4}}));
  BOOST_TEST(!intersects(ellipse4, Box3{{1.7, 1.7, 1.7}, {2, 2, 2}}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_segment_triangle, Coordinate,
                              CoordinatesList)
{
  using ArborX::Details::intersects;
  using Segment = ArborX::Experimental::Segment<3, Coordinate>;
  using Triangle = ArborX::Triangle<3, Coordinate>;

  constexpr Triangle triangle{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

  BOOST_TEST(intersects(Segment{{0, 0, 0}, {1, 1, 1}}, triangle));
  BOOST_TEST(intersects(Segment{{0, 0, 0}, {1, 0, 0}}, triangle));
  BOOST_TEST(intersects(Segment{{0, 0, 0}, {0, 1, 0}}, triangle));
  BOOST_TEST(intersects(Segment{{0, 0, 0}, {0, 0, 1}}, triangle));
  BOOST_TEST(intersects(Segment{{0, 0, 0}, {0.5, 0.25, 0.25}}, triangle));
  BOOST_TEST(!intersects(Segment{{0, 0, 0}, {0.45, 0.25, 0.25}}, triangle));
  BOOST_TEST(!intersects(Segment{{0.9, 0, 0}, {0, 0, 0.9}}, triangle));
}

using KDOP_2D_types = std::tuple<ArborX::Experimental::KDOP<2, 4, float>,
                                 ArborX::Experimental::KDOP<2, 4, double>,
                                 ArborX::Experimental::KDOP<2, 8, float>,
                                 ArborX::Experimental::KDOP<2, 8, double>>;
using KDOP_3D_types = std::tuple<ArborX::Experimental::KDOP<3, 6, float>,
                                 ArborX::Experimental::KDOP<3, 6, double>,
                                 ArborX::Experimental::KDOP<3, 14, float>,
                                 ArborX::Experimental::KDOP<3, 14, double>,
                                 ArborX::Experimental::KDOP<3, 18, float>,
                                 ArborX::Experimental::KDOP<3, 18, double>,
                                 ArborX::Experimental::KDOP<3, 26, float>,
                                 ArborX::Experimental::KDOP<3, 26, double>>;

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_kdop_kdop_2D, KDOP_t, KDOP_2D_types)
{
  using Coordinate = ArborX::GeometryTraits::coordinate_type_t<KDOP_t>;
  using Point = ArborX::Point<2, Coordinate>;

  KDOP_t x;
  BOOST_TEST(!intersects(x, x));
  expand(x, Point{1, 0});
  expand(x, Point{0, 1});
  BOOST_TEST(intersects(x, x));
  BOOST_TEST(!intersects(x, KDOP_t{}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_kdop_box_2D, KDOP_t, KDOP_2D_types)
{
  using Coordinate = ArborX::GeometryTraits::coordinate_type_t<KDOP_t>;
  using Point = ArborX::Point<2, Coordinate>;
  using Box = ArborX::Box<2, Coordinate>;

  KDOP_t x;
  BOOST_TEST(!intersects(x, Box{}));
  BOOST_TEST(!intersects(x, Box{{0, 0}, {1, 1}}));
  expand(x, Point{1, 0});
  expand(x, Point{0, 1});
  BOOST_TEST(!intersects(x, Box{}));
  BOOST_TEST(intersects(x, Box{{0, 0}, {1, 1}}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_point_kdop_2D, KDOP_t, KDOP_2D_types)
{
  using Coordinate = ArborX::GeometryTraits::coordinate_type_t<KDOP_t>;
  using Point = ArborX::Point<2, Coordinate>;

  constexpr bool is_kdop_2_4 = (KDOP_t::n_directions == 2);

  {
    KDOP_t x;
    BOOST_TEST(!intersects(Point{1, 1}, x));
  }
  {
    KDOP_t x; // rombus
    expand(x, Point{0.5f, 0});
    expand(x, Point{0.5f, 1});
    expand(x, Point{0, 0.5f});
    expand(x, Point{1, 0.5f});
    // unit square corners
    BOOST_TEST(intersects(Point{0, 0}, x) == is_kdop_2_4);
    BOOST_TEST(intersects(Point{1, 0}, x) == is_kdop_2_4);
    BOOST_TEST(intersects(Point{0, 1}, x) == is_kdop_2_4);
    BOOST_TEST(intersects(Point{1, 1}, x) == is_kdop_2_4);
    // rombus corners
    BOOST_TEST(intersects(Point{0.5f, 0}, x));
    BOOST_TEST(intersects(Point{0.5f, 1}, x));
    BOOST_TEST(intersects(Point{0, 0.5f}, x));
    BOOST_TEST(intersects(Point{1, 0.5f}, x));
    // unit square center
    BOOST_TEST(intersects(Point{0.5f, 0.5f}, x));
    // mid rombus diagonals
    BOOST_TEST(intersects(Point{0.75f, 0.25f}, x));
    BOOST_TEST(intersects(Point{0.25f, 0.25f}, x));
    BOOST_TEST(intersects(Point{0.25f, 0.75f}, x));
    BOOST_TEST(intersects(Point{0.75f, 0.75f}, x));
    // slightly outside of the diagonals
    BOOST_TEST(intersects(Point{0.8f, 0.2f}, x) == is_kdop_2_4);
    BOOST_TEST(intersects(Point{0.2f, 0.2f}, x) == is_kdop_2_4);
    BOOST_TEST(intersects(Point{0.2f, 0.8f}, x) == is_kdop_2_4);
    BOOST_TEST(intersects(Point{0.8f, 0.8f}, x) == is_kdop_2_4);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_kdop_kdop_3D, KDOP_t, KDOP_3D_types)
{
  using Coordinate = ArborX::GeometryTraits::coordinate_type_t<KDOP_t>;
  using Point = ArborX::Point<3, Coordinate>;

  KDOP_t x;
  BOOST_TEST(!intersects(x, x));
  expand(x, Point{1, 0, 0});
  expand(x, Point{0, 1, 0});
  expand(x, Point{0, 0, 1});
  BOOST_TEST(intersects(x, x));
  BOOST_TEST(!intersects(x, KDOP_t{}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_kdop_box_3D, KDOP_t, KDOP_3D_types)
{
  using Coordinate = ArborX::GeometryTraits::coordinate_type_t<KDOP_t>;
  using Box = ArborX::Box<3, Coordinate>;
  using Point = ArborX::Point<3, Coordinate>;

  KDOP_t x;
  BOOST_TEST(!intersects(x, Box{}));
  BOOST_TEST(!intersects(x, Box{{0, 0, 0}, {1, 1, 1}}));
  expand(x, Point{1, 0, 0});
  expand(x, Point{0, 1, 0});
  expand(x, Point{0, 0, 1});
  BOOST_TEST(!intersects(x, Box{}));
  BOOST_TEST(intersects(x, Box{{0, 0, 0}, {1, 1, 1}}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(intersects_point_kdop, KDOP_t, KDOP_3D_types)
{
  using Coordinate = ArborX::GeometryTraits::coordinate_type_t<KDOP_t>;
  using Point = ArborX::Point<3, Coordinate>;

  constexpr bool is_kdop_3_6 = (KDOP_t::n_directions == 3);
  constexpr bool is_kdop_3_14 = (KDOP_t::n_directions == 7);
  constexpr bool is_kdop_3_18 = (KDOP_t::n_directions == 9);

  {
    KDOP_t x;
    BOOST_TEST(!intersects(Point{1, 1, 1}, x));
  }
  {
    KDOP_t x; // unit cube with (1,1,0)--(1,1,1) edge chopped away
    // bottom face
    expand(x, Point{0, 0, 0});
    expand(x, Point{1, 0, 0});
    expand(x, Point{1, 0.75, 0});
    expand(x, Point{0.75, 1, 0});
    expand(x, Point{0, 1, 0});
    // top
    expand(x, Point{0, 0, 1});
    expand(x, Point{1, 0, 1});
    expand(x, Point{1, 0.75, 1});
    expand(x, Point{0.75, 1, 1});
    expand(x, Point{0, 1, 1});
    // test intersection with point on the missing edge
    BOOST_TEST(intersects(Point{1, 1, 0.5}, x) ==
               (is_kdop_3_6 || is_kdop_3_14));
    BOOST_TEST(intersects(Point{1, 1, 0.625}, x) ==
               (is_kdop_3_6 || is_kdop_3_14));
    BOOST_TEST(intersects(Point{1, 1, 0.375}, x) ==
               (is_kdop_3_6 || is_kdop_3_14));
    BOOST_TEST(intersects(Point{1, 1, 0.875}, x) == is_kdop_3_6);
    BOOST_TEST(intersects(Point{1, 1, 0.125}, x) == is_kdop_3_6);
    // with both ends of the edge
    BOOST_TEST(intersects(Point{1, 1, 0}, x) == is_kdop_3_6);
    BOOST_TEST(intersects(Point{1, 1, 1}, x) == is_kdop_3_6);
    // with centroid of unit cube
    BOOST_TEST(intersects(Point{0.5, 0.5, 0.5}, x));
    // with some point outside the unit cube
    BOOST_TEST(!intersects(Point{1, 2, 3}, x));
  }
  {
    KDOP_t x; // unit cube with (1,1,1) corner cut off
    // bottom face
    expand(x, Point{0, 0, 0});
    expand(x, Point{1, 0, 0});
    expand(x, Point{1, 1, 0});
    expand(x, Point{0, 1, 0});
    // top
    expand(x, Point{0, 0, 1});
    expand(x, Point{1, 0, 1});
    expand(x, Point{1, 0.75, 1});
    expand(x, Point{0.75, 1, 1});
    expand(x, Point{0, 1, 1});
    // test intersection with center of the missing corner
    BOOST_TEST(intersects(Point{1, 1, 1}, x) == (is_kdop_3_6 || is_kdop_3_18));
    // test with points on the edges out of that corner
    BOOST_TEST(intersects(Point{0.5, 1, 1}, x));
    BOOST_TEST(intersects(Point{0.875, 1, 1}, x) ==
               (is_kdop_3_6 || is_kdop_3_18));
    BOOST_TEST(intersects(Point{1, 0.5, 1}, x));
    BOOST_TEST(intersects(Point{1, 0.875, 1}, x) ==
               (is_kdop_3_6 || is_kdop_3_18));
    BOOST_TEST(intersects(Point{1, 1, 0.5}, x));
    BOOST_TEST(intersects(Point{1, 1, 0.875}, x) ==
               (is_kdop_3_6 || is_kdop_3_18));
    // with centroid of unit cube
    BOOST_TEST(intersects(Point{0.5, 0.5, 0.5}, x));
    // with some point outside the unit cube
    BOOST_TEST(!intersects(Point{1, 2, 3}, x));
  }
}
