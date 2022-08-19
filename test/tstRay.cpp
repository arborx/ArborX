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
#include <ArborX_Box.hpp>
#include <ArborX_Ray.hpp>
#include <ArborX_Triangle.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(intersects_box)
{
  using ArborX::Box;
  using ArborX::Point;
  using ArborX::Experimental::Ray;

  constexpr Box unit_box{{0, 0, 0}, {1, 1, 1}};

  // origin is within the box
  BOOST_TEST(intersects(Ray{{.5, .5, .5}, {1, 0, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{.5, .5, .5}, {0, 1, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{.5, .5, .5}, {0, 0, 1}}, unit_box));
  BOOST_TEST(intersects(Ray{{.5, .5, .5}, {1, 1, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{.5, .5, .5}, {1, 1, 1}}, unit_box));

  // origin is outside the box
  // hit the center of the face
  BOOST_TEST(intersects(Ray{{-1, .5, .5}, {1, 0, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{-1, .5, .5}, {-1, 0, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{-1, .5, .5}, {0, 1, 1}}, unit_box));
  BOOST_TEST(intersects(Ray{{2, .5, .5}, {-1, 0, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{2, .5, .5}, {1, 0, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{2, .5, .5}, {0, 1, 1}}, unit_box));
  BOOST_TEST(intersects(Ray{{-1, 1.5, .5}, {1, -1, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{-1, 1.5, .5}, {1, 0, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{-1, 1.5, 1.5}, {1, -1, -1}}, unit_box));
  BOOST_TEST(!intersects(Ray{{-1, 1.5, 1.5}, {1, 0, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{-1, 1.5, 1.5}, {1, -1, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{-1, 1.5, 1.5}, {1, 0, -1}}, unit_box));
  BOOST_TEST(intersects(Ray{{2, -.5, .5}, {-1, 1, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{2, -.5, .5}, {-1, 0, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{2, -.5, 1.5}, {-1, 1, -1}}, unit_box));
  BOOST_TEST(!intersects(Ray{{2, -.5, 1.5}, {-1, 0, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{2, -.5, 1.5}, {-1, 1, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{2, -.5, 1.5}, {-1, 0, -1}}, unit_box));

  // hit the center of an edge
  BOOST_TEST(intersects(Ray{{-1, 0, .5}, {1, 0, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{-1, 0, .5}, {-1, 0, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{-1, 0, .5}, {0, 1, 1}}, unit_box));
  BOOST_TEST(intersects(Ray{{2, 0, .5}, {-1, 0, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{2, 0, .5}, {1, 0, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{2, 0, .5}, {0, 1, 1}}, unit_box));
  BOOST_TEST(intersects(Ray{{-1, -1, .5}, {1, 1, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{-1, 1, .5}, {1, -1, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{-1, -2, .5}, {1, 2, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{2, 2, .5}, {-1, -1, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{2, -1, .5}, {-1, 1, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{2, 1, .5}, {-1, -1, 0}}, unit_box));

  // hit a corner
  BOOST_TEST(intersects(Ray{{-0.5, 1.5, 1.5}, {1, -1, -1}}, unit_box));
  BOOST_TEST(!intersects(Ray{{-1, 1, 1}, {-1, 0, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{-1, 1, 1}, {0, 1, 1}}, unit_box));
  BOOST_TEST(!intersects(Ray{{2, 1, 1}, {1, 0, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{2, 1, 1}, {0, 1, 1}}, unit_box));
  BOOST_TEST(intersects(Ray{{-1, -1, -1}, {1, 1, 1}}, unit_box));
  BOOST_TEST(intersects(Ray{{2, 2, 2}, {-1, -1, -1}}, unit_box));

  BOOST_TEST(!intersects(Ray{{1, 2, 3}, {4, 5, 6}}, unit_box));
  BOOST_TEST(intersects(Ray{{1, 2, 3}, {-1, -2, -3}}, unit_box));

  // origin is on the box (no 0*inf).
  BOOST_TEST(intersects(Ray{{0, 0, 0}, {1, 1, 1}}, unit_box));
  BOOST_TEST(intersects(Ray{{0, 0, 0}, {-1, -1, -1}}, unit_box));
  BOOST_TEST(intersects(Ray{{1, 1, 1}, {-1, -1, -1}}, unit_box));
  BOOST_TEST(intersects(Ray{{1, 1, 1}, {1, 1, 1}}, unit_box));

  BOOST_TEST(intersects(Ray{{1, .5, .5}, {-1, 0, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{1, .5, .5}, {1, 0, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{.5, 1, .5}, {0, -1, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{.5, 1, .5}, {0, 1, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{.5, .5, 1}, {0, 0, -1}}, unit_box));
  BOOST_TEST(intersects(Ray{{.5, .5, 1}, {0, 0, 1}}, unit_box));

  BOOST_TEST(intersects(Ray{{0, 0, .5}, {1, 1, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{0, 0, .5}, {-1, -1, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{1, 1, .5}, {-1, -1, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{1, 1, .5}, {1, 1, 0}}, unit_box));

  BOOST_TEST(intersects(Ray{{0, 0.5, 0}, {-1, -2, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{0.5, 0, 0.5}, {2, 0, -1}}, unit_box));

  // origin is on the box (with 0*inf).
  BOOST_TEST(intersects(Ray{{0, 0, 0}, {0, 1, -1}}, unit_box));
  BOOST_TEST(intersects(Ray{{0, 0, 0}, {-1, 0, -1}}, unit_box));
  BOOST_TEST(intersects(Ray{{0, 0, 0}, {-1, 1, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{0, 0, 0}, {0, 0, 1}}, unit_box));
  BOOST_TEST(intersects(Ray{{0, 0, 0}, {-1, 0, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{0, 0, 0}, {0, -1, 0}}, unit_box));

  BOOST_TEST(intersects(Ray{{1, 1, 1}, {0, -1, -1}}, unit_box));
  BOOST_TEST(intersects(Ray{{1, 1, 1}, {1, 0, 1}}, unit_box));
  BOOST_TEST(intersects(Ray{{1, 1, 1}, {-1, -1, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{1, 1, 1}, {1, 0, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{1, 1, 1}, {0, 1, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{1, 1, 1}, {0, 0, -1}}, unit_box));

  BOOST_TEST(intersects(Ray{{0, 1, 1}, {0, 0, 1}}, unit_box));
  BOOST_TEST(intersects(Ray{{1, 0, 1}, {1, 0, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{1, 1, 0}, {0, 1, 0}}, unit_box));

  BOOST_TEST(intersects(Ray{{0, 0, 1}, {-1, 0, -1}}, unit_box));
  BOOST_TEST(intersects(Ray{{0, 0, 1}, {0, 0, 1}}, unit_box));
  BOOST_TEST(intersects(Ray{{1, 0, 0}, {0, 1, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{1, 0, 0}, {-1, 0, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{0, 1, 0}, {0, 0, 1}}, unit_box));
  BOOST_TEST(intersects(Ray{{0, 1, 0}, {0, 0, -1}}, unit_box));

  // more cases with 0*inf:
  BOOST_TEST(intersects(Ray{{0, 1.5, 1.5}, {0, -1, -1}}, unit_box));
  BOOST_TEST(!intersects(Ray{{0, 1.5, 1.5}, {0, 1, 1}}, unit_box));
  BOOST_TEST(intersects(Ray{{1.5, 0, 1.5}, {-1, 0, -1}}, unit_box));
  BOOST_TEST(!intersects(Ray{{1.5, 0, 1.5}, {1, 0, 1}}, unit_box));
  BOOST_TEST(intersects(Ray{{1.5, 1.5, 0}, {-1, -1, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{1.5, 1.5, 0}, {1, 1, 0}}, unit_box));

  BOOST_TEST(intersects(Ray{{1, 1.5, 1.5}, {0, -1, -1}}, unit_box));
  BOOST_TEST(!intersects(Ray{{1, 1.5, 1.5}, {0, 1, 1}}, unit_box));
  BOOST_TEST(intersects(Ray{{1.5, 1, 1.5}, {-1, 0, -1}}, unit_box));
  BOOST_TEST(!intersects(Ray{{1.5, 1, 1.5}, {1, 0, 1}}, unit_box));
  BOOST_TEST(intersects(Ray{{1.5, 1.5, 1}, {-1, -1, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{1.5, 1.5, 1}, {1, 1, 0}}, unit_box));

  BOOST_TEST(intersects(Ray{{1.5, 0, 0}, {-1, 0, 2}}, unit_box));
  BOOST_TEST(intersects(Ray{{1.5, 0, 0}, {-1, 2, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{1.5, 0, 0}, {-1, 0, 2.1}}, unit_box));
  BOOST_TEST(!intersects(Ray{{1.5, 0, 0}, {-1, 2.1, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{1.5, 1, 1}, {-1, 0, -2}}, unit_box));
  BOOST_TEST(intersects(Ray{{1.5, 1, 1}, {-1, -2, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{1.5, 1, 1}, {-1, 0, -2.1}}, unit_box));
  BOOST_TEST(!intersects(Ray{{1.5, 1, 1}, {-1, -2.1, 0}}, unit_box));

  BOOST_TEST(intersects(Ray{{0, 1.5, 0}, {0, -1, 2}}, unit_box));
  BOOST_TEST(intersects(Ray{{0, 1.5, 0}, {2, -1, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{0, 1.5, 0}, {0, -1, 2.1}}, unit_box));
  BOOST_TEST(!intersects(Ray{{0, 1.5, 0}, {2.1, -1, 0}}, unit_box));
  BOOST_TEST(intersects(Ray{{1, 1.5, 1}, {0, -1, -2}}, unit_box));
  BOOST_TEST(intersects(Ray{{1, 1.5, 1}, {-2, -1, 0}}, unit_box));
  BOOST_TEST(!intersects(Ray{{1, 1.5, 1}, {0, -1, -2.1}}, unit_box));
  BOOST_TEST(!intersects(Ray{{1, 1.5, 1}, {-2.1, -1, 0}}, unit_box));
}

#define ARBORX_TEST_RAY_BOX_INTERSECTION(ray, box, t0_ref, t1_ref)             \
  do                                                                           \
  {                                                                            \
    float t0;                                                                  \
    float t1;                                                                  \
    BOOST_TEST(ArborX::Experimental::intersection(ray, box, t0, t1));          \
    BOOST_TEST(t0 == t0_ref);                                                  \
    BOOST_TEST(t1 == t1_ref);                                                  \
  } while (false)

#define ARBORX_TEST_RAY_BOX_NO_INTERSECTION(ray, box)                          \
  do                                                                           \
  {                                                                            \
    float t0;                                                                  \
    float t1;                                                                  \
    constexpr auto inf = KokkosExt::ArithmeticTraits::infinity<float>::value;  \
    BOOST_TEST(!ArborX::Experimental::intersection(ray, box, t0, t1));         \
    BOOST_TEST((t0 == inf || t1 == -inf));                                     \
  } while (false)

BOOST_AUTO_TEST_CASE(ray_box_intersection, *boost::unit_test::tolerance(1e-6f))
{
  using ArborX::Box;
  using ArborX::Experimental::Ray;

  constexpr Box unit_box{{0, 0, 0}, {1, 1, 1}};

  auto const sqrtf_5 = std::sqrt(5.f);
  auto const sqrtf_3 = std::sqrt(3.f);
  auto const sqrtf_2 = std::sqrt(2.f);

  // clang-format off
  // origin is within the box
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{.5, .5, .5}, {1, 0, 0}}), unit_box, -.5f, .5f);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{.5, .5, .5}, {0, 1, 0}}), unit_box, -.5f, .5f);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{.5, .5, .5}, {0, 0, 1}}), unit_box, -.5f, .5f);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{.5, .5, .5}, {1, 1, 0}}), unit_box, -.5f*sqrtf_2, .5f*sqrtf_2);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{.5, .5, .5}, {1, 1, 1}}), unit_box, -.5f*sqrtf_3, .5f*sqrtf_3);

  // origin is outside the box
  // hit the center of the face
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{-1, .5, .5}, {1, 0, 0}}), unit_box, 1.f, 2.f);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{-1, .5, .5}, {-1, 0, 0}}), unit_box, -2.f, -1.f);
  ARBORX_TEST_RAY_BOX_NO_INTERSECTION((Ray{{-1, .5, .5}, {0, 1, 1}}), unit_box);
  ARBORX_TEST_RAY_BOX_NO_INTERSECTION((Ray{{-1, .5, .5}, {0, 1, 1}}), unit_box);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{2, .5, .5}, {-1, 0, 0}}), unit_box, 1.f, 2.f);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{2, .5, .5}, {1, 0, 0}}), unit_box, -2.f, -1.f);
  ARBORX_TEST_RAY_BOX_NO_INTERSECTION((Ray{{2, .5, .5}, {0, 1, 1}}), unit_box);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{-1, 1.5, .5}, {1, -1, 0}}), unit_box, sqrtf_2, 1.5f*sqrtf_2);
  ARBORX_TEST_RAY_BOX_NO_INTERSECTION((Ray{{-1, 1.5, .5}, {1, 0, 0}}), unit_box);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{-1, 1.5, 1.5}, {1, -1, -1}}), unit_box, sqrtf_3, 1.5f*sqrtf_3);
  ARBORX_TEST_RAY_BOX_NO_INTERSECTION((Ray{{-1, 1.5, 1.5}, {1, 0, 0}}), unit_box);
  ARBORX_TEST_RAY_BOX_NO_INTERSECTION((Ray{{-1, 1.5, 1.5}, {1, -1, 0}}), unit_box);
  ARBORX_TEST_RAY_BOX_NO_INTERSECTION((Ray{{-1, 1.5, 1.5}, {1, 0, -1}}), unit_box);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{2, -.5, .5}, {-1, 1, 0}}), unit_box, sqrtf_2, 1.5f*sqrtf_2);
  ARBORX_TEST_RAY_BOX_NO_INTERSECTION((Ray{{2, -.5, .5}, {-1, 0, 0}}), unit_box);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{2, -.5, 1.5}, {-1, 1, -1}}), unit_box, sqrtf_3, 1.5f*sqrtf_3);
  ARBORX_TEST_RAY_BOX_NO_INTERSECTION((Ray{{2, -.5, 1.5}, {-1, 0, 0}}), unit_box);
  ARBORX_TEST_RAY_BOX_NO_INTERSECTION((Ray{{2, -.5, 1.5}, {-1, 1, 0}}), unit_box);
  ARBORX_TEST_RAY_BOX_NO_INTERSECTION((Ray{{2, -.5, 1.5}, {-1, 0, -1}}), unit_box);

  // hit the center of an edge
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{-1, 0, .5}, {1, 0, 0}}), unit_box, 1.f, 2.f);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{-1, 0, .5}, {-1, 0, 0}}), unit_box, -2.f, -1.f);
  ARBORX_TEST_RAY_BOX_NO_INTERSECTION((Ray{{-1, 0, .5}, {0, 1, 1}}), unit_box);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{2, 0, .5}, {-1, 0, 0}}), unit_box, 1.f, 2.f);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{2, 0, .5}, {1, 0, 0}}), unit_box, -2.f, -1.f);
  ARBORX_TEST_RAY_BOX_NO_INTERSECTION((Ray{{2, 0, .5}, {0, 1, 1}}), unit_box);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{-1, -1, .5}, {1, 1, 0}}), unit_box, sqrtf_2, 2*sqrtf_2);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{-1, 1, .5}, {1, -1, 0}}), unit_box, sqrtf_2, sqrtf_2);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{-1, -2, .5}, {1, 2, 0}}), unit_box, sqrtf_5, 1.5f*sqrtf_5);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{2, 2, .5}, {-1, -1, 0}}), unit_box, sqrtf_2, 2*sqrtf_2);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{2, -1, .5}, {-1, 1, 0}}), unit_box, sqrtf_2, 2*sqrtf_2);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{2, 1, .5}, {-1, -1, 0}}), unit_box, sqrtf_2, sqrtf_2);

  // hit a corner
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{-0.5, 1.5, 1.5}, {1, -1, -1}}), unit_box, .5f*sqrtf_3, 1.5f*sqrtf_3);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{-1, 1, 1}, {-1, 0, 0}}), unit_box,-2.f, -1.f);
  ARBORX_TEST_RAY_BOX_NO_INTERSECTION((Ray{{-1, 1, 1}, {0, 1, 1}}), unit_box);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{2, 1, 1}, {1, 0, 0}}), unit_box, -2.f, -1.f);
  ARBORX_TEST_RAY_BOX_NO_INTERSECTION((Ray{{2, 1, 1}, {0, 1, 1}}), unit_box);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{-1, -1, -1}, {1, 1, 1}}), unit_box, sqrtf_3, 2*sqrtf_3);
  ARBORX_TEST_RAY_BOX_INTERSECTION((Ray{{2, 2, 2}, {-1, -1, -1}}), unit_box, sqrtf_3, 2*sqrtf_3);
  // clang-format on
}

#undef ARBORX_TEST_RAY_BOX_INTERSECTION
#undef ARBORX_TEST_RAY_BOX_NO_INTERSECTION

BOOST_AUTO_TEST_CASE(ray_box_distance)
{
  using ArborX::Box;
  using ArborX::Experimental::Ray;

  constexpr Box unit_box{{0, 0, 0}, {1, 1, 1}};
  constexpr auto inf = KokkosExt::ArithmeticTraits::infinity<float>::value;

  // clang-format off
  // origin is within the box
  BOOST_TEST(ArborX::Experimental::distance(Ray{{.5, .5, .5}, {1, 0, 0}}, unit_box) == 0.f);
  // origin outside box, ray hitting box
  BOOST_TEST(ArborX::Experimental::distance(Ray{{.5, .5, -.5}, {0, 0, 1}}, unit_box) == .5f);
  // origin outside box, ray missing box
  BOOST_TEST(ArborX::Experimental::distance(Ray{{.5, .5, -.5}, {0, 0, -1}}, unit_box) == inf);
}

// NOTE until boost 1.70 need to cast both operands when comparing floating
// points
BOOST_AUTO_TEST_CASE(overlap_distance_sphere,
                     *boost::unit_test::tolerance(1e-6f))
{
  using ArborX::Sphere;
  using ArborX::Experimental::Ray;
  constexpr Sphere unit_sphere{{0, 0, 0}, 1};

  auto const sqrtf_3 = std::sqrt(3.f);
  auto const half_sqrtf_2 = std::sqrt(2.f) / 2;
  auto const half_sqrtf_3 = std::sqrt(3.f) / 2;

  // hit center of the sphere
  BOOST_TEST(overlapDistance(Ray{{-2, 0, 0}, {1, 0, 0}}, unit_sphere) == 2.f);
  BOOST_TEST(overlapDistance(Ray{{0, 3, 0}, {0, -1, 0}}, unit_sphere) == 2.f);
  BOOST_TEST(overlapDistance(Ray{{0, 0, -4}, {0, 0, 1}}, unit_sphere) == 2.f);
  BOOST_TEST(overlapDistance(Ray{{-2, -2, -2}, {1, 1, 1}}, unit_sphere) == 2.f);
  // miss it
  BOOST_TEST(overlapDistance(Ray{{-2, -2, -2}, {1, 0, 0}}, unit_sphere) == 0.f);
  BOOST_TEST(overlapDistance(Ray{{1, 2, 3}, {4, 5, 6}}, unit_sphere) == 0.f);
  // half-radius
  BOOST_TEST(overlapDistance(Ray{{-2, 0, 0.5}, {1, 0, 0}}, unit_sphere) ==
             sqrtf_3);
  BOOST_TEST(overlapDistance(Ray{{0, -0.5, 2}, {0, 0, -1}}, unit_sphere) ==
             sqrtf_3);
  // touch surface but no intersection
  BOOST_TEST(overlapDistance(Ray{{1, -2, 0}, {0, 1, 0}}, unit_sphere) == 0.f);
  BOOST_TEST(overlapDistance(Ray{{2, 2, -1}, {-1, -1, 0}}, unit_sphere) == 0.f);

  // behind the origin
  BOOST_TEST(overlapDistance(Ray{{-2, 0, 0}, {-1, 0, 0}}, unit_sphere) == 0.f);
  BOOST_TEST(overlapDistance(Ray{{0, 2, 2}, {0, 1, 1}}, unit_sphere) == 0.f);

  // origin inside
  BOOST_TEST(overlapDistance(Ray{{0.5, 0, 0}, {0, 1, 1}}, unit_sphere) ==
             half_sqrtf_3);
  BOOST_TEST(overlapDistance(Ray{{0, -half_sqrtf_3, 0}, {0, 0, 1}},
                             unit_sphere) == 0.5f);
  BOOST_TEST(overlapDistance(Ray{{0, 0, half_sqrtf_2}, {1, -1, 0}},
                             unit_sphere) == half_sqrtf_2);
  BOOST_TEST(overlapDistance(Ray{{0, 0.6, 0}, {0, 1, 0}}, unit_sphere) == 0.4f);
  BOOST_TEST(overlapDistance(Ray{{0, 0, 0}, {1, 2, 3}}, unit_sphere) == 1.f);

  // origin on surface
  //   tangent
  BOOST_TEST(overlapDistance(Ray{{0, 0, 1}, {1, -1, 0}}, unit_sphere) == 0.f);
  BOOST_TEST(overlapDistance(Ray{{0, 1, 0}, {2, 0, 3}}, unit_sphere) == 0.f);
  BOOST_TEST(
      overlapDistance(Ray{{half_sqrtf_2, half_sqrtf_2, 0}, {1, -1, 0}},
                      unit_sphere) == 0.f,
      boost::test_tools::tolerance(5e-4f)); // inside at machine precision...
  BOOST_TEST(overlapDistance(Ray{{-half_sqrtf_3, 0.5, 0}, {1, 2, 0}},
                             unit_sphere) == 0.f);
  BOOST_TEST(overlapDistance(Ray{{0.5, half_sqrtf_3, 0}, {2, -1, 0}},
                             unit_sphere) == 0.f);
  //   directed at sphere
  BOOST_TEST(overlapDistance(Ray{{1, 0, 0}, {-2, 0, 0}}, unit_sphere) == 2.f);
  BOOST_TEST(overlapDistance(Ray{{1, 0, 0}, {-2, 0, 0}}, unit_sphere) == 2.f);
  BOOST_TEST(overlapDistance(Ray{{0, -half_sqrtf_2, half_sqrtf_2}, {0, 1, -1}},
                             unit_sphere) == 2.f);
  BOOST_TEST(overlapDistance(Ray{{half_sqrtf_3, 0.5, 0}, {-1, 0, 0}},
                             unit_sphere) == sqrtf_3);
  //   directed away
  BOOST_TEST(overlapDistance(Ray{{0, 0, 1}, {1, 1, 1}}, unit_sphere) == 0.f);
  BOOST_TEST(overlapDistance(Ray{{half_sqrtf_3, 0.5, 0}, {1, 0, 0}},
                             unit_sphere) == 0.f);
}

#define ARBORX_TEST_RAY_SPHERE_INTERSECTION(ray, sphere, t0_ref, t1_ref)       \
  do                                                                           \
  {                                                                            \
    float t0;                                                                  \
    float t1;                                                                  \
    BOOST_TEST(ArborX::Experimental::intersection(ray, sphere, t0, t1));       \
    BOOST_TEST(t0 == t0_ref);                                                  \
    BOOST_TEST(t1 == t1_ref);                                                  \
  } while (false)

#define ARBORX_TEST_RAY_SPHERE_NO_INTERSECTION(ray, sphere)                    \
  do                                                                           \
  {                                                                            \
    float t0;                                                                  \
    float t1;                                                                  \
    constexpr auto inf = KokkosExt::ArithmeticTraits::infinity<float>::value;  \
    BOOST_TEST(!ArborX::Experimental::intersection(ray, sphere, t0, t1));      \
    BOOST_TEST((t0 == inf && t1 == -inf));                                     \
  } while (false)

BOOST_AUTO_TEST_CASE(ray_sphere_intersection,
                     *boost::unit_test::tolerance(1e-6f))
{
  using ArborX::Sphere;
  using ArborX::Experimental::Ray;

  constexpr Sphere unit_sphere{{0, 0, 0}, 1};

  auto const sqrtf_3 = std::sqrt(3.f);
  auto const sqrtf_2 = std::sqrt(2.f);

  // clang-format off
  // hit center of the sphere
  ARBORX_TEST_RAY_SPHERE_INTERSECTION((Ray{{-2, 0, 0}, {1, 0, 0}}), unit_sphere, 1.f, 3.f);
  ARBORX_TEST_RAY_SPHERE_INTERSECTION((Ray{{0, 3, 0}, {0, -1, 0}}), unit_sphere, 2.f, 4.f);
  ARBORX_TEST_RAY_SPHERE_INTERSECTION((Ray{{0, 0, -4}, {0, 0, 1}}), unit_sphere, 3.f, 5.f);
  ARBORX_TEST_RAY_SPHERE_INTERSECTION((Ray{{-2, -2, -2}, {1, 1, 1}}), unit_sphere, 2*sqrtf_3-1, 2*sqrtf_3+1);
  ARBORX_TEST_RAY_SPHERE_INTERSECTION((Ray{{1, 0, 0}, {1, 0, 0}}), unit_sphere, -2.f, 0.f);
  ARBORX_TEST_RAY_SPHERE_INTERSECTION((Ray{{0, 3, 0}, {0, 1, 0}}), unit_sphere, -4.f, -2.f);
  ARBORX_TEST_RAY_SPHERE_INTERSECTION((Ray{{0, 0, -4}, {0, 0, -1}}), unit_sphere, -5.f, -3.f);
  ARBORX_TEST_RAY_SPHERE_INTERSECTION((Ray{{2, 2, 2}, {1, 1, 1}}), unit_sphere, -2*sqrtf_3-1, -2*sqrtf_3+1);

  ARBORX_TEST_RAY_SPHERE_NO_INTERSECTION((Ray{{-2, -2, -2}, {1, 0, 0}}), unit_sphere);

  // half-radius
  ARBORX_TEST_RAY_SPHERE_INTERSECTION((Ray{{-2, 0, 0.5}, {1, 0, 0}}), unit_sphere, 2-sqrtf_3/2, 2+sqrtf_3/2);
  ARBORX_TEST_RAY_SPHERE_INTERSECTION((Ray{{0, -0.5, 2}, {0, 0, -1}}), unit_sphere, 2-sqrtf_3/2, 2+sqrtf_3/2);

 // touch surface
 ARBORX_TEST_RAY_SPHERE_INTERSECTION((Ray{{1, -2, 0}, {0, 1, 0}}), unit_sphere, 2.f, 2.f);
 // ARBORX_TEST_RAY_SPHERE_INTERSECTION((Ray{{2, 2, -1}, {-1, -1, 0}}), unit_sphere, 2*sqrtf_2, 2*sqrtf_2);  // machine precision

 // behind the origin
 ARBORX_TEST_RAY_SPHERE_INTERSECTION((Ray{{-2, 0, 0}, {-1, 0, 0}}), unit_sphere, -3.f, -1.f);
 ARBORX_TEST_RAY_SPHERE_INTERSECTION((Ray{{0, 2, 2}, {0, 1, 1}}), unit_sphere, -2*sqrtf_2-1, -2*sqrtf_2+1);
  // clang-format on
}

#undef ARBORX_TEST_RAY_SPHERE_INTERSECTION
#undef ARBORX_TEST_RAY_SPHERE_NO_INTERSECTION

BOOST_AUTO_TEST_CASE(intersects_triangle)
{
  using ArborX::Experimental::Ray;
  using ArborX::Experimental::Triangle;
  constexpr Triangle unit_triangle{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};

  BOOST_TEST(intersects(Ray{{.1, .2, .3}, {0, 0, -1}}, unit_triangle));
  BOOST_TEST(intersects(Ray{{1.1, 1.2, 1}, {-1, -1, -1}}, unit_triangle));
  BOOST_TEST(intersects(Ray{{-1.9, 3.2, -1}, {2, -3, 1}}, unit_triangle));
  BOOST_TEST(!intersects(Ray{{1, 2, 3}, {1, 1, 0}}, unit_triangle));
  BOOST_TEST(!intersects(Ray{{1, 2, 3}, {1, 0, 0}}, unit_triangle));
  BOOST_TEST(!intersects(Ray{{1, 2, 3}, {0, 1, 0}}, unit_triangle));

  // ray origin on the triangle
  BOOST_TEST(intersects(Ray{{.1, .2, 0}, {0, 0, 1}}, unit_triangle));
  BOOST_TEST(intersects(Ray{{.1, .2, 0}, {0, 0, -1}}, unit_triangle));
  BOOST_TEST(intersects(Ray{{.1, .2, 0}, {1, 2, 3}}, unit_triangle));

  // ray directed away from the triangle
  BOOST_TEST(!intersects(Ray{{.1, .2, .3}, {0, 0, 1}}, unit_triangle));
  BOOST_TEST(!intersects(Ray{{1.0, 1.0, 0.0}, {0, 1, 0}}, unit_triangle));

  // ray in the same plane as the triangle
  BOOST_TEST(intersects(Ray{{.3, .3, 0}, {1, 1, 0}}, unit_triangle));
  BOOST_TEST(intersects(Ray{{-0.1, 0, 0}, {1, 0, 0}}, unit_triangle));
  BOOST_TEST(intersects(Ray{{0.1, -0.2, 0}, {0, 1, 0}}, unit_triangle));
  BOOST_TEST(!intersects(Ray{{-1, -1, 0}, {0, 1, 0}}, unit_triangle));

  BOOST_TEST(intersects(Ray{{1.0, 0.0, 0.0}, {-1, 1, 0}}, unit_triangle));
  BOOST_TEST(intersects(Ray{{1.0, 0.0, 0.0}, {1, 2, 3}}, unit_triangle));

  // ray misses the triangle
  BOOST_TEST(!intersects(Ray{{-1, 2, -3}, {0, 0, 1}}, unit_triangle));

  // ray hits vertices
  BOOST_TEST(intersects(Ray{{0, 0, -1}, {0, 0, 1}}, unit_triangle));
  BOOST_TEST(intersects(Ray{{1, 0, -2}, {0, 0, 1}}, unit_triangle));
  BOOST_TEST(intersects(Ray{{0, 1, -3}, {0, 0, 1}}, unit_triangle));
  BOOST_TEST(intersects(Ray{{1, 2, 3}, {-1, -2, -3}}, unit_triangle));
  BOOST_TEST(intersects(Ray{{1, 2, 3}, {0, -2, -3}}, unit_triangle));
  BOOST_TEST(intersects(Ray{{1, 2, 3}, {-1, -1, -3}}, unit_triangle));

  // ray hits edges
  BOOST_TEST(intersects(Ray{{.1, 0, -1}, {0, 0, 1}}, unit_triangle));
  BOOST_TEST(intersects(Ray{{0, .2, -2}, {0, 0, 1}}, unit_triangle));
  BOOST_TEST(intersects(Ray{{.5, .5, -3}, {0, 0, 1}}, unit_triangle));

  // ray in a plane parallel to the triangle
  BOOST_TEST(!intersects(Ray{{-0.1, 0, 1}, {1, 0, 0}}, unit_triangle));

  constexpr Triangle tilted_triangle{{0, 0, 0}, {2, 0, 1}, {0, 2, 1}};

  // ray in the same plane as the triangle
  BOOST_TEST(!intersects(Ray{{10, 0, 0}, {1, 1, 1}}, tilted_triangle));
  BOOST_TEST(intersects(Ray{{3, 3, 3}, {-1, -1, -1}}, tilted_triangle));

  // ray in a plane parallel to the triangle
  BOOST_TEST(!intersects(Ray{{0., 0., 0.1}, {1, 1, 1}}, tilted_triangle));
}

#define ARBORX_TEST_RAY_TRIANGLE_INTERSECTION(ray, triangle, t0_ref, t1_ref)   \
  do                                                                           \
  {                                                                            \
    float t0;                                                                  \
    float t1;                                                                  \
    BOOST_TEST(ArborX::Experimental::intersection(ray, triangle, t0, t1));     \
    BOOST_TEST(t0 == t0_ref);                                                  \
    BOOST_TEST(t1 == t1_ref);                                                  \
  } while (false)

#define ARBORX_TEST_RAY_TRIANGLE_NO_INTERSECTION(ray, triangle)                \
  do                                                                           \
  {                                                                            \
    float t0;                                                                  \
    float t1;                                                                  \
    constexpr auto inf = KokkosExt::ArithmeticTraits::infinity<float>::value;  \
    BOOST_TEST(!ArborX::Experimental::intersection(ray, triangle, t0, t1));    \
    BOOST_TEST((t0 == inf && t1 == -inf));                                     \
  } while (false)

BOOST_AUTO_TEST_CASE(ray_triangle_intersection,
                     *boost::unit_test::tolerance(2e-6f))
{
  using ArborX::Point;
  using ArborX::Experimental::Ray;
  using ArborX::Experimental::Triangle;

  constexpr Triangle unit_triangle{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  constexpr Triangle narrow_triangle{{0.5, 0.5, 0}, {0.24, 0.74, 0}, {0, 1, 0}};

  auto const sqrtf_3 = std::sqrt(3.f);
  auto const sqrtf_2 = std::sqrt(2.f);

  // clang-format off
  // intersection forward, ray is perpendicular to the triangle
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{0.01, 0.2, -1.0}, {0, 0, 1}}), unit_triangle, 1.f, 1.f);
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{0.01, 0.2, 1.0}, {0, 0, -1}}), unit_triangle, 1.f, 1.f);
  // intersection backward, ray is perpendicular to the triangle
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{0.01, 0.2, -1.0}, {0, 0, -1}}), unit_triangle, -1.f, -1.f);
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{0.01, 0.2, 1.0}, {0, 0, 1}}), unit_triangle, -1.f, -1.f);
  // ray in a plane parallel to the triangle
  ARBORX_TEST_RAY_TRIANGLE_NO_INTERSECTION((Ray{{0.01, 0.2, -1.0}, {1, 0, 0}}), unit_triangle);
  // intersection forward
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{0.0, 0.0, -0.3}, {1, 1, 1}}), unit_triangle, 0.3f*sqrtf_3, 0.3f*sqrtf_3);
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{0.0, 0.3, -0.3}, {1, 0, 1}}), unit_triangle, 0.3f*sqrtf_2, 0.3f*sqrtf_2);
  // intersection backward
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{0.0, 0.0, -0.3}, {-1, -1, -1}}), unit_triangle, -0.3f*sqrtf_3, -0.3f*sqrtf_3);
  // ray intersection forward with edges
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{-1.0, 0.0, 0.0}, {1, 0, 0}}), unit_triangle, 1.f, 2.f);
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{-1.0, 2.0, 0.0}, {1, -1, 0}}), unit_triangle, sqrtf_2, 2.0f*sqrtf_2);
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{2.0, -1.0, 0.0}, {-1, 1, 0}}), unit_triangle, sqrtf_2, 2.0f*sqrtf_2);
  // ray intersection backward with edges
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{-1.0, 2.0, 0.0}, {-1, 1, 0}}), unit_triangle, -sqrtf_2, -2.0f*sqrtf_2);
  // ray origin on the edge
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{0.5,  0.5, 0.0}, {-1, 1, 0}}), unit_triangle, 0.f, 0.f);
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{1.0,  0.0, 0.0}, {-1, 1, 0}}), unit_triangle, 0.f, 0.f);
  // ray origin on the vertice
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{0.0,  1.0, 0.0}, {-1, 1, 0}}), unit_triangle, 0.f, 0.f);
  // ray intersection backward with the vertice
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{1.0,  1.0, 0.0}, {0, 1, 0}}), unit_triangle, -1.f, -1.f);

  // ray coplanar to the triangle misses
  ARBORX_TEST_RAY_TRIANGLE_NO_INTERSECTION((Ray{{1.0,  1.0, 0.0}, {-1, 1, 0}}), unit_triangle);

  // ray coplanar to the triangle
  auto const sqrtf_1p01=std::sqrt(1.01f);
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{-4.0, 0.5, 0.0}, {5.0, -0.5, 0.0}}), unit_triangle, 4.f*sqrtf_1p01, 5.f*sqrtf_1p01);

  //narrow_triangle
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{-1.0, 2.0, 0.0}, {1, -1, 0}}), narrow_triangle, sqrtf_2, 1.5f*sqrtf_2);
  ARBORX_TEST_RAY_TRIANGLE_NO_INTERSECTION((Ray{{-1.0, 2.0, 0.0}, {1, -1.02, 0}}), narrow_triangle);

  // a pyramid-shape test
  // These tests are inspired by the Fig. 1 in the paper [1] Woop, S, et al. (2013),
  // the left subfigure shows the crack in the middle where the edges meet, and the right
  // subfigure shows that the proposed algorithm fixed the issue. In their own word,
  // "Plucker coordinates guarantee watertightness along the edges, but edges do not meet
  // exactly at the vertices. The algorithm described in this paper fixed this issue, and
  // guarantees watertightness along the edges and at the vertices". I assume the below
  // test would fail the "Plucker coordinates"-based algorithm, thus it is necessary to
  // keep them here to show the watertightness of the current algorithm implemented.
  constexpr Point O{1.0, 1.0, 1.0};
  constexpr Point A{2.0, 2.0, 0.0};
  constexpr Point B{2.0, -1.0, 0.0};
  constexpr Point C{-1.0, -1.0, 0.0};
  constexpr Point D{-1.0, 2.0f, 0.0f};

  constexpr Triangle triangle_up{D, A, O};
  constexpr Triangle triangle_right{O, A, B};
  constexpr Triangle triangle_down{B, O, C};
  constexpr Triangle triangle_left{C, D, O};

  // ray hits the vertice shared by four triangles
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{0.0, 0.0, 0.0}, {1, 1, 1}}), triangle_up, sqrtf_3, sqrtf_3);
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{0.0, 0.0, 0.0}, {1, 1, 1}}), triangle_right, sqrtf_3, sqrtf_3);
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{0.0, 0.0, 0.0}, {1, 1, 1}}), triangle_down, sqrtf_3, sqrtf_3);
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{0.0, 0.0, 0.0}, {1, 1, 1}}), triangle_left, sqrtf_3, sqrtf_3);

  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{0.0, 0.0, 0.0}, {-1, -1, -1}}), triangle_up, -sqrtf_3, -sqrtf_3);

  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{2.0, 2.0, 2.0}, {-1, -1, -1}}), triangle_up, sqrtf_3, sqrtf_3);
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{2.0, 2.0, 2.0}, {-1, -1, -1}}), triangle_right, sqrtf_3, sqrtf_3);
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{2.0, 2.0, 2.0}, {-1, -1, -1}}), triangle_down, sqrtf_3, sqrtf_3);
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{2.0, 2.0, 2.0}, {-1, -1, -1}}), triangle_left, sqrtf_3, sqrtf_3);

  // ray hits one edge
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{0.0, 0.0, 0.0}, {0, 0, 1}}), triangle_down, 0.5f, 0.5f);
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{0.0, -1.0, 0.5}, {0, 1, 0}}), triangle_down, 1.0f, 1.0f);
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{0.0, 0.0, 0.0}, {0, 0, 1}}), triangle_left, 0.5f, 0.5f);
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{1.0, 1.0, 0.0}, {1, 1, 1}}), triangle_up, 0.5f*sqrtf_3, 0.5f*sqrtf_3);
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{1.0, 1.0, 0.0}, {1, 1, 1}}), triangle_right, 0.5f*sqrtf_3, 0.5f*sqrtf_3);

  // Problem with extreme sizes (compared to the above tests):
  // In the original algorithm, the u, v, w scale with the size of the problem, which
  // leads to precision problems. The problem is fixed by normalizing the distances
  // between the origin of the ray and the vertices.
  // These tests will fail if there is no normalization.
  float const size_s = 0.0001;
  Triangle small_triangle{{-size_s, 0, 0}, {0, size_s, 0}, {0, 0, size_s}};
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{0.0, 0.0, 0.0}, {-size_s, size_s, size_s}}), small_triangle, size_s/sqrtf_3, size_s/sqrtf_3);
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{-2.f*size_s, -size_s, 0.0}, {size_s, size_s, 0}}), small_triangle, sqrtf_2*size_s, 2.f*sqrtf_2*size_s);

  float const size_l = 10000;
  Triangle large_triangle{{-size_l, 0, 0}, {0, 10000, 0}, {0, 0, 10000}};
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{0.0, 0.0, 0.0}, {-size_l, size_l, size_l}}), large_triangle, size_l/sqrtf_3, size_l/sqrtf_3);
  ARBORX_TEST_RAY_TRIANGLE_INTERSECTION((Ray{{-2.f*size_l, -size_l, 0.0}, {size_l, size_l, 0}}), large_triangle, sqrtf_2*size_l, 2.f*sqrtf_2*size_l);

  // clang-format on
}

#undef ARBORX_TEST_RAY_TRIANGLE_INTERSECTION
#undef ARBORX_TEST_RAY_TRIANGLE_NO_INTERSECTION

BOOST_AUTO_TEST_CASE(make_euclidean_vector)
{
  using ArborX::Experimental::makeVector;
  using ArborX::Experimental::Vector;
  static_assert(makeVector({0, 0, 0}, {1, 2, 3}) == Vector{1, 2, 3});
  static_assert(makeVector({1, 2, 3}, {4, 5, 6}) == Vector{3, 3, 3});
}

BOOST_AUTO_TEST_CASE(dot_product)
{
  using ArborX::Experimental::dotProduct;
  static_assert(dotProduct({1, 0, 0}, {1, 0, 0}) == 1);
  static_assert(dotProduct({1, 0, 0}, {0, 1, 0}) == 0);
  static_assert(dotProduct({1, 0, 0}, {0, 0, 1}) == 0);
  static_assert(dotProduct({1, 1, 1}, {1, 1, 1}) == 3);
}

BOOST_AUTO_TEST_CASE(cross_product)
{
  using ArborX::Experimental::crossProduct;
  using ArborX::Experimental::Vector;
  static_assert(crossProduct({1, 0, 0}, {1, 0, 0}) == Vector{0, 0, 0});
  static_assert(crossProduct({1, 0, 0}, {0, 1, 0}) == Vector{0, 0, 1});
  static_assert(crossProduct({1, 0, 0}, {0, 0, 1}) == Vector{0, -1, 0});
  static_assert(crossProduct({0, 1, 0}, {1, 0, 0}) == Vector{0, 0, -1});
  static_assert(crossProduct({0, 1, 0}, {0, 1, 0}) == Vector{0, 0, 0});
  static_assert(crossProduct({0, 1, 0}, {0, 0, 1}) == Vector{1, 0, 0});
  static_assert(crossProduct({1, 1, 1}, {1, 1, 1}) == Vector{0, 0, 0});
}
#undef static_assert
