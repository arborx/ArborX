/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
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

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(intersects_box)
{
  using ArborX::Box;
  using ArborX::Point;
  using ArborX::Experimental::Ray;

  Box unit_box{{0, 0, 0}, {1, 1, 1}};

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
