/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "ArborX_BoostRangeAdapters.hpp"
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_Box.hpp>
#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_Point.hpp>

#include <Kokkos_Core.hpp>

#include <boost/range/algorithm.hpp> // reverse_copy, replace_if, count, generate, count_if
#include <boost/range/algorithm_ext.hpp> // iota
#include <boost/range/numeric.hpp>       // accumulate
#include <boost/test/unit_test.hpp>

#include <random>
#include <sstream>

#define BOOST_TEST_MODULE BoostRangeAdapters

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE(range_algorithms)
{
  Kokkos::View<int[4], Kokkos::HostSpace> w("w");

  boost::iota(w, 0);
  std::stringstream ss;
  boost::reverse_copy(w, std::ostream_iterator<int>(ss, " "));
  BOOST_TEST(ss.str() == "3 2 1 0 ");

  boost::replace_if(w, [](int i) { return (i > 1); }, -1);
  BOOST_TEST(w == std::vector<int>({0, 1, -1, -1}), tt::per_element());

  BOOST_TEST(boost::count(w, -1), 2);
  BOOST_TEST(boost::accumulate(w, 5), 4);
}

BOOST_AUTO_TEST_CASE(point_cloud)
{
  using ArborX::Point;
  using ArborX::Details::distance;
  double const seed = 3.14;
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<double> distribution(-1., 1.);
  int const n = 10000;
  Kokkos::View<Point *, Kokkos::HostSpace> cloud("cloud", n);
  boost::generate(cloud, [&distribution, &generator]() {
    Point p;
    p[0] = distribution(generator);
    p[1] = distribution(generator);
    p[2] = distribution(generator);
    return p;
  });

  Point const origin = {{0., 0., 0.}};
  double const radius = 1.;
  // 4/3 pi 1^3 / 2^3
  double const pi = 6. *
                    static_cast<double>(boost::count_if(
                        cloud,
                        [origin, radius](Point point) {
                          return (distance(point, origin) <= radius);
                        })) /
                    static_cast<double>(n);

  double const relative_tolerance = .05;
  BOOST_TEST(pi == 3.14, tt::tolerance(relative_tolerance));
}
