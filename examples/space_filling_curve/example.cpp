/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>
#include <random>
#include <vector>

#include <filling_curves.hh>

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using hc = flecsi::hilbert_curve_u<3, uint64_t>;
  using mc = flecsi::morton_curve_u<3, uint64_t>;
  using point_t = flecsi::space_vector_u<double, 3>;
  using range_t = std::array<point_t, 2>;
  range_t range = {point_t{0., 0., 0.}, point_t{1., 1., 1.}};

  std::vector<point_t> points = {
      point_t{.25, .25, .25}, point_t{.25, .25, .5}, point_t{.25, .5, .5},
      point_t{.25, .5, .25},  point_t{.5, .5, .25},  point_t{.5, .5, .5},
      point_t{.5, .25, .5},   point_t{.5, .25, .25}, point_t{0., 0., 0.},
      point_t{0., 0., 1.},    point_t{0., 1., 0.},   point_t{0., 1., 1.},
      point_t{1., 0., 0.},    point_t{1., 0., 1.},   point_t{1., 1., 0.},
      point_t{1., 1., 1.}};
  int const n = points.size();
  for (int i = 0; i < n; ++i)
  {
    std::cout << "   "
              << ArborX::Details::morton3D(points[i][0], points[i][1],
                                           points[i][2])
              << "  ";
    std::cout << "   " << mc{range, points[i]}.value() << "  ";
    std::cout << "   " << hc{range, points[i]}.value() << "  ";
  }

  return 0;
}
