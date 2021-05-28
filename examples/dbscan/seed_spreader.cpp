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

#include "seed_spreader.hpp"

#include <ArborX_Exception.hpp>

#include <iostream>
#include <random>

// Seed spreader (SS) generator (as proposed in [1])
//
// Input:
//   d          : dimensionality [int]
//   n          : target cardinality [int]
//   rho_restart: restart probability [float]
//   rho_noise  : noise percentage [float]
//   c_reset    : local counter [int]
//   r_shift    : local shift [float]
//   r_vicinity : radius of the ball to generate local points [float]
//
// Seed spreader move about in space, and spits out data points around the
// current location. Every time the counter c_reset reaches zero, the spreader
// moves over distance r_shift in random direction, and resets the c_reset
// counter.
//
// The spreader works in steps. Each step:
// - with probability rho_restart, the spreader restarts by jumping to a random
//   location in the data space
// - the spreader produces a point uniformly at random in the ball centered at
//   its current location with radius r_vicinity, and decreases c_reset by 1.
// The process repeats n*(1 - rho_noise) steps (generating this many points).
// Then, n*rho_noise random points in the data space added.
//
// Every time a restart happens, the spreader is likely to start generating a
// new cluster.
//
// The underlying data space is [0, 10^5] in every dimension.
//
// The settings from [1]:
//   c_reset = 100
//   rho_noise = 1/10^4
//   rho_restart = 10/(n*(1 - rho_noise))
//
// The values of r_vicinity and r_shift are set in two different ways:
// - Similar density clusters
//   r_vicinity = 100
//   r_shift = 50*d
// - Variable density clusters
//   r_vicinity = 100 * ((i % 10) + 1) [i is the number of restarts taken place]
//   r_shift    = r_vicinity * d/2
//
// [1] J. Gan and Y. Tao. "On the hardness and approximation of Euclidean
//     DBSCAN." ACM Transactions on Database Systems (TODS), 2017.
std::vector<ArborX::Point> seedSpreader(int n, int dim, bool variable_density,
                                        int num_clusters, int c_reset,
                                        float rho_noise)
{
  // TODO: Once ArborX::Point allows any dimension, this constraint can be
  // removed.
  ARBORX_ASSERT(dim <= 3);

  // NOTE: The paper used 10^5, however it seems to produce clusters that are
  // too far from each other. I prefer 10^4, and it looks better when
  // visualized.
  float const L = 1e4;
  float const rho_restart = num_clusters / (n * (1 - rho_noise));
  int const num_different_densities = (variable_density ? 10 : 1);
  int const n_wo_noise = n * (1 - rho_noise);

  double const seed = 2.71;
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<float> distribution(0, 1);
  auto rd = [&distribution, &generator]() { return distribution(generator); };

  std::vector<ArborX::Point> points(n);

  ArborX::Point origin;
  float r_vicinity;
  float r_shift;
  int count;
  bool do_reset = true;
  int num_resets = 0;
  for (int i = 0; i < n_wo_noise; ++i)
  {
    if (do_reset)
    {
      // Jump to a random location in data space
      for (int d = 0; d < dim; ++d)
        origin[d] = rd() * L;

      // Update new cluster density parameters
      // This works for both constact and variable densities
      r_vicinity = 100 * ((num_resets % num_different_densities) + 1);
      r_shift = (r_vicinity * dim) / 2;

      ++num_resets;
      count = c_reset;

      do_reset = false;
    }

    if (count == 0)
    {
      // Move over distance r_shift in a random direction
      ArborX::Point direction;
      float norm = 0.f;
      for (int d = 0; d < dim; ++d)
      {
        direction[d] = 2 * rd() - 1;
        norm += direction[d] * direction[d];
      }
      norm = std::sqrt(norm);
      for (int d = 0; d < dim; ++d)
        origin[d] += direction[d] * r_shift / norm;

      count = c_reset;
    }

    // Choose a point uniformly at random in the ball centered around current
    // location with radius r_vicinity
    auto &new_point = points[i];
    new_point = {0.f, 0.f, 0.f};
    for (int d = 0; d < dim; ++d)
      new_point[d] = origin[d] + (2 * rd() - 1) * r_vicinity;

    --count;

    if (rd() <= rho_restart)
      do_reset = true;
  }

  // Add noise
  for (int i = n_wo_noise; i < n; ++i)
  {
    auto &new_point = points[i];
    new_point = {0.f, 0.f, 0.f};
    for (int d = 0; d < dim; ++d)
      new_point[d] = rd() * L;
  }

  return points;
}
