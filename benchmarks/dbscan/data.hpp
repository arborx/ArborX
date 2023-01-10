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

#ifndef DATA_HPP
#define DATA_HPP

#include <ArborX_Exception.hpp>
#include <ArborX_HyperPoint.hpp>

#include <random>
#include <vector>

using ArborX::ExperimentalHyperGeometry::Point;

template <int DIM>
std::vector<Point<DIM>> sampleData(std::vector<Point<DIM>> const &data,
                                   int num_samples)
{
  std::vector<Point<DIM>> sampled_data(num_samples);

  std::default_random_engine generator(17);
  std::uniform_int_distribution distribution;
  auto rand = [&distribution, &generator]() { return distribution(generator); };

  // Knuth algorithm
  int const N = data.size();
  int const M = num_samples;
  for (int in = 0, im = 0; in < N && im < M; ++in)
  {
    int rn = N - in;
    int rm = M - im;
    if (rand() % rn < rm)
      sampled_data[im++] = data[in];
  }
  return sampled_data;
}

template <int DIM>
std::vector<Point<DIM>> loadData(std::string const &filename,
                                 bool binary = true, int max_num_points = -1,
                                 int num_samples = -1)
{
  std::cout << "Reading in \"" << filename << "\" in "
            << (binary ? "binary" : "text") << " mode...";
  std::cout.flush();

  std::ifstream input;
  if (!binary)
    input.open(filename);
  else
    input.open(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  std::vector<Point<DIM>> v;

  int num_points = 0;
  int dim = 0;
  if (!binary)
  {
    input >> num_points;
    input >> dim;
  }
  else
  {
    input.read(reinterpret_cast<char *>(&num_points), sizeof(int));
    input.read(reinterpret_cast<char *>(&dim), sizeof(int));
  }

  ARBORX_ASSERT(dim == DIM);

  if (max_num_points > 0 && max_num_points < num_points)
    num_points = max_num_points;

  v.resize(num_points);
  if (!binary)
  {
    auto it = std::istream_iterator<float>(input);
    for (int i = 0; i < num_points; ++i)
      for (int d = 0; d < DIM; ++d)
        v[i][d] = *it++;
  }
  else
  {
    // Directly read into a point
    input.read(reinterpret_cast<char *>(v.data()),
               num_points * sizeof(Point<DIM>));
  }
  input.close();
  std::cout << "done\nRead in " << num_points << " " << dim << "D points"
            << std::endl;

  if (num_samples > 0 && num_samples < (int)v.size())
    v = sampleData(v, num_samples);

  return v;
}

template <int DIM, typename Generator>
auto randomDomainPoint(Generator &generator, float L)
{
  std::uniform_real_distribution<float> distribution(0.f, 1.f);
  auto rd = [&distribution, &generator]() { return distribution(generator); };

  Point<DIM> point;
  for (int d = 0; d < DIM; ++d)
    point[d] = rd() * L;

  return point;
}

template <int DIM, typename Generator>
auto randomBallPoint(Generator &generator, Point<DIM> const &center,
                     float radius)
{
  std::uniform_real_distribution<float> distribution(-1.f, 1.f);
  auto rd = [&distribution, &generator]() { return distribution(generator); };

  Point<DIM> p;
  float norm2;
  do
  {
    norm2 = 0.f;
    for (int d = 0; d < DIM; ++d)
    {
      p[d] = rd();
      norm2 += p[d] * p[d];
    }
  } while (norm2 > 1.f);
  for (int d = 0; d < DIM; ++d)
    p[d] = center[d] + p[d] * radius;
  return p;
}

template <int DIM, typename Generator>
auto randomShiftPoint(Generator &generator, Point<DIM> const &center,
                      float radius)
{
  std::normal_distribution<float> distribution(0.f, 1.f);
  auto rd = [&distribution, &generator]() { return distribution(generator); };

  Point<DIM> direction;
  float norm = 0.f;
  for (int d = 0; d < DIM; ++d)
  {
    direction[d] = rd();
    norm += direction[d] * direction[d];
  }
  norm = std::sqrt(norm);

  Point<DIM> p;
  for (int d = 0; d < DIM; ++d)
    p[d] = center[d] + (direction[d] / norm) * radius;

  return p;
}

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
// current location. Every time the counter reaches zero, the spreader moves
// over distance r_shift in random direction, and resets the counter to c_reset.
//
// The spreader works in steps. Each step:
// - with probability rho_restart, the spreader restarts by jumping to a random
//   location in the data space, and resets the counter to c_reset
// - the spreader produces a point uniformly at random in the ball centered at
//   its current location with radius r_vicinity, and decreases the counter
//   by 1.
// The process repeats n*(1 - rho_noise) steps (generating this many points).
// Then, n*rho_noise random points in the data space added (uar).
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
//   r_vicinity = 100 * ((i % 10) + 1) [i is a number of restarts taken place]
//   r_shift    = r_vicinity * d/2
//
// [1] J. Gan and Y. Tao. "On the hardness and approximation of Euclidean
//     DBSCAN." ACM Transactions on Database Systems (TODS), 2017.
template <int DIM>
std::vector<Point<DIM>> GanTao(int n, bool variable_density = false,
                               int num_clusters = 10, int c_reset = 100,
                               float rho_noise = 1e-4)
{
  // FIXME
  float const L = 1e6;
  int const n_wo_noise = n - (n * rho_noise);
  int const num_different_densities = (variable_density ? 10 : 1);
  double const rho_restart = double(num_clusters - 1) / n_wo_noise;

  // This have to be independent to not introduce bias
  std::default_random_engine generator_ball(1);
  std::default_random_engine generator_shift(2);
  std::default_random_engine generator_center(3);
  std::default_random_engine generator_restart(4);

  std::uniform_real_distribution<double> uniform_double_distribution(0.f, 1.f);
  auto rd_restart = [&uniform_double_distribution, &generator_restart]() {
    return uniform_double_distribution(generator_restart);
  };

  auto random_point_in_domain = [&generator_center, L]() {
    return randomDomainPoint<DIM>(generator_center, L);
  };
  auto random_point_shift = [&generator_shift](auto const &c, float r) {
    return randomShiftPoint(generator_shift, c, r);
  };
  auto random_point_in_ball = [&generator_ball](auto const &c, float r) {
    return randomBallPoint(generator_ball, c, r);
  };

  std::vector<Point<DIM>> points(n);

  Point<DIM> origin;
  float r_vicinity;
  float r_shift;
  int count;
  bool do_restart = true;
  int num_restarts = 0;
  for (int i = 0; i < n_wo_noise; ++i)
  {
    if (do_restart)
    {
      // Jump to a random location in data space
      origin = random_point_in_domain();

      // Update new cluster density parameters (the formulas works for both
      // constant and variable densities).
      r_vicinity = 100 * ((num_restarts % num_different_densities) + 1);
      r_shift = (r_vicinity * DIM) / 2;

      ++num_restarts;
      count = c_reset;

      do_restart = false;
    }

    if (count == 0)
    {
      // Move over distance r_shift in a random direction
      origin = random_point_shift(origin, r_shift);

      count = c_reset;
    }

    // Choose a point in the ball centered around current location with radius
    // r_vicinity
    points[i] = random_point_in_ball(origin, r_vicinity);
    --count;

    if (rd_restart() < rho_restart)
      do_restart = true;
  }

  // Add noise
  for (int i = n_wo_noise; i < n; ++i)
    points[i] = random_point_in_domain();

  std::cout << "Generated " << n << " " << DIM << "D points" << std::endl;

  return points;
}

#endif
