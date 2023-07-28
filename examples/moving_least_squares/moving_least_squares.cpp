/****************************************************************************
 * Copyright (c) 2017-2023 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

// Example taken from DataTransferKit
// (https://github.com/ORNL-CEES/DataTransferKit)

#include <ArborX.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

std::ostream &operator<<(std::ostream &os, ArborX::Point const &p)
{
  os << '(' << p[0] << ',' << p[1] << ',' << p[2] << ')';
  return os;
}

class RBFWendland_0
{
public:
  RBFWendland_0(double radius)
      : _radius(radius)
  {}

  KOKKOS_INLINE_FUNCTION double operator()(double x)
  {
    x /= _radius;
    return (1. - x) * (1. - x);
  }

private:
  double _radius;
};

struct MVPolynomialBasis_Quad_3D
{
  static constexpr std::size_t size = 10;

  template <typename Double3D>
  KOKKOS_INLINE_FUNCTION Kokkos::Array<double, size>
  operator()(Double3D const &p) const
  {
    return {{1., p[0], p[1], p[2], p[0] * p[0], p[0] * p[1], p[0] * p[2],
             p[1] * p[1], p[1] * p[2], p[2] * p[2]}};
  }
};

// Func to evaluate
template <typename Double3D>
KOKKOS_INLINE_FUNCTION double func(Double3D const &p) {
  return Kokkos::sin(p[0]) * Kokkos::cos(p[1]) + p[2];
} 

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  constexpr float cube_half_side = 10.;              // [-10, 10]^3 cube
  constexpr float cube_side = 2 * cube_half_side;
  constexpr std::size_t source_points_side = 100; // [-10, 10]^3 grid
  constexpr std::size_t target_points_num = 10'000;   // random [-10, 10]^3

  constexpr std::size_t source_points_num =
    source_points_side * source_points_side * source_points_side;

  auto source_points = Kokkos::View<ArborX::Point *, MemorySpace>(
    "source_points", source_points_num);
  auto target_points = Kokkos::View<ArborX::Point *, MemorySpace>(
    "target_points", target_points_num);

  // Generate source points
  Kokkos::parallel_for(
    "source_fill",
    Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>>(
      {0, 0, 0},
      {source_points_side, source_points_side, source_points_side}),
    KOKKOS_LAMBDA (int const i, int const j, int const k) {
      source_points(
        i * source_points_side * source_points_side +
        j * source_points_side +
        k
      ) = ArborX::Point {
        (static_cast<float>(i) / (source_points_side - 1) - .5f) * cube_side,
        (static_cast<float>(j) / (source_points_side - 1) - .5f) * cube_side,
        (static_cast<float>(k) / (source_points_side - 1) - .5f) * cube_side
      };
  });

  // Generate target points
  auto random_pool =
    Kokkos::Random_XorShift64_Pool<ExecutionSpace>(time(nullptr));
  Kokkos::parallel_for(
    "target_fill",
    Kokkos::RangePolicy<ExecutionSpace>(0, target_points_num),
    KOKKOS_LAMBDA (const int i) {
      auto gen = random_pool.get_state();
      target_points(i) = ArborX::Point {
        gen.frand(0., 1.),
        gen.frand(0., 1.),
        gen.frand(0., 1.),
      };
    });

  // Arrange source points as tree
  auto source_tree =
    ArborX::BVH<MemorySpace>(ExecutionSpace{}, source_points);
}
