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
// with MLS resolution from
// (http://dx.doi.org/10.1016/j.jcp.2015.11.055)

#include <ArborX.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <limits>
#include <iomanip>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

std::ostream &operator<<(std::ostream &os, ArborX::Point const &p)
{
  os << '(' << p[0] << ',' << p[1] << ',' << p[2] << ')';
  return os;
}

struct RBFWendland_0
{
  KOKKOS_INLINE_FUNCTION double operator()(double x)
  {
    x /= _radius;
    return (1. - x) * (1. - x);
  }

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

  constexpr float cube_half_side = 10.; // [-10, 10]^3 cube
  constexpr float cube_side = 2 * cube_half_side;
  constexpr std::size_t source_points_side = 100; // [-10, 10]^3 grid
  constexpr std::size_t target_points_num = 10'000; // random [-10, 10]^3
  constexpr std::size_t num_neighbors = MVPolynomialBasis_Quad_3D::size; // ???

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
  
  // Create the queries
  // For each target point we query the closest source points
  auto queries = Kokkos::View<ArborX::Nearest<ArborX::Point>*, MemorySpace>(
    "queries", target_points_num);
  Kokkos::parallel_for(
    "make_queries",
    Kokkos::RangePolicy<ExecutionSpace>(0, target_points_num),
    KOKKOS_LAMBDA (const int i) {
      queries(i) = ArborX::nearest(target_points(i), num_neighbors);
  });

  // Perform the query
  auto indices = Kokkos::View<int *, MemorySpace>("indices", 0);
  auto offsets = Kokkos::View<int *, MemorySpace>("offsets", 0);
  source_tree.query(ExecutionSpace{}, queries, indices, offsets);

  // Now that we have the neighbors, we recompute their position using
  // their target point as the origin.
  // This is used as an optimisation later in the algorithm
  auto tr_source_points = Kokkos::View<ArborX::Point**, MemorySpace>(
    "tr_source_points", target_points_num, num_neighbors);
  Kokkos::parallel_for(
    "transform_source_points",
    Kokkos::RangePolicy<ExecutionSpace>(0, target_points_num),
    KOKKOS_LAMBDA (const int i) {
      for (int j = offsets(i); j < offsets(i+1); j++) {
        tr_source_points(i, j - offsets(i)) = ArborX::Point {
          source_points(j)[0] - target_points(i)[0],
          source_points(j)[1] - target_points(i)[1],
          source_points(j)[2] - target_points(i)[2],
        };
      }
  });

  // Compute the radii for the weight (phi) vector
  auto radii = Kokkos::View<double*, MemorySpace>(
    "radii", target_points_num);
  constexpr double epsilon = std::numeric_limits<double>::epsilon();
  Kokkos::parallel_for(
    "radii_computation",
    Kokkos::RangePolicy<ExecutionSpace>(0, target_points_num),
    KOKKOS_LAMBDA (const int i) {
      double radius = 10. * epsilon;

      for (int j = 0; j < num_neighbors; j++) {
        double norm = ArborX::Details::distance(
          tr_source_points(i, j),
          ArborX::Point{0., 0., 0.});
        radius = (radius < norm) ? norm : radius;
      }

      radii(i) = 1.1 * radius;
  });

  // Compute the weight (phi) vector
  auto phi = Kokkos::View<double**, MemorySpace>(
    "phi", target_points_num, num_neighbors);
  Kokkos::parallel_for(
    "phi_computation",
    Kokkos::RangePolicy<ExecutionSpace>(0, phi.extent(0)),
    KOKKOS_LAMBDA (const int i) {
      auto rbf = RBFWendland_0 { radii(i) };

      for (int j = 0; j < phi.extent(1); j++) {
        double norm = ArborX::Details::distance(
          tr_source_points(i, j),
          ArborX::Point{0., 0., 0.});
        phi(i, j) = rbf(norm);
      }
  });

  // Compute multivariable Vandermonde (P) matrix
  auto p = Kokkos::View<double***, MemorySpace>(
    "vandermonde",
      target_points_num,
      num_neighbors,
      MVPolynomialBasis_Quad_3D::size
  );
  Kokkos::parallel_for(
    "vandermonde_computation",
    Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
      {0, 0}, {target_points_num, num_neighbors}),
    KOKKOS_LAMBDA (const int i, const int j) {
      auto basis = MVPolynomialBasis_Quad_3D{}(tr_source_points(i, j));

      for (int k = 0; k < MVPolynomialBasis_Quad_3D::size; k++) {
        p(i, j, k) = basis[k];
      }
  });

  // Compute moment (A) matrix
  auto a = Kokkos::View<double***, MemorySpace>(
    "A",
      target_points_num,
      MVPolynomialBasis_Quad_3D::size,
      MVPolynomialBasis_Quad_3D::size
  );
  Kokkos::parallel_for(
    "A_computation",
    Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>>(
      {0, 0, 0},
      {
        target_points_num,
        MVPolynomialBasis_Quad_3D::size,
        MVPolynomialBasis_Quad_3D::size
    }),
    KOKKOS_LAMBDA (const int i, const int j, const int k) {
      double tmp = 0;
      for (int l = 0; l < num_neighbors; l++) {
        tmp += p(i, l, j) * p(i, l, k) * phi(i, l);
      }

      a(i, j, k) = tmp;
  });

  // Inverse moment matrix
  // Gaussian inverse method. Both matrix are used and modifications on the
  // first one are applied to the second
  // Kind of works, errors out quite often.
  // A better method should be employed (SVD?)
  auto a_inv = Kokkos::View<double***, MemorySpace>(
    "A_inv",
      target_points_num,
      MVPolynomialBasis_Quad_3D::size,
      MVPolynomialBasis_Quad_3D::size
  );
  Kokkos::parallel_for(
    "A_inv_computation",
    Kokkos::RangePolicy<ExecutionSpace>(0, target_points_num),
    KOKKOS_LAMBDA (const int i) {
      for (int j = 0; j < MVPolynomialBasis_Quad_3D::size; j++) {
        for (int k = 0; k < MVPolynomialBasis_Quad_3D::size; k++) {
          a_inv(i, j, k) = (j == k) * 1.;
        }
      }

      // This needs to be done for every column
      for (int j = 0; j < MVPolynomialBasis_Quad_3D::size; j++) {

        // We find the line with a non-negative element on column j
        int k = j;
        for (; k < MVPolynomialBasis_Quad_3D::size; k++) {
          if (a(i, k, j) != 0.0) break;
        }

        // We divide the line with said value
        double tmp = a(i, k, j);
        for (int l = 0; l < MVPolynomialBasis_Quad_3D::size; l++) {
          a(i, k, l) /= tmp;
          a_inv(i, k, l) /= tmp;
        }

        // If line and column are not the same, move the column to the top
        if (k != j) {
          for (int l = 0; l < MVPolynomialBasis_Quad_3D::size; l++) {
            double tmp = a(i, k, l);
            a(i, k, l) = a(i, j, l);
            a(i, j, l) = tmp;

            tmp = a_inv(i, k, l);
            a_inv(i, k, l) = a_inv(i, j, l);
            a_inv(i, j, l) = tmp;
          }
        }

        // Now, set at zero all other elements of the column (Ll <- Ll - a*Lj)
        for (int l = 0; l < MVPolynomialBasis_Quad_3D::size; l++) {
          if (l == j || a(i, l, j) == 0.0) continue;
          double mul = a(i, l, j);

          for (int m = 0; m < MVPolynomialBasis_Quad_3D::size; m++) {
            a(i, l, m) -= mul * a(i, j, m);
            a_inv(i, l, m) -= mul * a_inv(i, j, m);
          }
          a(i, l, j) = 0.0;
        }

        // Now a_inv should contain the inverse of a
      }
  });

  // Compute the coefficients
  auto coeffs = Kokkos::View<double**, MemorySpace>(
    "coefficients", target_points_num, MVPolynomialBasis_Quad_3D::size);
  Kokkos::parallel_for(
    "coefficients_computation",
    Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
      {0, 0}, {target_points_num, MVPolynomialBasis_Quad_3D::size}),
    KOKKOS_LAMBDA (const int i, const int j) {
      double tmp = 0;

      for (int k = 0; k < MVPolynomialBasis_Quad_3D::size; k++) {
        tmp += a_inv(i, 0, j) * p(i, k, j) * phi(i, k);
      }

      coeffs(i, j) = tmp;
  });
}
