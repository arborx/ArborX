/****************************************************************************
 * Copyright (c) 2023 by the ArborX authors                                 *
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

#include <limits>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

struct RBFWendland_0
{
  KOKKOS_INLINE_FUNCTION float operator()(float x)
  {
    x /= _radius;
    return (1. - x) * (1. - x);
  }

  float _radius;
};

struct MVPolynomialBasis_3D
{
  static constexpr std::size_t size = 4;

  KOKKOS_INLINE_FUNCTION Kokkos::Array<float, size>
  operator()(ArborX::Point const &p) const
  {
    return {{1., p[0], p[1], p[2]}};
  }
};

// Function to approximate
KOKKOS_INLINE_FUNCTION float manufactured_solution(ArborX::Point const &p)
{
  return p[2] + p[0];
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);
  constexpr std::size_t num_neighbors = 5;
  constexpr std::size_t cube_side = 4;
  constexpr std::size_t source_points_num = cube_side * cube_side * cube_side;
  constexpr std::size_t target_points_num = 4;

  auto source_points = Kokkos::View<ArborX::Point *, MemorySpace>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "MLS_EX::source_points"),
      source_points_num);
  auto target_points = Kokkos::View<ArborX::Point *, MemorySpace>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "MLS_EX::target_points"),
      target_points_num);
  auto target_points_host = Kokkos::create_mirror_view(target_points);

  // Generate source points (Organized within a [-10, 10]^3 cube)
  Kokkos::parallel_for(
      "MLS_EX::source_points_init",
      Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>>(
          {0, 0, 0}, {cube_side, cube_side, cube_side}),
      KOKKOS_LAMBDA(int const i, int const j, int const k) {
        source_points(i * cube_side * cube_side + j * cube_side + k) =
            ArborX::Point{20.f * (float(i) / (cube_side - 1) - .5f),
                          20.f * (float(j) / (cube_side - 1) - .5f),
                          20.f * (float(k) / (cube_side - 1) - .5f)};
      });

  // Generate target points
  target_points_host(0) = ArborX::Point{0.f, 0.f, 0.f};
  target_points_host(1) = ArborX::Point{5.f, 5.f, 5.f};
  target_points_host(2) = ArborX::Point{-5.f, 5.f, 3.f};
  target_points_host(3) = ArborX::Point{1.f, -3.3f, 7.f};
  Kokkos::deep_copy(target_points, target_points_host);

  // Organize source points as tree
  ArborX::BVH<MemorySpace> source_tree(ExecutionSpace{}, source_points);

  // Create the queries
  // For each target point we query the closest source points
  auto queries = Kokkos::View<ArborX::Nearest<ArborX::Point> *, MemorySpace>(
      "MLS_EX::queries", target_points_num);
  Kokkos::parallel_for(
      "MLS_EX::make_queries",
      Kokkos::RangePolicy<ExecutionSpace>(0, target_points_num),
      KOKKOS_LAMBDA(int const i) {
        queries(i) = ArborX::nearest(target_points(i), num_neighbors);
      });

  // Perform the query
  auto indices = Kokkos::View<int *, MemorySpace>("MLS_EX::indices", 0);
  auto offsets = Kokkos::View<int *, MemorySpace>("MLS_EX::offsets", 0);
  source_tree.query(ExecutionSpace{}, queries, indices, offsets);

  // Now that we have the neighbors, we recompute their position using
  // their target point as the origin.
  // This is used as an optimisation later in the algorithm
  auto tr_source_points = Kokkos::View<ArborX::Point **, MemorySpace>(
      "MLS_EX::tr_source_points", target_points_num, num_neighbors);
  Kokkos::parallel_for(
      "MLS_EX::transform_source_points",
      Kokkos::RangePolicy<ExecutionSpace>(0, target_points_num),
      KOKKOS_LAMBDA(int const i) {
        for (int j = offsets(i); j < offsets(i + 1); j++)
        {
          tr_source_points(i, j - offsets(i)) = ArborX::Point{
              source_points(indices(j))[0] - target_points(i)[0],
              source_points(indices(j))[1] - target_points(i)[1],
              source_points(indices(j))[2] - target_points(i)[2],
          };
        }
      });

  // Compute the radii for the weight (phi) vector
  auto radii =
      Kokkos::View<float *, MemorySpace>("MLS_EX::radii", target_points_num);
  constexpr float epsilon = std::numeric_limits<float>::epsilon();
  Kokkos::parallel_for(
      "MLS_EX::radii_computation",
      Kokkos::RangePolicy<ExecutionSpace>(0, target_points_num),
      KOKKOS_LAMBDA(int const i) {
        float radius = 10. * epsilon;

        for (int j = 0; j < num_neighbors; j++)
        {
          float norm = ArborX::Details::distance(tr_source_points(i, j),
                                                 ArborX::Point{0., 0., 0.});
          radius = (radius < norm) ? norm : radius;
        }

        radii(i) = 1.1 * radius;
      });

  // Compute the weight (phi) vector
  auto phi = Kokkos::View<float **, MemorySpace>(
      "MLS_EX::phi", target_points_num, num_neighbors);
  Kokkos::parallel_for(
      "MLS_EX::phi_computation",
      Kokkos::RangePolicy<ExecutionSpace>(0, phi.extent(0)),
      KOKKOS_LAMBDA(int const i) {
        auto rbf = RBFWendland_0{radii(i)};

        for (int j = 0; j < phi.extent(1); j++)
        {
          float norm = ArborX::Details::distance(tr_source_points(i, j),
                                                 ArborX::Point{0., 0., 0.});
          phi(i, j) = rbf(norm);
        }
      });

  // Compute multivariable Vandermonde (P) matrix
  auto p = Kokkos::View<float ***, MemorySpace>(
      "MLS_EX::vandermonde", target_points_num, num_neighbors,
      MVPolynomialBasis_3D::size);
  Kokkos::parallel_for(
      "MLS_EX::vandermonde_computation",
      Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
          {0, 0}, {target_points_num, num_neighbors}),
      KOKKOS_LAMBDA(int const i, int const j) {
        auto basis = MVPolynomialBasis_3D{}(tr_source_points(i, j));

        for (int k = 0; k < MVPolynomialBasis_3D::size; k++)
        {
          p(i, j, k) = basis[k];
        }
      });

  // Compute moment (A) matrix
  auto a = Kokkos::View<float ***, MemorySpace>("MLS_EX::A", target_points_num,
                                                MVPolynomialBasis_3D::size,
                                                MVPolynomialBasis_3D::size);
  Kokkos::parallel_for(
      "MLS_EX::A_computation",
      Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<3>>(
          {0, 0, 0}, {target_points_num, MVPolynomialBasis_3D::size,
                      MVPolynomialBasis_3D::size}),
      KOKKOS_LAMBDA(int const i, int const j, int const k) {
        float tmp = 0;
        for (int l = 0; l < num_neighbors; l++)
        {
          tmp += p(i, l, j) * p(i, l, k) * phi(i, l);
        }

        a(i, j, k) = tmp;
      });

  // Inverse moment matrix
  // Gaussian inverse method. Both matrix are used and modifications on the
  // first one are applied to the second
  // Kind of works, errors out quite often.
  // A better method should be employed (SVD?)
  auto a_inv = Kokkos::View<float ***, MemorySpace>(
      "MLS_EX::A_inv", target_points_num, MVPolynomialBasis_3D::size,
      MVPolynomialBasis_3D::size);
  Kokkos::parallel_for(
      "MLS_EX::A_inv_computation",
      Kokkos::RangePolicy<ExecutionSpace>(0, target_points_num),
      KOKKOS_LAMBDA(int const i) {
        for (int j = 0; j < MVPolynomialBasis_3D::size; j++)
        {
          for (int k = 0; k < MVPolynomialBasis_3D::size; k++)
          {
            a_inv(i, j, k) = (j == k) * 1.;
          }
        }

        // This needs to be done for every column
        for (int j = 0; j < MVPolynomialBasis_3D::size; j++)
        {

          // We find the line with a non-negative element on column j
          int k = j;
          for (; k < MVPolynomialBasis_3D::size; k++)
          {
            if (a(i, k, j) != 0.0)
              break;
          }

          // We divide the line with said value
          float tmp = a(i, k, j);
          for (int l = 0; l < MVPolynomialBasis_3D::size; l++)
          {
            a(i, k, l) /= tmp;
            a_inv(i, k, l) /= tmp;
          }

          // If line and column are not the same, move the column to the top
          if (k != j)
          {
            for (int l = 0; l < MVPolynomialBasis_3D::size; l++)
            {
              float tmp = a(i, k, l);
              a(i, k, l) = a(i, j, l);
              a(i, j, l) = tmp;

              tmp = a_inv(i, k, l);
              a_inv(i, k, l) = a_inv(i, j, l);
              a_inv(i, j, l) = tmp;
            }
          }

          // Now, set at zero all other elements of the column (Ll <- Ll - a*Lj)
          for (int l = 0; l < MVPolynomialBasis_3D::size; l++)
          {
            if (l == j || a(i, l, j) == 0.0)
              continue;
            float mul = a(i, l, j);

            for (int m = 0; m < MVPolynomialBasis_3D::size; m++)
            {
              a(i, l, m) -= mul * a(i, j, m);
              a_inv(i, l, m) -= mul * a_inv(i, j, m);
            }
            a(i, l, j) = 0.0;
          }

          // Now a_inv should contain the inverse of a
        }
      });

  // Compute the coefficients
  auto coeffs = Kokkos::View<float **, MemorySpace>(
      "MLS_EX::coefficients", target_points_num, num_neighbors);
  Kokkos::parallel_for(
      "MLS_EX::coefficients_computation",
      Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
          {0, 0}, {target_points_num, num_neighbors}),
      KOKKOS_LAMBDA(int const i, int const j) {
        float tmp = 0;

        for (int k = 0; k < MVPolynomialBasis_3D::size; k++)
        {
          tmp += a_inv(i, 0, k) * p(i, j, k) * phi(i, j);
        }

        coeffs(i, j) = tmp;
      });

  // Compute source values
  auto source_values = Kokkos::View<float *, MemorySpace>(
      "MLS_EX::source_values", source_points_num);
  Kokkos::parallel_for(
      "MLS_EX::source_evaluation",
      Kokkos::RangePolicy<ExecutionSpace>(0, source_points_num),
      KOKKOS_LAMBDA(int const i) {
        source_values(i) = manufactured_solution(source_points(i));
      });

  // Compute target values via interpolation
  auto target_values = Kokkos::View<float *, MemorySpace>(
      "MLS_EX::target_values", target_points_num);
  Kokkos::parallel_for(
      "MLS_EX::target_interpolation",
      Kokkos::RangePolicy<ExecutionSpace>(0, target_points_num),
      KOKKOS_LAMBDA(int const i) {
        float tmp = 0;
        for (int j = offsets(i); j < offsets(i + i); j++)
        {
          tmp += coeffs(i, j - offsets(i)) * source_values(indices(j));
        }
        target_values(i) = tmp;
      });

  // Compute target values via evaluation
  auto target_values_exact = Kokkos::View<float *, MemorySpace>(
      "MLS_EX::target_values_exact", target_points_num);
  Kokkos::parallel_for(
      "MLS_EX::target_evaluation",
      Kokkos::RangePolicy<ExecutionSpace>(0, target_points_num),
      KOKKOS_LAMBDA(int const i) {
        target_values_exact(i) = manufactured_solution(target_points(i));
      });

  // Show difference
  auto target_values_host = Kokkos::create_mirror_view(target_values);
  Kokkos::deep_copy(target_values_host, target_values);
  auto target_values_exact_host =
      Kokkos::create_mirror_view(target_values_exact);
  Kokkos::deep_copy(target_values_exact_host, target_values_exact);

  for (int i = 0; i < target_points_num; i++)
  {
    float error =
        Kokkos::abs(target_values_host(i) - target_values_exact_host(i));
    std::cout << "==== Target " << i << '\n'
              << "Interpolation: " << target_values_host(i) << '\n'
              << "Real value   : " << target_values_exact_host(i) << '\n'
              << "Absolute err.: " << error << "\n====\n";
  }
}
