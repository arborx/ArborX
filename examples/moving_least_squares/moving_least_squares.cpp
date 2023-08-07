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

#include <cmath>
#include <limits>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

struct RBFWendland_0
{
  KOKKOS_INLINE_FUNCTION float operator()(float x)
  {
    x /= _radius;
    return (1.f - x) * (1.f - x);
  }

  float _radius;
};

struct MVPolynomialBasis_3D
{
  static constexpr std::size_t size = 10;

  KOKKOS_INLINE_FUNCTION Kokkos::Array<float, size>
  operator()(ArborX::Point const &p) const
  {
    return {{1.f, p[0], p[1], p[2], p[0] * p[0], p[0] * p[1], p[0] * p[2],
             p[1] * p[1], p[1] * p[2], p[2] * p[2]}};
  }
};

struct TargetPoints
{
  Kokkos::View<ArborX::Point *, MemorySpace> target_points;
  std::size_t num_neighbors;
};

template <>
struct ArborX::AccessTraits<TargetPoints, ArborX::PredicatesTag>
{
  static KOKKOS_FUNCTION std::size_t size(TargetPoints const &tp)
  {
    return tp.target_points.extent(0);
  }

  static KOKKOS_FUNCTION auto get(TargetPoints const &tp, std::size_t i)
  {
    return ArborX::nearest(tp.target_points(i), tp.num_neighbors);
  }

  using memory_space = MemorySpace;
};

// Function to approximate
KOKKOS_INLINE_FUNCTION float manufactured_solution(ArborX::Point const &p)
{
  return Kokkos::sin(p[0]) * p[2] + p[1];
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  constexpr float epsilon = std::numeric_limits<float>::epsilon();
  constexpr std::size_t num_neighbors = MVPolynomialBasis_3D::size;
  constexpr std::size_t cube_side = 10;
  constexpr std::size_t source_points_num = cube_side * cube_side * cube_side;
  constexpr std::size_t target_points_num = 4;

  ExecutionSpace space{};

  Kokkos::View<ArborX::Point *, MemorySpace> source_points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::source_points"),
      source_points_num);
  Kokkos::View<ArborX::Point *, MemorySpace> target_points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::target_points"),
      target_points_num);
  auto target_points_host = Kokkos::create_mirror_view(target_points);

  // Generate source points (Organized within a [-10, 10]^3 cube)
  Kokkos::parallel_for(
      "Example::source_points_init",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(space, {0, 0, 0},
                                             {cube_side, cube_side, cube_side}),
      KOKKOS_LAMBDA(int const i, int const j, int const k) {
        source_points(i * cube_side * cube_side + j * cube_side + k) =
            ArborX::Point{20.f * (float(i) / (cube_side - 1) - .5f),
                          20.f * (float(j) / (cube_side - 1) - .5f),
                          20.f * (float(k) / (cube_side - 1) - .5f)};
      });

  // Generate target points
  target_points_host(0) = ArborX::Point{1.f, 0.f, 1.f};
  target_points_host(1) = ArborX::Point{5.f, 5.f, 5.f};
  target_points_host(2) = ArborX::Point{-5.f, 5.f, 3.f};
  target_points_host(3) = ArborX::Point{1.f, -3.3f, 7.f};
  Kokkos::deep_copy(space, target_points, target_points_host);

  // Organize source points as tree
  ArborX::BVH<MemorySpace> source_tree(space, source_points);

  // Perform the query
  Kokkos::View<int *, MemorySpace> indices("Example::indices", 0);
  Kokkos::View<int *, MemorySpace> offsets("Example::offsets", 0);
  source_tree.query(space, TargetPoints{target_points, num_neighbors}, indices,
                    offsets);

  // Now that we have the neighbors, we recompute their position using
  // their target point as the origin.
  // This is used as an optimisation later in the algorithm
  Kokkos::View<ArborX::Point **, MemorySpace> tr_source_points(
      "Example::tr_source_points", target_points_num, num_neighbors);
  Kokkos::parallel_for(
      "Example::transform_source_points",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, target_points_num),
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
  Kokkos::View<float *, MemorySpace> radii("Example::radii", target_points_num);
  Kokkos::parallel_for(
      "Example::radii_computation",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, target_points_num),
      KOKKOS_LAMBDA(int const i) {
        float radius = 10.f * epsilon;

        for (int j = 0; j < num_neighbors; j++)
        {
          float norm = ArborX::Details::distance(tr_source_points(i, j),
                                                 ArborX::Point{0.f, 0.f, 0.f});
          radius = (radius < norm) ? norm : radius;
        }

        radii(i) = 1.1f * radius;
      });

  // Compute the weight (phi) vector
  Kokkos::View<float **, MemorySpace> phi("Example::phi", target_points_num,
                                          num_neighbors);
  Kokkos::parallel_for(
      "Example::phi_computation",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, target_points_num),
      KOKKOS_LAMBDA(int const i) {
        RBFWendland_0 rbf{radii(i)};

        for (int j = 0; j < num_neighbors; j++)
        {
          float norm = ArborX::Details::distance(tr_source_points(i, j),
                                                 ArborX::Point{0.f, 0.f, 0.f});
          phi(i, j) = rbf(norm);
        }
      });

  // Compute multivariable Vandermonde (P) matrix
  Kokkos::View<float ***, MemorySpace> p("Example::vandermonde",
                                         target_points_num, num_neighbors,
                                         MVPolynomialBasis_3D::size);
  Kokkos::parallel_for(
      "Example::vandermonde_computation",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
          space, {0, 0}, {target_points_num, num_neighbors}),
      KOKKOS_LAMBDA(int const i, int const j) {
        auto basis = MVPolynomialBasis_3D{}(tr_source_points(i, j));

        for (int k = 0; k < MVPolynomialBasis_3D::size; k++)
        {
          p(i, j, k) = basis[k];
        }
      });

  // Compute moment (A) matrix
  Kokkos::View<float ***, MemorySpace> a("Example::A", target_points_num,
                                         MVPolynomialBasis_3D::size,
                                         MVPolynomialBasis_3D::size);
  Kokkos::parallel_for(
      "Example::A_computation",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(space, {0, 0, 0},
                                             {target_points_num,
                                              MVPolynomialBasis_3D::size,
                                              MVPolynomialBasis_3D::size}),
      KOKKOS_LAMBDA(int const i, int const j, int const k) {
        float tmp = 0;
        for (int l = 0; l < num_neighbors; l++)
        {
          tmp += p(i, l, j) * p(i, l, k) * phi(i, l);
        }

        a(i, j, k) = tmp;
      });

  // Pseudo-inverse moment matrix using SVD
  // We must find U, E (diagonal and positive) and V such that A = U.E.V^T
  // We also know that A is symmetric (by construction), so U = SV where S is
  // a sign matrix (only 1 or -1 in the diagonal, 0 elsewhere).
  // Thus A = U.E.S.U^T
  static constexpr float pi_4 = M_PI_4;
  Kokkos::View<float ***, MemorySpace> a_inv(
      "Example::A_inv", target_points_num, MVPolynomialBasis_3D::size,
      MVPolynomialBasis_3D::size);
  Kokkos::View<float ***, MemorySpace> svd_u(
      "Example::SVD::U", target_points_num, MVPolynomialBasis_3D::size,
      MVPolynomialBasis_3D::size);
  Kokkos::View<float ***, MemorySpace> svd_es(
      "Example::SVD::E.S", target_points_num, MVPolynomialBasis_3D::size,
      MVPolynomialBasis_3D::size);
  Kokkos::deep_copy(space, svd_es, a);
  Kokkos::parallel_for(
      "Example::A_inv_computation",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, target_points_num),
      KOKKOS_LAMBDA(int const i) {
        for (int j = 0; j < MVPolynomialBasis_3D::size; j++)
        {
          for (int k = 0; k < MVPolynomialBasis_3D::size; k++)
          {
            svd_u(i, j, k) = (j == k) * 1.f;
          }
        }

        // This finds the biggest off-diagonal value of E.S as well as its
        // coordinates. Being symmetric, we can always check on the upper
        // triangle (and always have q > p)
        auto argmax = [=](int &p, int &q) {
          float max = 0.f;
          p = -1;
          q = -1;
          for (int j = 0; j < MVPolynomialBasis_3D::size; j++)
          {
            for (int k = j + 1; k < MVPolynomialBasis_3D::size; k++)
            {
              float val = Kokkos::abs(svd_es(i, j, k));
              if (max < val)
              {
                max = val;
                p = j;
                q = k;
              }
            }
          }

          return max;
        };

        // Iterative approach, we will "deconstruct" E.S until only the diagonal
        // is relevent inside the matrix
        // It is possible to prove that, at each step, the "norm" of the matrix
        // is strictly less that of the previous
        int p, q;
        float norm = argmax(p, q);
        while (norm > epsilon)
        {
          // Our submatrix is now
          // +----------+----------+   +---+---+
          // | es(p, p) | es(p, q) |   | a | b |
          // +----------+----------+ = +---+---+
          // | es(q, p) | es(q, q) |   | b | c |
          // +----------+----------+   +---+---+
          float a = svd_es(i, p, p);
          float b = svd_es(i, p, q);
          float c = svd_es(i, q, q);

          float theta, u, v;
          if (a == c)
          {
            theta = pi_4;
            u = a + b;
            v = a - b;
          }
          else
          {
            theta = .5f * Kokkos::atanf((2.f * b) / (a - c));
            float cos2 = Kokkos::cosf(2.f * theta);
            u = .5f * (a + c + (a - c) / cos2);
            v = .5f * (a + c - (a - c) / cos2);
          }
          float cos = Kokkos::cosf(theta);
          float sin = Kokkos::sinf(theta);

          // We must now apply the rotation matrix to the left
          // and right of E.S and on the right of U

          // Left of E.S (mult by R(theta)^T)
          for (int j = 0; j < MVPolynomialBasis_3D::size; j++)
          {
            float es_ipj = svd_es(i, p, j);
            float es_iqj = svd_es(i, q, j);
            svd_es(i, p, j) = cos * es_ipj + sin * es_iqj;
            svd_es(i, q, j) = -sin * es_ipj + cos * es_iqj;
          }

          // Right of E.S (mult by R(theta))
          for (int j = 0; j < MVPolynomialBasis_3D::size; j++)
          {
            float es_ijp = svd_es(i, j, p);
            float es_ijq = svd_es(i, j, q);
            svd_es(i, j, p) = cos * es_ijp + sin * es_ijq;
            svd_es(i, j, q) = -sin * es_ijp + cos * es_ijq;
          }

          // Right of U (mult by R(theta))
          for (int j = 0; j < MVPolynomialBasis_3D::size; j++)
          {
            float u_ijp = svd_u(i, j, p);
            float u_ijq = svd_u(i, j, q);
            svd_u(i, j, p) = cos * u_ijp + sin * u_ijq;
            svd_u(i, j, q) = -sin * u_ijp + cos * u_ijq;
          }

          // These should theorically hold but is it ok to force them to their
          // real value?
          svd_es(i, p, p) = u;
          svd_es(i, q, q) = v;
          svd_es(i, p, q) = 0.f;
          svd_es(i, q, p) = 0.f;

          norm = argmax(p, q);
        }

        // We should now have a correct U and E.S
        // We'll compute the pseudo inverse of A by taking the
        // pseudo inverse of E.S which is simply inverting the diagonal of
        // E.S. We have pseudoA = U^T.pseudoES.U
        for (int j = 0; j < MVPolynomialBasis_3D::size; j++)
        {
          for (int k = 0; k < MVPolynomialBasis_3D::size; k++)
          {
            float value = 0.f;
            for (int l = 0; l < MVPolynomialBasis_3D::size; l++)
            {
              if (Kokkos::abs(svd_es(i, l, l)) >= epsilon)
              {
                value += svd_u(i, j, l) * svd_u(i, k, l) / svd_es(i, l, l);
              }
            }

            a_inv(i, j, k) = value;
          }
        }
      });

  // Compute the coefficients
  Kokkos::View<float **, MemorySpace> coeffs("Example::coefficients",
                                             target_points_num, num_neighbors);
  Kokkos::parallel_for(
      "Example::coefficients_computation",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
          space, {0, 0}, {target_points_num, num_neighbors}),
      KOKKOS_LAMBDA(int const i, int const j) {
        float tmp = 0.f;

        for (int k = 0; k < MVPolynomialBasis_3D::size; k++)
        {
          tmp += a_inv(i, 0, k) * p(i, j, k) * phi(i, j);
        }

        coeffs(i, j) = tmp;
      });

  // Compute source values
  Kokkos::View<float *, MemorySpace> source_values("Example::source_values",
                                                   source_points_num);
  Kokkos::parallel_for(
      "Example::source_evaluation",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, source_points_num),
      KOKKOS_LAMBDA(int const i) {
        source_values(i) = manufactured_solution(source_points(i));
      });

  // Compute target values via interpolation
  Kokkos::View<float *, MemorySpace> target_values("Example::target_values",
                                                   target_points_num);
  Kokkos::parallel_for(
      "Example::target_interpolation",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, target_points_num),
      KOKKOS_LAMBDA(int const i) {
        float tmp = 0;
        for (int j = offsets(i); j < offsets(i + 1); j++)
        {
          tmp += coeffs(i, j - offsets(i)) * source_values(indices(j));
        }
        target_values(i) = tmp;
      });

  // Compute target values via evaluation
  Kokkos::View<float *, MemorySpace> target_values_exact(
      "Example::target_values_exact", target_points_num);
  Kokkos::parallel_for(
      "Example::target_evaluation",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, target_points_num),
      KOKKOS_LAMBDA(int const i) {
        target_values_exact(i) = manufactured_solution(target_points(i));
      });

  // Show difference
  auto target_values_host = Kokkos::create_mirror_view(target_values);
  Kokkos::deep_copy(space, target_values_host, target_values);
  auto target_values_exact_host =
      Kokkos::create_mirror_view(target_values_exact);
  Kokkos::deep_copy(space, target_values_exact_host, target_values_exact);

  float error = 0.f;
  for (int i = 0; i < target_points_num; i++)
  {
    error = Kokkos::max(
        Kokkos::abs(target_values_host(i) - target_values_exact_host(i)),
        error);
    std::cout << "==== Target " << i << '\n'
              << "Interpolation: " << target_values_host(i) << '\n'
              << "Real value   : " << target_values_exact_host(i) << '\n';
  }
  std::cout << "====\nMaximum error: " << error << std::endl;
}
