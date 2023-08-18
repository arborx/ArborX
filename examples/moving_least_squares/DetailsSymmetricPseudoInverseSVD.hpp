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

#pragma once

#include <Kokkos_Core.hpp>

#include <cmath>
#include <limits>

namespace Details
{

// This finds the biggest off-diagonal value of E.S as well as its
// coordinates. Being symmetric, we can always check on the upper
// triangle (and always have q > p)
template <typename Matrices>
KOKKOS_FUNCTION typename Matrices::non_const_value_type
spisvdArgmaxOffDiagonal(Matrices const &es, int const i, int &p, int &q)
{
  using value_t = typename Matrices::non_const_value_type;

  std::size_t const size = es.extent(1);
  value_t max = 0;
  p = q = 0;

  for (int j = 0; j < size; j++)
  {
    for (int k = j + 1; k < size; k++)
    {
      value_t val = Kokkos::abs(es(i, j, k));
      if (max < val)
      {
        max = val;
        p = j;
        q = k;
      }
    }
  }

  return max;
}

// Pseudo-inverse of symmetric matrices using SVD
// We must find U, E (diagonal and positive) and V such that A = U.E.V^T
// We also know that A is symmetric (by construction), so U = SV where S is
// a sign matrix (only 1 or -1 in the diagonal, 0 elsewhere).
// Thus A = U.E.S.U^T and A^-1 = U.[ E^-1.S ].U^T
template <typename ExecutionSpace, typename Matrices>
Kokkos::View<typename Matrices::non_const_value_type ***,
             typename Matrices::memory_space>
symmetricPseudoInverseSVD(ExecutionSpace const &space, Matrices const &mats)
{
  using value_t = typename Matrices::non_const_value_type;
  using memory_space = typename Matrices::memory_space;

  std::size_t const num_matrices = mats.extent(0);
  std::size_t const size = mats.extent(1);
  constexpr value_t epsilon = std::numeric_limits<value_t>::epsilon();
  constexpr value_t pi_4 = value_t(M_PI_4);

  // ==> Initialisation
  // E.S is the input matrix
  // U is the identity
  Kokkos::View<value_t ***, memory_space> es(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::SPISVD::ES"),
      mats.layout());
  Kokkos::View<value_t ***, memory_space> u(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::SPISVD::U"),
      mats.layout());
  Kokkos::parallel_for(
      "Example::SPISVD::ES_U_init",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(space, {0, 0, 0},
                                             {num_matrices, size, size}),
      KOKKOS_LAMBDA(int const i, int const j, int const k) {
        es(i, j, k) = value_t(mats(i, j, k));
        u(i, j, k) = value_t((j == k));
      });

  // ==> Loop
  // Iterative approach, we will "deconstruct" E.S until only the diagonal
  // is relevent inside the matrix
  // It is possible to prove that, at each step, the "norm" of the matrix
  // is strictly less that of the previous
  // For all the loops, the following equality holds: A = U.E.S.U^T
  Kokkos::parallel_for(
      "Example::SPISVD::compute_ES_U",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_matrices),
      KOKKOS_LAMBDA(int const i) {
        int p, q;
        value_t norm = spisvdArgmaxOffDiagonal(es, i, p, q);
        while (norm > epsilon)
        {
          value_t a = es(i, p, p);
          value_t b = es(i, p, q);
          value_t c = es(i, q, q);

          // Our submatrix is now
          // +----------+----------+   +---+---+
          // | es(p, p) | es(p, q) |   | a | b |
          // +----------+----------+ = +---+---+
          // | es(q, p) | es(q, q) |   | b | c |
          // +----------+----------+   +---+---+

          // Lets compute x, y and theta such that
          // +---+---+              +---+---+
          // | a | b |              | x | 0 |
          // +---+---+ = R(theta) * +---+---+ * R(theta)^T
          // | b | c |              | 0 | y |
          // +---+---+              +---+---+

          value_t theta, x, y;
          if (a == c) // <-- better to check if |a - c| < epsilon?
          {
            theta = pi_4;
            x = a + b;
            y = a - b;
          }
          else
          {
            theta = Kokkos::atan((2 * b) / (a - c)) / 2;
            value_t a_c_cos2 = (a - c) / Kokkos::cos(2 * theta);
            x = (a + c + a_c_cos2) / 2;
            y = (a + c - a_c_cos2) / 2;
          }
          value_t cos = Kokkos::cos(theta);
          value_t sin = Kokkos::sin(theta);

          // Now lets compute the following new values for U amd E.S
          // E.S <- R'(theta)^T . E.S . R'(theta)
          // U  <- U . R'(theta)

          // R'(theta)^T . E.S
          for (int j = 0; j < size; j++)
          {
            value_t es_ipj = es(i, p, j);
            value_t es_iqj = es(i, q, j);
            es(i, p, j) = cos * es_ipj + sin * es_iqj;
            es(i, q, j) = -sin * es_ipj + cos * es_iqj;
          }

          // [R'(theta)^T . E.S] . R'(theta)
          for (int j = 0; j < size; j++)
          {
            value_t es_ijp = es(i, j, p);
            value_t es_ijq = es(i, j, q);
            es(i, j, p) = cos * es_ijp + sin * es_ijq;
            es(i, j, q) = -sin * es_ijp + cos * es_ijq;
          }

          // U . R'(theta)
          for (int j = 0; j < size; j++)
          {
            value_t u_ijp = u(i, j, p);
            value_t u_ijq = u(i, j, q);
            u(i, j, p) = cos * u_ijp + sin * u_ijq;
            u(i, j, q) = -sin * u_ijp + cos * u_ijq;
          }

          // These should theorically hold but is it ok to force them to their
          // real value?
          es(i, p, p) = x;
          es(i, q, q) = y;
          es(i, p, q) = 0;
          es(i, q, p) = 0;

          norm = spisvdArgmaxOffDiagonal(es, i, p, q);
        }
      });

  // ==> Output
  // U and E.S are computed, we can now build the inverse
  // U.[ E^-1.S ].U^T
  Kokkos::View<value_t ***, memory_space> inv(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::SPISVD::inv"),
      mats.layout());
  Kokkos::parallel_for(
      "Example::SPISVD::inv_fill",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(space, {0, 0, 0},
                                             {num_matrices, size, size}),
      KOKKOS_LAMBDA(int const i, int const j, int const k) {
        value_t value = 0;
        for (int l = 0; l < size; l++)
        {
          value_t v = es(i, l, l);
          if (Kokkos::abs(v) > epsilon)
          {
            value += u(i, j, l) * u(i, k, l) / v;
          }
        }

        inv(i, j, k) = value;
      });

  return inv;
}

} // namespace Details