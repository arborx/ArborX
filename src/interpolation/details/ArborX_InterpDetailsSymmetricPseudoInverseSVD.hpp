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

#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX::Interpolation::Details
{

// Pseudo-inverse of symmetric matrices using SVD
// We must find U, E (diagonal and positive) and V such that A = U.E.V^T
// We also know that A is symmetric (by construction), so U = SV where S is
// a sign matrix (only 1 or -1 in the diagonal, 0 elsewhere).
// Thus A = U.E.S.U^T and A^-1 = U.[ E^-1.S ].U^T
// Here we have IO = A/A^-1, S = E.S/E^-1.S and U = U
template <typename InOutMatrix, typename SMatrix, typename UMatrix>
KOKKOS_FUNCTION void
symmetricPseudoInverseSVDSerialKernel(InOutMatrix &io, SMatrix &s, UMatrix &u)
{
  using value_t = typename InOutMatrix::non_const_value_type;
  std::size_t const size = io.extent(0);

  // We first initialize u as the identity matrix and copy io to s
  for (std::size_t i = 0; i < size; i++)
    for (std::size_t j = 0; j < size; j++)
    {
      u(i, j) = value_t(i == j);
      s(i, j) = io(i, j);
    }

  // We define the "norm" that will return where to perform the elimination
  auto argmaxUpperTriangle = [=](std::size_t &p, std::size_t &q) {
    value_t max = 0;
    p = q = 0;

    for (std::size_t i = 0; i < size; i++)
      for (std::size_t j = i + 1; j < size; j++)
      {
        value_t val = Kokkos::abs(s(i, j));
        if (max < val)
        {
          max = val;
          p = i;
          q = j;
        }
      }

    return max;
  };

  std::size_t p, q;
  value_t norm = argmaxUpperTriangle(p, q);

  // What value of epsilon is acceptable? Too small and we falsely inverse 0.
  // Too big and we are not accurate enough. The float's epsilon seems to be an
  // accurate epsilon for both calculations (maybe use 2 different epsilons?)
  static constexpr value_t epsilon = Kokkos::Experimental::epsilon_v<float>;
  while (norm > epsilon)
  {
    value_t a = s(p, p);
    value_t b = s(p, q);
    value_t c = s(q, q);

    // Our submatrix is now
    // +---------+---------+   +---+---+
    // | s(p, p) | s(p, q) |   | a | b |
    // +---------+---------+ = +---+---+
    // | s(q, p) | s(q, q) |   | b | c |
    // +---------+---------+   +---+---+

    // Lets compute x, y and theta such that
    // +---+---+              +---+---+
    // | a | b |              | x | 0 |
    // +---+---+ = R(theta) * +---+---+ * R(theta)^T
    // | b | c |              | 0 | y |
    // +---+---+              +---+---+

    // Alternative without trig
    // if (a != c)
    //   u = (2 * b) / (a - c)
    //   v = 1 / sqrt(u * u + 1)
    //   cos = sqrt((1 + v) / 2)
    //   sin = sgn(u) * sqrt((1 - v) / 2)
    //   x = (a + c + (a - c) / v) / 2
    //   y = (a + c - (a - c) / v) / 2 or y = a + c - x
    // else
    //   cos = sqrt(2) / 2
    //   sin = sqrt(2) / 2
    //   x = a + b
    //   y = a - b

    value_t theta, x, y;
    if (a == c) // <-- better to check if |a - c| < epsilon?
    {
      theta = Kokkos::numbers::pi_v<value_t> / 4;
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
    value_t cos_theta = Kokkos::cos(theta);
    value_t sin_theta = Kokkos::sin(theta);

    // Now lets compute the following new values for U amd E.S
    // M <- R'(theta)^T . S . R'(theta)
    // U  <- U . R'(theta)

    // R'(theta)^T . S
    for (std::size_t i = 0; i < size; i++)
    {
      value_t s_pi = s(p, i);
      value_t s_qi = s(q, i);
      s(p, i) = cos_theta * s_pi + sin_theta * s_qi;
      s(q, i) = -sin_theta * s_pi + cos_theta * s_qi;
    }

    // [R'(theta)^T . S] . R'(theta)
    for (std::size_t i = 0; i < size; i++)
    {
      value_t s_ip = s(i, p);
      value_t s_iq = s(i, q);
      s(i, p) = cos_theta * s_ip + sin_theta * s_iq;
      s(i, q) = -sin_theta * s_ip + cos_theta * s_iq;
    }

    // U . R'(theta)
    for (std::size_t i = 0; i < size; i++)
    {
      value_t u_ip = u(i, p);
      value_t u_iq = u(i, q);
      u(i, p) = cos_theta * u_ip + sin_theta * u_iq;
      u(i, q) = -sin_theta * u_ip + cos_theta * u_iq;
    }

    // These should theorically hold but is it ok to force them to their
    // real value?
    s(p, p) = x;
    s(q, q) = y;
    s(p, q) = 0;
    s(q, p) = 0;

    norm = argmaxUpperTriangle(p, q);
  }

  // We compute the max to get a range of the invertible eigen values
  value_t max = epsilon;
  for (std::size_t i = 0; i < size; i++)
    max = Kokkos::max(Kokkos::abs(s(i, i)), max);

  // We inverse the diagonal of S, except if "0" is found
  for (std::size_t i = 0; i < size; i++)
    s(i, i) = (Kokkos::abs(s(i, i)) < max * epsilon) ? 0 : 1 / s(i, i);

  // Then we fill out IO as the pseudo inverse
  for (std::size_t i = 0; i < size; i++)
    for (std::size_t j = 0; j < size; j++)
    {
      value_t tmp = 0;
      for (std::size_t k = 0; k < size; k++)
        tmp += s(k, k) * u(i, k) * u(j, k);
      io(i, j) = tmp;
    }
}

template <typename ExecutionSpace, typename InOutMatrices>
void symmetricPseudoInverseSVD(ExecutionSpace const &space, InOutMatrices &ios)
{
  // InOutMatrices is a single or list of matrices (i.e 2 or 3D view)
  static_assert(Kokkos::is_view_v<InOutMatrices>, "In-out data must be a view");
  static_assert(InOutMatrices::rank == 3 || InOutMatrices::rank == 2,
                "In-out view must be a matrix or a list of matrices");
  static_assert(
      KokkosExt::is_accessible_from<typename InOutMatrices::memory_space,
                                    ExecutionSpace>::value,
      "In-out view must be accessible from the execution space");
  using view_t = Kokkos::View<typename InOutMatrices::non_const_data_type,
                              typename InOutMatrices::memory_space>;

  if constexpr (view_t::rank == 3)
  {
    assert(ios.extent(1) == ios.extent(2)); // Matrices must be square

    view_t s_matrices(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "ArborX::SymmetricPseudoInverseSVD::sigma_list"),
        ios.extent(0), ios.extent(1), ios.extent(2));
    view_t u_matrices(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "ArborX::SymmetricPseudoInverseSVD::givens_list"),
        ios.extent(0), ios.extent(1), ios.extent(2));

    Kokkos::parallel_for(
        "ArborX::SymmetricPseudoInverseSVD::computation_list",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, ios.extent(0)),
        KOKKOS_LAMBDA(int const i) {
          auto io = Kokkos::subview(ios, i, Kokkos::ALL, Kokkos::ALL);
          auto s = Kokkos::subview(s_matrices, i, Kokkos::ALL, Kokkos::ALL);
          auto u = Kokkos::subview(u_matrices, i, Kokkos::ALL, Kokkos::ALL);
          symmetricPseudoInverseSVDSerialKernel(io, s, u);
        });
  }
  else if constexpr (view_t::rank == 2)
  {
    assert(ios.extent(0) == ios.extent(1)); // Matrix must be square

    view_t s_matrices(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "ArborX::SymmetricPseudoInverseSVD::sigma"),
        ios.extent(0), ios.extent(1));
    view_t u_matrices(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "ArborX::SymmetricPseudoInverseSVD::givens"),
        ios.extent(0), ios.extent(1));

    Kokkos::parallel_for(
        "ArborX::SymmetricPseudoInverseSVD::computation",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, 1),
        KOKKOS_LAMBDA(int const) {
          symmetricPseudoInverseSVDSerialKernel(ios, s_matrices, u_matrices);
        });
  }
}

} // namespace ArborX::Interpolation::Details