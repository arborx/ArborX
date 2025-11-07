/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_SYMMETRIC_SVD_HPP
#define ARBORX_SYMMETRIC_SVD_HPP

#include <kokkos_ext/ArborX_KokkosExtAccessibilityTraits.hpp>
#include <misc/ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <Kokkos_Swap.hpp>

namespace ArborX::Details
{

template <typename Matrix>
KOKKOS_INLINE_FUNCTION void
ensureIsSquareMatrix([[maybe_unused]] Matrix const &A)
{
  static_assert(Kokkos::is_view_v<Matrix>, "Matrix must be a view");
  static_assert(Matrix::rank() == 2, "Matrix must be 2D");
  KOKKOS_ASSERT(A.extent(0) == A.extent(1));
}

template <typename Matrix>
KOKKOS_INLINE_FUNCTION void ensureIsSquareSymmetricMatrix(Matrix const &A)
{
  ensureIsSquareMatrix(A);

  [[maybe_unused]] auto is_symmetric = [&]() {
    int const n = A.extent(0);
    for (int i = 0; i < n; i++)
      for (int j = i + 1; j < n; j++)
        if (A(i, j) != A(j, i))
          return false;
    return true;
  };

  KOKKOS_ASSERT(is_symmetric());
}

// Gets the argmax from the upper triangle part of a matrix
template <typename Matrix>
KOKKOS_FUNCTION auto argmaxUpperTriangle(Matrix const &A)
{
  ensureIsSquareMatrix(A);
  using Value = typename Matrix::non_const_value_type;

  struct
  {
    Value max = 0;
    int row = 0;
    int col = 0;
  } result;

  int const n = A.extent(0);
  for (int i = 0; i < n; i++)
    for (int j = i + 1; j < n; j++)
    {
      Value val = Kokkos::abs(A(i, j));
      if (result.max < val)
      {
        result.max = val;
        result.row = i;
        result.col = j;
      }
    }

  return result;
}

// Compute x, y and theta such that
// +---+---+              +---+---+
// | a | b |              | x | 0 |
// +---+---+ = R(theta) * +---+---+ * R(theta)^T
// | b | c |              | 0 | y |
// +---+---+              +---+---+
template <typename Value>
KOKKOS_FUNCTION void
jacobiRotationCoefficients(Value a, Value b, Value c, Value &x, Value &y,
                           Value &cos_theta, Value &sin_theta)
{
  KOKKOS_ASSERT(b != 0);

  // Calculate rotation angle theta ensuring |sin(theta)| <= |cos(theta)|, see
  // https://en.wikipedia.org/wiki/Jacobi_rotation#Numerically_stable_computation
  // or §8.4 in
  // Golub, Gene H.; Van Loan, Charles F. (1996), Matrix Computations (3rd ed.),
  // Baltimore: Johns Hopkins University Press, ISBN 978-0-8018-5414-9
  auto tau = (c - a) / (2 * b);
  auto tan_theta = Kokkos::copysign(
      1 / (Kokkos::abs(tau) + Kokkos::sqrt(1 + tau * tau)), tau);

  // FIXME: use rhypot
  cos_theta = Kokkos::rsqrt(1 + tan_theta * tan_theta);
  sin_theta = -tan_theta * cos_theta;

  x = a - tan_theta * b;
  y = c + tan_theta * b;
}

// SVD of a symmetric matrix
// We must find U, E (diagonal and positive) and V such that A = U.E.V^T
// We also suppose, as the input, that A is symmetric, so U = SV where S is
// a sign matrix (only 1 or -1 on the diagonal, 0 elsewhere).
// Thus A = U.ES.U^T.
//
// A <=> initial ES
// D <=> final ES
// U <=> U
template <typename Matrix, typename Diagonal, typename Unitary>
KOKKOS_FUNCTION void symmetricSVDKernel(Matrix &A, Diagonal &D, Unitary &U)
{
  ensureIsSquareSymmetricMatrix(A);
  static_assert(!std::is_const_v<typename Matrix::value_type>,
                "A must be writable");

  static_assert(Kokkos::is_view_v<Diagonal>, "D must be a view");
  static_assert(Diagonal::rank() == 1, "D must be 1D");
  static_assert(!std::is_const_v<typename Diagonal::value_type>,
                "D must be writable");

  ensureIsSquareMatrix(U);
  static_assert(!std::is_const_v<typename Unitary::value_type>,
                "U must be writable");
  static_assert(std::is_same_v<typename Matrix::value_type,
                               typename Diagonal::value_type> &&
                    std::is_same_v<typename Diagonal::value_type,
                                   typename Unitary::value_type>,
                "All input matrices must have the same value type");
  KOKKOS_ASSERT(A.extent(0) == D.extent(0) && D.extent(0) == U.extent(0));

  using Value = typename Matrix::non_const_value_type;
  int const n = A.extent(0);

  // We first initialize 'U' as the identity matrix
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      U(i, j) = Value(i == j);

  static constexpr auto epsilon = Kokkos::Experimental::epsilon_v<Value>;
  while (true)
  {
    // We have a guarantee that p < q
    auto const [max_val, p, q] = argmaxUpperTriangle(A);
    if (max_val <= epsilon)
      break;

    Value cos_theta;
    Value sin_theta;
    Value x;
    Value y;
    jacobiRotationCoefficients(A(p, p), A(p, q), A(q, q), x, y, cos_theta,
                               sin_theta);

    // A = R(theta) * A * R(theta)^T
    A(p, p) = x;
    A(q, q) = y;
    A(p, q) = 0;
    for (int i = 0; i < n; ++i)
    {
      if (i == p || i == q)
        continue;

      auto ip = i;
      auto pi = p;
      auto iq = i;
      auto qi = q;

      if (i > p)
        Kokkos::kokkos_swap(ip, pi);
      if (i > q)
        Kokkos::kokkos_swap(iq, qi);

      auto const a_ip = A(ip, pi);
      auto const a_iq = A(iq, qi);
      A(ip, pi) = cos_theta * a_ip + sin_theta * a_iq;
      A(iq, qi) = -sin_theta * a_ip + cos_theta * a_iq;
    }

    // U = U * R(theta)^T
    for (int i = 0; i < n; i++)
    {
      auto const u_ip = U(i, p);
      auto const u_iq = U(i, q);
      U(i, p) = cos_theta * u_ip + sin_theta * u_iq;
      U(i, q) = -sin_theta * u_ip + cos_theta * u_iq;
    }
  }

  // Extract eigenvalues from the diagonal of the resulting A
  for (int i = 0; i < n; i++)
    D(i) = A(i, i);
}

// Pseudo-inverse of symmetric matrices using SVD
// We must find U, E (diagonal and positive) and V such that A = U.E.V^T
// We also suppose, as the input, that A is symmetric, so U = SV where S is
// a sign matrix (only 1 or -1 on the diagonal, 0 elsewhere).
// Thus A = U.ES.U^T and A^-1 = U.[ ES^-1 ].U^T
//
// A <=> initial ES
// D <=> final ES
// U <=> U
template <typename Matrix, typename Diagonal, typename Unitary>
KOKKOS_FUNCTION void symmetricPseudoInverseSVDKernel(Matrix &A, Diagonal &D,
                                                     Unitary &U)
{
  symmetricSVDKernel(A, D, U);

  int const n = A.extent(0);

  using Value = typename Matrix::non_const_value_type;

  // Compute the max to get a range of the invertible eigenvalues
  Value max_eigen = 0;
  for (int i = 0; i < n; i++)
    max_eigen = Kokkos::max(Kokkos::abs(D(i)), max_eigen);

  // The standard tolerance for forming pseudo-inverse is to only invert
  // singular values that are max(m,n) * \epsilon * ||A||_2.
  // ||A||_2 is equal to the max singular value (max abs diagonal).
  constexpr auto epsilon = Kokkos::Experimental::epsilon_v<Value>;
  auto const tolerance = n * max_eigen * epsilon;
  for (int i = 0; i < n; i++)
    D(i) = (Kokkos::abs(D(i)) <= tolerance) ? 0 : 1 / D(i);

  // Then we fill out 'A' as the pseudo inverse
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
    {
      A(i, j) = 0;
      for (int k = 0; k < n; k++)
        A(i, j) += U(i, k) * D(k) * U(j, k);
    }
}

} // namespace ArborX::Details

#endif
