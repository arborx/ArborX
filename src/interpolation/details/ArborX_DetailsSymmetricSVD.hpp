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

#ifndef ARBORX_DETAILS_SYMMETRIC_SVD_HPP
#define ARBORX_DETAILS_SYMMETRIC_SVD_HPP

#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

namespace ArborX::Interpolation::Details
{

template <typename Matrix>
KOKKOS_INLINE_FUNCTION void
ensureIsSquareMatrix([[maybe_unused]] Matrix const &mat)
{
  static_assert(Kokkos::is_view_v<Matrix>, "Matrix must be a view");
  static_assert(Matrix::rank == 2, "Matrix must be 2D");
  KOKKOS_ASSERT(mat.extent(0) == mat.extent(1));
}

template <typename Matrix>
KOKKOS_INLINE_FUNCTION void ensureIsSquareSymmetricMatrix(Matrix const &mat)
{
  ensureIsSquareMatrix(mat);

  [[maybe_unused]] auto is_symmetric = [&]() {
    int const size = mat.extent(0);
    for (int i = 0; i < size; i++)
      for (int j = i + 1; j < size; j++)
        if (mat(i, j) != mat(j, i))
          return false;
    return true;
  };

  KOKKOS_ASSERT(is_symmetric());
}

// Gets the argmax from the upper triangle part of a matrix
template <typename Matrix>
KOKKOS_FUNCTION auto argmaxUpperTriangle(Matrix const &mat)
{
  ensureIsSquareMatrix(mat);
  using Value = typename Matrix::non_const_value_type;

  struct
  {
    Value max = 0;
    int row = 0;
    int col = 0;
  } result;

  int const size = mat.extent(0);
  for (int i = 0; i < size; i++)
    for (int j = i + 1; j < size; j++)
    {
      Value val = Kokkos::abs(mat(i, j));
      if (result.max < val)
      {
        result.max = val;
        result.row = i;
        result.col = j;
      }
    }

  return result;
}

// SVD of a symmetric matrix
// We must find U, E (diagonal and positive) and V such that A = U.E.V^T
// We also suppose, as the input, that A is symmetric, so U = SV where S is
// a sign matrix (only 1 or -1 on the diagonal, 0 elsewhere).
// Thus A = U.ES.U^T.
//
// mat <=> initial ES
// diag <=> final ES
// unit <=> U
template <typename Matrix, typename Diag, typename Unit>
KOKKOS_FUNCTION void symmetricSVDKernel(Matrix &mat, Diag &diag, Unit &unit)
{
  ensureIsSquareSymmetricMatrix(mat);
  static_assert(!std::is_const_v<typename Matrix::value_type>,
                "mat must be writable");

  static_assert(Kokkos::is_view_v<Diag>, "diag must be a view");
  static_assert(Diag::rank == 1, "diag must be 1D");
  static_assert(!std::is_const_v<typename Diag::value_type>,
                "diag must be writable");

  ensureIsSquareMatrix(unit);
  static_assert(!std::is_const_v<typename Unit::value_type>,
                "unit must be writable");
  static_assert(
      std::is_same_v<typename Matrix::value_type, typename Diag::value_type> &&
          std::is_same_v<typename Diag::value_type, typename Unit::value_type>,
      "All input matrices must have the same value type");
  KOKKOS_ASSERT(mat.extent(0) == diag.extent(0) &&
                diag.extent(0) == unit.extent(0));
  using Value = typename Matrix::non_const_value_type;
  int const size = mat.extent(0);

  // We first initialize 'unit' as the identity matrix
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      unit(i, j) = Value(i == j);

  static constexpr Value epsilon = Kokkos::Experimental::epsilon_v<float>;
  while (true)
  {
    // We have a guarantee that p < q
    auto const [max_val, p, q] = argmaxUpperTriangle(mat);
    if (max_val <= epsilon)
      break;

    auto const a = mat(p, p);
    auto const b = mat(p, q);
    auto const c = mat(q, q);

    // Our submatrix is now
    // +-----------+-----------+   +---+---+
    // | mat(p, p) | mat(p, q) |   | a | b |
    // +-----------+-----------+ = +---+---+
    // | mat(q, p) | mat(q, q) |   | b | c |
    // +-----------+-----------+   +---+---+

    // Let's compute x, y and theta such that
    // +---+---+              +---+---+
    // | a | b |              | x | 0 |
    // +---+---+ = R(theta) * +---+---+ * R(theta)^T
    // | b | c |              | 0 | y |
    // +---+---+              +---+---+

    Value cos_theta;
    Value sin_theta;
    Value x;
    Value y;
    if (a == c)
    {
      cos_theta = Kokkos::sqrt(Value(2)) / 2;
      sin_theta = cos_theta;
      x = a + b;
      y = a - b;
    }
    else
    {
      auto const u = (2 * b) / (a - c);
      auto const v = 1 / Kokkos::sqrt(u * u + 1);
      cos_theta = Kokkos::sqrt((1 + v) / 2);
      sin_theta = Kokkos::copysign(Kokkos::sqrt((1 - v) / 2), u);
      x = (a + c + (a - c) / v) / 2;
      y = a + c - x;
    }

    // Now let's compute the following new values for 'unit' and 'mat'
    // mat  <- R'(theta)^T . mat . R'(theta)
    // unit <- unit . R'(theta)

    // R'(theta)^T . mat . R'(theta)
    for (int i = 0; i < p; i++)
    {
      auto const es_ip = mat(i, p);
      auto const es_iq = mat(i, q);
      mat(i, p) = cos_theta * es_ip + sin_theta * es_iq;
      mat(i, q) = -sin_theta * es_ip + cos_theta * es_iq;
    }
    mat(p, p) = x;
    mat(p, q) = 0;
    for (int i = p + 1; i < q; i++)
    {
      auto const es_pi = mat(p, i);
      auto const es_iq = mat(i, q);
      mat(p, i) = cos_theta * es_pi + sin_theta * es_iq;
      mat(i, q) = -sin_theta * es_pi + cos_theta * es_iq;
    }
    mat(q, q) = y;
    for (int i = q + 1; i < size; i++)
    {
      auto const es_pi = mat(p, i);
      auto const es_qi = mat(q, i);
      mat(p, i) = cos_theta * es_pi + sin_theta * es_qi;
      mat(q, i) = -sin_theta * es_pi + cos_theta * es_qi;
    }

    // unit . R'(theta)
    for (int i = 0; i < size; i++)
    {
      auto const u_ip = unit(i, p);
      auto const u_iq = unit(i, q);
      unit(i, p) = cos_theta * u_ip + sin_theta * u_iq;
      unit(i, q) = -sin_theta * u_ip + cos_theta * u_iq;
    }
  }

  for (int i = 0; i < size; i++)
    diag(i) = mat(i, i);
}

// Pseudo-inverse of symmetric matrices using SVD
// We must find U, E (diagonal and positive) and V such that A = U.E.V^T
// We also suppose, as the input, that A is symmetric, so U = SV where S is
// a sign matrix (only 1 or -1 on the diagonal, 0 elsewhere).
// Thus A = U.ES.U^T and A^-1 = U.[ ES^-1 ].U^T
//
// mat <=> initial ES
// diag <=> final ES
// unit <=> U
template <typename Matrix, typename Diag, typename Unit>
KOKKOS_FUNCTION void symmetricPseudoInverseSVDKernel(Matrix &mat, Diag &diag,
                                                     Unit &unit)
{
  symmetricSVDKernel(mat, diag, unit);

  int const size = mat.extent(0);

  using Value = typename Matrix::non_const_value_type;
  constexpr Value epsilon = Kokkos::Experimental::epsilon_v<float>;

  // We compute the max to get a range of the invertible eigenvalues
  auto max_eigen = epsilon;
  for (int i = 0; i < size; i++)
    max_eigen = Kokkos::max(Kokkos::abs(diag(i)), max_eigen);
  auto const threshold = max_eigen * epsilon;

  // We invert the diagonal of 'mat', except if "0" is found
  for (int i = 0; i < size; i++)
    diag(i) = (Kokkos::abs(diag(i)) < threshold) ? 0 : 1 / diag(i);

  // Then we fill out 'mat' as the pseudo inverse
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
    {
      mat(i, j) = 0;
      for (int k = 0; k < size; k++)
        mat(i, j) += diag(k) * unit(i, k) * unit(j, k);
    }
}

} // namespace ArborX::Interpolation::Details

#endif
