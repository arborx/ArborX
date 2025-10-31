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

namespace ArborX::Details
{

template <typename Matrix>
KOKKOS_INLINE_FUNCTION void
ensureIsSquareMatrix([[maybe_unused]] Matrix const &mat)
{
  static_assert(Kokkos::is_view_v<Matrix>, "Matrix must be a view");
  static_assert(Matrix::rank() == 2, "Matrix must be 2D");
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

// Compute x, y and theta such that
// +---+---+              +---+---+
// | a | b |              | x | 0 |
// +---+---+ = R(theta) * +---+---+ * R(theta)^T
// | b | c |              | 0 | y |
// +---+---+              +---+---+
template <typename Value>
KOKKOS_FUNCTION void svd2x2(Value a, Value b, Value c, Value &x, Value &y,
                            Value &cos_theta, Value &sin_theta)
{
  if (b == 0)
  {
    cos_theta = 1;
    sin_theta = 0;
    x = a;
    y = c;
    return;
  }

  auto const trace = a + c;
  auto const diff = a - c;
  auto const root = Kokkos::sqrt(diff * diff + 4 * b * b);

  if (trace > 0)
  {
    x = (trace + root) / 2;
    auto t = 1 / x;
    y = (a * t) * c - (b * t) * b;
  }
  else if (trace < 0)
  {
    y = (trace - root) / 2;
    auto t = 1 / y;
    x = (a * t) * c - (b * t) * b;
  }
  else
  {
    x = root / 2;
    y = -root / 2;
  }

  auto const alpha = diff + (diff > 0 ? root : -root);
  auto const beta = 2 * b;

  if (Kokkos::abs(alpha) > Kokkos::abs(beta))
  {
    auto const t = -beta / alpha;
    sin_theta = 1 / Kokkos::sqrt(1 + t * t);
    cos_theta = t * sin_theta;
  }
  else
  {
    auto const t = -alpha / beta;
    cos_theta = 1 / Kokkos::sqrt(1 + t * t);
    sin_theta = t * cos_theta;
  }

  if (diff > 0)
  {
    auto const t = cos_theta;
    cos_theta = -sin_theta;
    sin_theta = t;
  }
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
  static_assert(Diag::rank() == 1, "diag must be 1D");
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

  static constexpr Value epsilon = Kokkos::Experimental::epsilon_v<Value>;
  while (true)
  {
    // We have a guarantee that p < q
    auto const [max_val, p, q] = argmaxUpperTriangle(mat);
    if (max_val <= epsilon)
      break;

    Value cos_theta;
    Value sin_theta;
    Value x;
    Value y;
    svd2x2(mat(p, p), mat(p, q), mat(q, q), x, y, cos_theta, sin_theta);

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

  // Compute the max to get a range of the invertible eigenvalues
  Value max_eigen = 0;
  for (int i = 0; i < size; i++)
    max_eigen = Kokkos::max(Kokkos::abs(diag(i)), max_eigen);

  constexpr auto epsilon = Kokkos::Experimental::epsilon_v<Value>;
  Value zero_scaling = epsilon;
  if constexpr (std::is_same_v<Value, double>)
    zero_scaling = 1e-10;

  // Set a threshold below which eigenvalues are considered to be "0"
  auto const threshold = Kokkos::max(max_eigen * zero_scaling, 5 * epsilon);

  // Invert diagonal ignoring "0"
  for (int i = 0; i < size; i++)
    diag(i) = (Kokkos::abs(diag(i)) < threshold) ? 0 : 1 / diag(i);

  // We invert the diagonal of 'mat', except if "0" is found

  // Then we fill out 'mat' as the pseudo inverse
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
    {
      mat(i, j) = 0;
      for (int k = 0; k < size; k++)
        mat(i, j) += unit(i, k) * diag(k) * unit(j, k);
    }
}

} // namespace ArborX::Details

#endif
