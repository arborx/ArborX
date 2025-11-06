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

// Function to perform a single Jacobi rotation
template <typename Matrix, typename Unit>
KOKKOS_FUNCTION void jacobi_rotate(Matrix &A, Unit &V, int p, int q)
{
  using Value = typename Matrix::non_const_value_type;

  Value app = A(p, p);
  Value aqq = A(q, q);
  Value apq = A(p, q);

  // Calculate rotation angle phi ensuring |sin(phi)| < |cos(phi)|, see
  // https://en.wikipedia.org/wiki/Jacobi_rotation#Numerically_stable_computation
  // or §8.4 in
  // Golub, Gene H.; Van Loan, Charles F. (1996), Matrix Computations (3rd ed.),
  // Baltimore: Johns Hopkins University Press, ISBN 978-0-8018-5414-9
  Value tau = (aqq - app) / (2.0 * apq);
  Value t = 1.0 / (abs(tau) + sqrt(1.0 + tau * tau));
  if (tau < 0)
    t = -t;

  Value c = 1.0 / sqrt(1.0 + t * t); // cos(phi)
  Value s = t * c;                   // sin(phi)

  // Update the matrix A (A' = J^T * A * J)
  Value new_app = app - t * apq;
  Value new_aqq = aqq + t * apq;

  A(p, p) = new_app;
  A(q, q) = new_aqq;
  A(p, q) = A(q, p) = 0.0; // The goal of the rotation

  int n = A.extent_int(0);

  // Update the remaining off-diagonal elements in row/column p and q
  for (int i = 0; i < n; ++i)
  {
    if (i != p && i != q)
    {
      Value aip = A(i, p);
      Value aiq = A(i, q);
      A(i, p) = A(p, i) = c * aip - s * aiq;
      A(i, q) = A(q, i) = s * aip + c * aiq;
    }
  }

  // Update the eigenvector matrix V (V' = V * J)
  for (int i = 0; i < n; ++i)
  {
    Value vip = V(i, p);
    Value viq = V(i, q);
    V(i, p) = c * vip - s * viq;
    V(i, q) = s * vip + c * viq;
  }
}

// SVD of a symmetric matrix
// We must find U, E (diagonal and non-negative) and V such that A = U.E.V^T
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

  // Iterative diagonalization
  while (true)
  {
    // We have a guarantee that p < q
    auto const [max_val, p, q] = argmaxUpperTriangle(mat);
    if (max_val <= epsilon)
      break;

    jacobi_rotate(mat, unit, p, q);
  }

  // Extract eigenvalues from the diagonal of the resulting A
  for (int i = 0; i < size; ++i)
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
  constexpr Value epsilon = Kokkos::Experimental::epsilon_v<Value>;

  // We compute the max to get a range of the invertible eigenvalues
  auto max_eigen = epsilon;
  for (int i = 0; i < size; i++)
    max_eigen = Kokkos::max(Kokkos::abs(diag(i)), max_eigen);
  auto const threshold = size * max_eigen * epsilon;

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

} // namespace ArborX::Details

#endif
