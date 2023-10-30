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

#ifndef ARBORX_INTERP_DETAILS_SYMMETRIC_PSEUDO_INVERSE_SVD_HPP
#define ARBORX_INTERP_DETAILS_SYMMETRIC_PSEUDO_INVERSE_SVD_HPP

#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>

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
  using value_t = typename Matrix::non_const_value_type;

  struct
  {
    value_t max = 0;
    int row = 0;
    int col = 0;
  } result;

  int const size = mat.extent(0);
  for (int i = 0; i < size; i++)
    for (int j = i + 1; j < size; j++)
    {
      value_t val = Kokkos::abs(mat(i, j));
      if (result.max < val)
      {
        result.max = val;
        result.row = i;
        result.col = j;
      }
    }

  return result;
}

// Pseudo-inverse of symmetric matrices using SVD
// We must find U, E (diagonal and positive) and V such that A = U.E.V^T
// We also suppose, as the input, that A is symmetric, so U = SV where S is
// a sign matrix (only 1 or -1 on the diagonal, 0 elsewhere).
// Thus A = U.ES.U^T and A^-1 = U.[ ES^-1 ].U^T
template <typename AMatrix, typename ESMatrix, typename UMatrix>
KOKKOS_FUNCTION void
symmetricPseudoInverseSVDSerialKernel(AMatrix &A, ESMatrix &ES, UMatrix &U)
{
  ensureIsSquareSymmetricMatrix(A);
  static_assert(!std::is_const_v<typename AMatrix::value_type>,
                "A must be writable");
  ensureIsSquareMatrix(ES);
  static_assert(!std::is_const_v<typename ESMatrix::value_type>,
                "ES must be writable");
  ensureIsSquareMatrix(U);
  static_assert(!std::is_const_v<typename UMatrix::value_type>,
                "U must be writable");
  static_assert(std::is_same_v<typename AMatrix::value_type,
                               typename ESMatrix::value_type> &&
                    std::is_same_v<typename ESMatrix::value_type,
                                   typename UMatrix::value_type>,
                "All input matrices must have the same value type");
  KOKKOS_ASSERT(A.extent(0) == ES.extent(0) && ES.extent(0) == U.extent(0));
  using value_t = typename AMatrix::non_const_value_type;
  int const size = A.extent(0);

  // We first initialize U as the identity matrix and copy A to ES
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
    {
      U(i, j) = value_t(i == j);
      ES(i, j) = A(i, j);
    }

  static constexpr value_t epsilon = Kokkos::Experimental::epsilon_v<float>;
  while (true)
  {
    // We have a guarantee that p < q
    auto const [max_val, p, q] = argmaxUpperTriangle(ES);
    if (max_val <= epsilon)
      break;

    auto const a = ES(p, p);
    auto const b = ES(p, q);
    auto const c = ES(q, q);

    // Our submatrix is now
    // +----------+----------+   +---+---+
    // | ES(p, p) | ES(p, q) |   | a | b |
    // +----------+----------+ = +---+---+
    // | ES(q, p) | ES(q, q) |   | b | c |
    // +----------+----------+   +---+---+

    // Let's compute x, y and theta such that
    // +---+---+              +---+---+
    // | a | b |              | x | 0 |
    // +---+---+ = R(theta) * +---+---+ * R(theta)^T
    // | b | c |              | 0 | y |
    // +---+---+              +---+---+

    value_t cos_theta;
    value_t sin_theta;
    value_t x;
    value_t y;
    if (a == c)
    {
      cos_theta = Kokkos::sqrt(value_t(2)) / 2;
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

    // Now let's compute the following new values for U and ES
    // ES <- R'(theta)^T . ES . R'(theta)
    // U  <- U . R'(theta)

    // R'(theta)^T . ES . R'(theta)
    for (int i = 0; i < p; i++)
    {
      auto const es_ip = ES(i, p);
      auto const es_iq = ES(i, q);
      ES(i, p) = cos_theta * es_ip + sin_theta * es_iq;
      ES(i, q) = -sin_theta * es_ip + cos_theta * es_iq;
    }
    ES(p, p) = x;
    for (int i = p + 1; i < q; i++)
    {
      auto const es_pi = ES(p, i);
      auto const es_iq = ES(i, q);
      ES(p, i) = cos_theta * es_pi + sin_theta * es_iq;
      ES(i, q) = -sin_theta * es_pi + cos_theta * es_iq;
    }
    ES(q, q) = y;
    for (int i = q + 1; i < size; i++)
    {
      auto const es_pi = ES(p, i);
      auto const es_qi = ES(q, i);
      ES(p, i) = cos_theta * es_pi + sin_theta * es_qi;
      ES(q, i) = -sin_theta * es_pi + cos_theta * es_qi;
    }
    ES(p, q) = 0;

    // U . R'(theta)
    for (int i = 0; i < size; i++)
    {
      auto const u_ip = U(i, p);
      auto const u_iq = U(i, q);
      U(i, p) = cos_theta * u_ip + sin_theta * u_iq;
      U(i, q) = -sin_theta * u_ip + cos_theta * u_iq;
    }
  }

  // We compute the max to get a range of the invertible eigenvalues
  auto max_eigen = epsilon;
  for (int i = 0; i < size; i++)
    max_eigen = Kokkos::max(Kokkos::abs(ES(i, i)), max_eigen);

  // We invert the diagonal of ES, except if "0" is found
  for (int i = 0; i < size; i++)
    ES(i, i) = (Kokkos::abs(ES(i, i)) < max_eigen * epsilon) ? 0 : 1 / ES(i, i);

  // Then we fill out A as the pseudo inverse
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
    {
      value_t tmp = 0;
      for (int k = 0; k < size; k++)
        tmp += ES(k, k) * U(i, k) * U(j, k);
      A(i, j) = tmp;
    }
}

template <typename ExecutionSpace, typename InOutMatrices>
void symmetricPseudoInverseSVD(ExecutionSpace const &space,
                               InOutMatrices &matrices)
{
  // InOutMatrices is a list of square symmetric matrices (3D view)
  static_assert(Kokkos::is_view_v<InOutMatrices>, "matrices must be a view");
  static_assert(!std::is_const_v<typename InOutMatrices::value_type>,
                "matrices must be writable");
  static_assert(InOutMatrices::rank == 3,
                "matrices must be a list of square matrices");
  static_assert(
      KokkosExt::is_accessible_from<typename InOutMatrices::memory_space,
                                    ExecutionSpace>::value,
      "matrices must be accessible from the execution space");

  ARBORX_ASSERT(matrices.extent(1) == matrices.extent(2)); // Must be square

  InOutMatrices ESs(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::SymmetricPseudoInverseSVD::ESs"),
      matrices.extent(0), matrices.extent(1), matrices.extent(2));
  InOutMatrices Us(Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                                      "ArborX::SymmetricPseudoInverseSVD::Us"),
                   matrices.extent(0), matrices.extent(1), matrices.extent(2));

  Kokkos::parallel_for(
      "ArborX::SymmetricPseudoInverseSVD::computations",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, matrices.extent(0)),
      KOKKOS_LAMBDA(int const i) {
        auto A = Kokkos::subview(matrices, i, Kokkos::ALL, Kokkos::ALL);
        auto ES = Kokkos::subview(ESs, i, Kokkos::ALL, Kokkos::ALL);
        auto U = Kokkos::subview(Us, i, Kokkos::ALL, Kokkos::ALL);
        symmetricPseudoInverseSVDSerialKernel(A, ES, U);
      });
}

} // namespace ArborX::Interpolation::Details

#endif