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
KOKKOS_INLINE_FUNCTION void isSquareMatrix(Matrix const &mat)
{
  static_assert(Kokkos::is_view_v<Matrix>, "Matrix must be a view");
  static_assert(Matrix::rank == 2, "Matrix must be 2D");
  KOKKOS_ASSERT(mat.extent(0) == mat.extent(1));
}

// Gets the argmax from the upper triamgle part of a matrix
template <typename Matrix>
KOKKOS_FUNCTION auto argmaxUpperTriangle(Matrix const &mat)
{
  isSquareMatrix(mat);
  using value_t = typename Matrix::non_const_value_type;
  std::size_t const size = mat.extent(0);

  struct
  {
    value_t max = 0;
    std::size_t row = 0;
    std::size_t col = 0;
  } result;

  for (std::size_t i = 0; i < size; i++)
    for (std::size_t j = i + 1; j < size; j++)
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
// a sign matrix (only 1 or -1 in the diagonal, 0 elsewhere).
// Thus A = U.E.S.U^T and A^-1 = U.[ E^-1.S ].U^T
// Here inv_matrix is A and A^-1, sigma is E.S and ortho is U
template <typename InvMatrix, typename Sigma, typename Ortho>
KOKKOS_FUNCTION void
symmetricPseudoInverseSVDSerialKernel(InvMatrix &inv_matrix, Sigma &sigma,
                                      Ortho &ortho)
{
  isSquareMatrix(inv_matrix);
  static_assert(!std::is_const_v<typename InvMatrix::value_type>,
                "inv_matrix must be writable");
  isSquareMatrix(sigma);
  static_assert(!std::is_const_v<typename Sigma::value_type>,
                "sigma must be writable");
  isSquareMatrix(ortho);
  static_assert(!std::is_const_v<typename Ortho::value_type>,
                "ortho must be writable");
  static_assert(std::is_same_v<typename InvMatrix::value_type,
                               typename Sigma::value_type> &&
                    std::is_same_v<typename Sigma::value_type,
                                   typename Ortho::value_type>,
                "Each input matrix must have the same value type");
  KOKKOS_ASSERT(inv_matrix.extent(0) == sigma.extent(0) &&
                sigma.extent(0) == ortho.extent(0));
  using value_t = typename InvMatrix::non_const_value_type;
  std::size_t const size = inv_matrix.extent(0);

  // We first initialize ortho as the identity matrix and copy inv_matrix to
  // sigma
  for (std::size_t i = 0; i < size; i++)
    for (std::size_t j = 0; j < size; j++)
    {
      ortho(i, j) = value_t(i == j);
      sigma(i, j) = inv_matrix(i, j);
    }

  static constexpr value_t epsilon = Kokkos::Experimental::epsilon_v<float>;
  while (true)
  {
    // We have a guarantee that p < q
    auto [max, p, q] = argmaxUpperTriangle(sigma);
    if (max <= epsilon)
      break;

    value_t const a = sigma(p, p);
    value_t const b = sigma(p, q);
    value_t const c = sigma(q, q);

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

    value_t cos, sin, x, y;
    if (a == c)
    {
      cos = Kokkos::sqrt(value_t(2)) / 2;
      sin = Kokkos::sqrt(value_t(2)) / 2;
      x = a + b;
      y = a - b;
    }
    else
    {
      value_t const u = (2 * b) / (a - c);
      value_t const v = 1 / Kokkos::sqrt(u * u + 1);
      cos = Kokkos::sqrt((1 + v) / 2);
      sin = Kokkos::copysign(Kokkos::sqrt((1 - v) / 2), u);
      x = (a + c + (a - c) / v) / 2;
      y = a + c - x;
    }

    // Now lets compute the following new values for U amd E.S
    // M <- R'(theta)^T . S . R'(theta)
    // U <- U . R'(theta)

    // R'(theta)^T . S . R'(theta)
    std::size_t i = 0;
    for (; i < p; i++)
    {
      value_t const s_ip = sigma(i, p);
      value_t const s_iq = sigma(i, q);
      sigma(i, p) = cos * s_ip + sin * s_iq;
      sigma(i, q) = -sin * s_ip + cos * s_iq;
    }
    sigma(p, p) = x;
    i++;
    for (; i < q; i++)
    {
      value_t const s_pi = sigma(p, i);
      value_t const s_iq = sigma(i, q);
      sigma(p, i) = cos * s_pi + sin * s_iq;
      sigma(i, q) = -sin * s_pi + cos * s_iq;
    }
    sigma(q, q) = y;
    i++;
    for (; i < size; i++)
    {
      value_t const s_pi = sigma(p, i);
      value_t const s_qi = sigma(q, i);
      sigma(p, i) = cos * s_pi + sin * s_qi;
      sigma(q, i) = -sin * s_pi + cos * s_qi;
    }
    sigma(p, q) = 0;

    // U . R'(theta)
    for (std::size_t i = 0; i < size; i++)
    {
      value_t const o_ip = ortho(i, p);
      value_t const o_iq = ortho(i, q);
      ortho(i, p) = cos * o_ip + sin * o_iq;
      ortho(i, q) = -sin * o_ip + cos * o_iq;
    }
  }

  // We compute the max to get a range of the invertible eigen values
  value_t max_eigen = epsilon;
  for (std::size_t i = 0; i < size; i++)
    max_eigen = Kokkos::max(Kokkos::abs(sigma(i, i)), max_eigen);

  // We inverse the diagonal of S, except if "0" is found
  for (std::size_t i = 0; i < size; i++)
    sigma(i, i) =
        (Kokkos::abs(sigma(i, i)) < max_eigen * epsilon) ? 0 : 1 / sigma(i, i);

  // Then we fill out IO as the pseudo inverse
  for (std::size_t i = 0; i < size; i++)
    for (std::size_t j = 0; j < size; j++)
    {
      value_t tmp = 0;
      for (std::size_t k = 0; k < size; k++)
        tmp += sigma(k, k) * ortho(i, k) * ortho(j, k);
      inv_matrix(i, j) = tmp;
    }
}

template <typename ExecutionSpace, typename InvMatrices>
void symmetricPseudoInverseSVD(ExecutionSpace const &space,
                               InvMatrices &inv_matrices)
{
  // InvMatrices is a single or list of matrices (i.e 2 or 3D view)
  static_assert(Kokkos::is_view_v<InvMatrices>, "inv_matrices must be a view");
  static_assert(!std::is_const_v<typename InvMatrices::value_type>,
                "inv_matrices must be writable");
  static_assert(InvMatrices::rank == 3 || InvMatrices::rank == 2,
                "inv_matrices must be a matrix or a list of matrices");
  static_assert(
      KokkosExt::is_accessible_from<typename InvMatrices::memory_space,
                                    ExecutionSpace>::value,
      "inv_matrices must be accessible from the execution space");

  if constexpr (InvMatrices::rank == 3)
  {
    ARBORX_ASSERT(inv_matrices.extent(1) ==
                  inv_matrices.extent(2)); // Must be square

    InvMatrices sigmas(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::SymmetricPseudoInverseSVD::sigmas"),
        inv_matrices.extent(0), inv_matrices.extent(1), inv_matrices.extent(2));
    InvMatrices orthos(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::SymmetricPseudoInverseSVD::orthos"),
        inv_matrices.extent(0), inv_matrices.extent(1), inv_matrices.extent(2));

    Kokkos::parallel_for(
        "ArborX::SymmetricPseudoInverseSVD::computation_list",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, inv_matrices.extent(0)),
        KOKKOS_LAMBDA(int const i) {
          auto inv_matrix =
              Kokkos::subview(inv_matrices, i, Kokkos::ALL, Kokkos::ALL);
          auto sigma = Kokkos::subview(sigmas, i, Kokkos::ALL, Kokkos::ALL);
          auto ortho = Kokkos::subview(orthos, i, Kokkos::ALL, Kokkos::ALL);
          symmetricPseudoInverseSVDSerialKernel(inv_matrix, sigma, ortho);
        });
  }
  else if constexpr (InvMatrices::rank == 2)
  {
    ARBORX_ASSERT(inv_matrices.extent(0) ==
                  inv_matrices.extent(1)); // Must be square

    InvMatrices sigma(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::SymmetricPseudoInverseSVD::sigma"),
        inv_matrices.extent(0), inv_matrices.extent(1));
    InvMatrices ortho(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::SymmetricPseudoInverseSVD::ortho"),
        inv_matrices.extent(0), inv_matrices.extent(1));

    Kokkos::parallel_for(
        "ArborX::SymmetricPseudoInverseSVD::computation",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, 1),
        KOKKOS_LAMBDA(int const) {
          symmetricPseudoInverseSVDSerialKernel(inv_matrices, sigma, ortho);
        });
  }
}

} // namespace ArborX::Interpolation::Details

#endif