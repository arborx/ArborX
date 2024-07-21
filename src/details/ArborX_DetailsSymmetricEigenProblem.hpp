/****************************************************************************
 * Copyright (c) 2024 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DETAILS_EIGENPROBLEM_HPP
#define ARBORX_DETAILS_EIGENPROBLEM_HPP

#include <ArborX_DetailsVector.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX::Details
{

// Deledalle, C. A., Denis, L., Tabti, S., & Tupin, F. Closed-form
// expressions of the eigen decomposition of 2 x 2 and 3 x 3 Hermitian
// matrices. [Research Report] Université de Lyon. 2017
// https://hal.science/hal-01501221/
template <typename Coordinate>
KOKKOS_FUNCTION void symmetricEigenProblem(
    Kokkos::Array<Kokkos::Array<Coordinate, 2>, 2> const &matrix,
    Coordinate eigenvalues[2],
    Details::Vector<2, Coordinate> (&eigenvectors)[2])
{
  // Matrix:
  // | a c |
  // | c b |
  auto const a = matrix[0][0];
  auto const c = matrix[0][1];
  auto const b = matrix[1][1];
  KOKKOS_ASSERT(matrix[1][0] == matrix[0][1]);

  if (c == 0)
  {
    eigenvalues[0] = a;
    eigenvalues[1] = b;
    eigenvectors[0] = {1, 0};
    eigenvectors[1] = {0, 1};
    return;
  }

  auto delta = Kokkos::sqrt(4 * c * c + (a - b) * (a - b));
  eigenvalues[0] = (a + b - delta) / 2;
  eigenvalues[1] = (a + b + delta) / 2;
  eigenvectors[0] = {eigenvalues[1] - b, c};
  eigenvectors[1] = {eigenvalues[0] - b, c};
}

} // namespace ArborX::Details

#endif
