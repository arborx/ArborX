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

#ifndef ARBORX_INTERP_DETAILS_POLYNOMIAL_BASIS_HPP
#define ARBORX_INTERP_DETAILS_POLYNOMIAL_BASIS_HPP

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace ArborX::Interpolation
{

namespace Details
{

// Polynomial basis is computed in the same way as the following example
// diagrams. Each cell is the product of the variable times everything on the
// left cell and above.
//
// For example, the "degree 3 / variable y" has size 3 because it multiplies
// every element of "degree 2 / variable y" and "degree 2 / variable x".
//
//   Quadratic |  3D  ||        Cubic      |  2D
// ------------+----- || ------------------+-----
//    degree   | var  ||       degree      | var
// ---+---+----+      || ---+---+----+-----+
//  0 | 1 |  2 |      ||  0 | 1 |  2 |  3  |
// ---+---+----+      || ---+---+----+-----+
// ---+---+----+      || ---+---+----+-----+
//  1 |   |    |      ||  1 |   |    |     |
// ---+---+----+----- || ---+---+----+-----+-----
//    | x |    |  x   ||    | x |    |     |  x
//    |   | xx |      ||    |   | xx |     |
//    +---+----+----- ||    |   |    | xxx |
//    | y |    |  y   ||    +---+----+-----+-----
//    |   | xy |      ||    | y |    |     |  y
//    |   | yy |      ||    |   | xy |     |
//    +---+----+----- ||    |   | yy |     |
//    | z |    |  z   ||    |   |    | xxy |
//    |   | xz |      ||    |   |    | xyy |
//    |   | yz |      ||    |   |    | yyy |
//    |   | zz |      ||

// The size of each cell is 1 if it is at degree 1 or at variable x
// And the sum of the cells' size to the left and above. (1 alone is not
// counted). This essentially creates Pascal's triangle where line n corresponds
// to the "dim + deg = n"-th diagonal
template <std::size_t Dim, std::size_t Deg>
KOKKOS_FUNCTION constexpr auto polynomialBasisCellSizes()
{
  static_assert(Deg != 0 && Dim != 0,
                "Unable to compute cell sizes for a constant polynomial basis");

  struct
  {
    std::size_t arr[Deg][Dim]{};
  } result;
  auto &arr = result.arr;

  for (std::size_t dim = 0; dim < Dim; dim++)
    arr[0][dim] = 1;

  for (std::size_t deg = 0; deg < Deg; deg++)
    arr[deg][0] = 1;

  for (std::size_t deg = 1; deg < Deg; deg++)
    for (std::size_t dim = 1; dim < Dim; dim++)
      arr[deg][dim] = arr[deg - 1][dim] + arr[deg][dim - 1];

  return result;
}

// This returns the size of the polynomial basis, which is the sum of all the
// cells' sizes and 1.
template <std::size_t Dim, std::size_t Deg>
KOKKOS_FUNCTION constexpr std::size_t polynomialBasisSize()
{
  if constexpr (Deg != 0 && Dim != 0)
  {
    auto [arr] = polynomialBasisCellSizes<Dim, Deg>();
    std::size_t size = 1;

    for (std::size_t deg = 0; deg < Deg; deg++)
      for (std::size_t dim = 0; dim < Dim; dim++)
        size += arr[deg][dim];

    return size;
  }
  else
  {
    return 1;
  }
}

// This builds the array as described above
template <std::size_t Dim, std::size_t Deg, typename Point>
KOKKOS_FUNCTION auto polynomialBasis(Point const &p)
{
  using value_t = std::decay_t<decltype(p[0])>;
  Kokkos::Array<value_t, polynomialBasisSize<Dim, Deg>()> arr{};
  arr[0] = value_t(1);

  if constexpr (Deg != 0 && Dim != 0)
  {
    // Cannot use struct binding with constexpr
    static constexpr auto cell_sizes_struct =
        polynomialBasisCellSizes<Dim, Deg>();
    static constexpr auto &cell_sizes = cell_sizes_struct.arr;

    std::size_t prev_col = 0;
    std::size_t curr_col = 1;

    for (std::size_t deg = 0; deg < Deg; deg++)
    {
      std::size_t loc_offset = curr_col;
      for (std::size_t dim = 0; dim < Dim; dim++)
      {
        // copy the previous column and multply by p[dim]
        for (std::size_t i = 0; i < cell_sizes[deg][dim]; i++)
          arr[loc_offset + i] = arr[prev_col + i] * p[dim];

        loc_offset += cell_sizes[deg][dim];
      }

      prev_col = curr_col;
      curr_col = loc_offset;
    }
  }

  return arr;
}

template <std::size_t Deg>
struct PolynomialDegree : std::integral_constant<std::size_t, Deg>
{};

} // namespace Details
template <std::size_t Deg>
static constexpr Details::PolynomialDegree<Deg> PolynomialDegree{};

} // namespace ArborX::Interpolation

#endif