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

#include <ArborX_GeometryTraits.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace ArborX::Interpolation
{

namespace Details
{

// The goal of these functions is to evaluate the polynomial basis at any degree
// and dimension. For example, the polynomial basis of degree 2 evaluated at a
// point {x, y, z} would be [1, x, y, z, xx, xy, yy, xz, yz, zz].
//
// To compute it, the list of values is sliced against its degree and highest
// dimension. From the previous list, we would have the following sublists:
// - [1]                             at degree 0
// - [x], [y] and [z]                at degree 1
// - [xx], [xy, yy] and [xz, yz, zz] at degree 2
// One can then infer a recursive pattern that will be used to build the list:
// - [1]                             at degree 0
// - x*[1], y*[1] and z*[1]          at degree 1
// - x*[x], y*[x, y] and z*[x, y, z] at degree 2
// So, given a slice at degree n and dimension u, its values would be the
// product of all the slices of degree n-1 and of dimension u or less with the
// coordinate of dimension u.
//
// As another example, if we can take the polynomial basis of degree 3 evaluated
// at a point {x, y} would be [1, x, y, xx, xy, yy, xxx, xxy, xyy, yyy]. Its
// slices are:
// - [1]                       at degree 0
// - [x] and [y]               at degree 1
// - [xx] and [xy, yy]         at degree 2
// - [xxx] and [xxy, xyy, yyy] at degree 3
// And its recursive pattern:
// - [1]                       at degree 0
// - x*[1] and y*[1]           at degree 1
// - x*[x] and y*[x, y]        at degree 2
// - x*[xx] and y*[xx, xy, yy] at degree 3
//
// The lengths for the slices would be
// Deg \ Dim | x | y | z
// ----------+---+---+---
//     1     | 1 | 1 | 1
//     2     | 1 | 2 | 3
//     3     | 1 | 3 | 6
template <std::size_t DIM, std::size_t Degree>
KOKKOS_FUNCTION constexpr auto polynomialBasisSliceLengths()
{
  static_assert(DIM > 0, "Polynomial basis with no dimension is invalid");
  static_assert(
      Degree > 0,
      "Unable to compute slice lengths for a constant polynomial basis");

  struct
  {
    std::size_t arr[Degree][DIM]{};
  } result;
  auto &arr = result.arr;

  for (std::size_t dim = 0; dim < DIM; dim++)
    arr[0][dim] = 1;

  for (std::size_t deg = 0; deg < Degree; deg++)
    arr[deg][0] = 1;

  for (std::size_t deg = 1; deg < Degree; deg++)
    for (std::size_t dim = 1; dim < DIM; dim++)
      arr[deg][dim] = arr[deg - 1][dim] + arr[deg][dim - 1];

  return result;
}

// This returns the size of the polynomial basis. Counting the constant 1, the
// size would be "Deg + Dim choose Dim" or "Deg + Dim choose Deg".
template <std::size_t DIM, std::size_t Degree>
KOKKOS_FUNCTION constexpr std::size_t polynomialBasisSize()
{
  static_assert(DIM > 0, "Polynomial basis with no dimension is invalid");

  std::size_t result = 1;
  for (std::size_t k = 0; k < Kokkos::min(DIM, Degree); ++k)
    result = result * (DIM + Degree - k) / (k + 1);
  return result;
}

// This creates the list by building each slices in-place
template <std::size_t Degree, typename Point>
KOKKOS_FUNCTION auto evaluatePolynomialBasis(Point const &p)
{
  GeometryTraits::check_valid_geometry_traits(Point{});
  static_assert(GeometryTraits::is_point<Point>::value,
                "point must be a point");
  static constexpr std::size_t DIM = GeometryTraits::dimension_v<Point>;
  using value_t = typename GeometryTraits::coordinate_type<Point>::type;
  static_assert(DIM > 0, "Polynomial basis with no dimension is invalid");

  Kokkos::Array<value_t, polynomialBasisSize<DIM, Degree>()> arr{};
  arr[0] = value_t(1);

  if constexpr (Degree > 0)
  {
    // Cannot use structured binding with constexpr
    static constexpr auto slice_lengths_struct =
        polynomialBasisSliceLengths<DIM, Degree>();
    auto &slice_lengths = slice_lengths_struct.arr;

    std::size_t prev_col = 0;
    std::size_t curr_col = 1;

    for (std::size_t deg = 0; deg < Degree; deg++)
    {
      std::size_t loc_offset = curr_col;
      for (std::size_t dim = 0; dim < DIM; dim++)
      {
        // copy the previous column and multply by p[dim]
        for (std::size_t i = 0; i < slice_lengths[deg][dim]; i++)
          arr[loc_offset + i] = arr[prev_col + i] * p[dim];

        loc_offset += slice_lengths[deg][dim];
      }

      prev_col = curr_col;
      curr_col = loc_offset;
    }
  }

  return arr;
}

} // namespace Details

template <std::size_t Degree>
struct PolynomialDegree : std::integral_constant<std::size_t, Degree>
{};

} // namespace ArborX::Interpolation

#endif