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

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>

namespace Details
{

template <std::size_t Dim, std::size_t Deg>
KOKKOS_FUNCTION constexpr Kokkos::Array<Kokkos::Array<std::size_t, Dim>, Deg>
polynomialBasisColumnSizes()
{
  Kokkos::Array<Kokkos::Array<std::size_t, Dim>, Deg> arr{};

  for (std::size_t dim = 0; dim < Dim; dim++)
    arr[0][dim] = 1;
  for (std::size_t deg = 0; deg < Deg; deg++)
    arr[deg][0] = 1;

  for (std::size_t deg = 1; deg < Deg; deg++)
    for (std::size_t dim = 1; dim < Dim; dim++)
      arr[deg][dim] = arr[deg - 1][dim] + arr[deg][dim - 1];

  return arr;
}

template <std::size_t Dim, std::size_t Deg>
KOKKOS_FUNCTION constexpr std::size_t polynomialBasisSize()
{
  auto arr = polynomialBasisColumnSizes<Dim, Deg>();
  std::size_t size = 1;

  for (std::size_t deg = 0; deg < Deg; deg++)
    for (std::size_t dim = 0; dim < Dim; dim++)
      size += arr[deg][dim];

  return size;
}

template <typename Point, std::size_t Deg>
static constexpr std::size_t polynomialBasisSizeFromT =
    polynomialBasisSize<ArborX::GeometryTraits::dimension_v<Point>, Deg>();

template <typename Points, std::size_t Deg>
static constexpr std::size_t polynomialBasisSizeFromAT =
    polynomialBasisSizeFromT<
        typename ArborX::Details::AccessTraitsHelper<
            ArborX::AccessTraits<Points, ArborX::PrimitivesTag>>::type,
        Deg>;

template <typename Point, std::size_t Deg>
KOKKOS_FUNCTION auto polynomialBasis(Point const &p)
{
  static constexpr std::size_t dimension =
      ArborX::GeometryTraits::dimension_v<Point>;
  static constexpr auto column_details =
      polynomialBasisColumnSizes<dimension, Deg>();
  using value_t = typename ArborX::GeometryTraits::coordinate_type<Point>::type;

  Kokkos::Array<value_t, polynomialBasisSize<dimension, Deg>()> arr{};
  arr[0] = value_t(1);

  std::size_t prev_col = 0;
  std::size_t curr_col = 1;
  for (std::size_t deg = 0; deg < Deg; deg++)
  {
    std::size_t loc_offset = curr_col;
    for (std::size_t dim = 0; dim < dimension; dim++)
    {
      // copy the previous column and multply by p[dim]
      for (std::size_t i = 0; i < column_details[deg][dim]; i++)
        arr[loc_offset + i] = arr[prev_col + i] * p[dim];

      loc_offset += column_details[deg][dim];
    }

    prev_col = curr_col;
    curr_col = loc_offset;
  }

  return arr;
}

template <std::size_t Deg>
static constexpr std::integral_constant<std::size_t, Deg> degree{};

} // namespace Details
