/****************************************************************************
 * Copyright (c) 2017-2023 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_KDOP_HPP
#define ARBORX_KDOP_HPP

#include <ArborX_Box.hpp>
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_Point.hpp>
#include <detail/ArborX_GeometryAlgorithms.hpp>
#include <kokkos_ext/ArborX_KokkosExtArithmeticTraits.hpp>
#include <misc/ArborX_Vector.hpp>

#include <Kokkos_Array.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_MinMax.hpp>

namespace ArborX
{
namespace Details
{
template <int DIM, int k, typename Coordinate>
struct KDOP_Directions;

template <typename Coordinate>
struct KDOP_Directions<2, 4, Coordinate>
{
  static constexpr int n_directions = 2;
  static KOKKOS_FUNCTION auto const &directions()
  {
    using Direction = Vector<2, Coordinate>;
    static constexpr Kokkos::Array<Direction, n_directions> directions = {
        Direction{1, 0},
        Direction{0, 1},
    };
    return directions;
  }
};

template <typename Coordinate>
struct KDOP_Directions<2, 8, Coordinate>
{
  static constexpr int n_directions = 4;
  static KOKKOS_FUNCTION auto const &directions()
  {
    using Direction = Vector<2, Coordinate>;
    static constexpr Kokkos::Array<Direction, n_directions> directions = {
        Direction{1, 0},
        Direction{0, 1},
        Direction{1, 1},
        Direction{1, -1},
    };
    return directions;
  }
};

template <typename Coordinate>
struct KDOP_Directions<3, 6, Coordinate>
{
  static constexpr int n_directions = 3;
  static KOKKOS_FUNCTION auto const &directions()
  {
    using Direction = Vector<3, Coordinate>;
    static constexpr Kokkos::Array<Direction, n_directions> directions = {
        Direction{1, 0, 0},
        Direction{0, 1, 0},
        Direction{0, 0, 1},
    };
    return directions;
  }
};

template <typename Coordinate>
struct KDOP_Directions<3, 14, Coordinate>
{
  static constexpr int n_directions = 7;
  static KOKKOS_FUNCTION auto const &directions()
  {
    using Direction = Vector<3, Coordinate>;
    static constexpr Kokkos::Array<Direction, n_directions> directions = {
        Direction{1, 0, 0},
        Direction{0, 1, 0},
        Direction{0, 0, 1},
        // corners
        Direction{1, 1, 1},
        Direction{1, -1, 1},
        Direction{1, 1, -1},
        Direction{1, -1, -1},
    };
    return directions;
  }
};

template <typename Coordinate>
struct KDOP_Directions<3, 18, Coordinate>
{
  static constexpr int n_directions = 9;
  static KOKKOS_FUNCTION auto const &directions()
  {
    using Direction = Vector<3, Coordinate>;
    static constexpr Kokkos::Array<Direction, n_directions> directions = {
        Direction{1, 0, 0},
        Direction{0, 1, 0},
        Direction{0, 0, 1},
        // edges
        Direction{1, 1, 0},
        Direction{1, 0, 1},
        Direction{0, 1, 1},
        Direction{1, -1, 0},
        Direction{1, 0, -1},
        Direction{0, 1, -1},
    };
    return directions;
  }
};

template <typename Coordinate>
struct KDOP_Directions<3, 26, Coordinate>
{
  static constexpr int n_directions = 13;
  static KOKKOS_FUNCTION auto const &directions()
  {
    using Direction = Vector<3, Coordinate>;
    static constexpr Kokkos::Array<Direction, n_directions> directions = {
        Direction{1, 0, 0},
        Direction{0, 1, 0},
        Direction{0, 0, 1},
        // edges
        Direction{1, 1, 0},
        Direction{1, 0, 1},
        Direction{0, 1, 1},
        Direction{1, -1, 0},
        Direction{1, 0, -1},
        Direction{0, 1, -1},
        // corners
        Direction{1, 1, 1},
        Direction{1, -1, 1},
        Direction{1, 1, -1},
        Direction{1, -1, -1},
    };
    return directions;
  }
};
} // namespace Details

namespace Experimental
{

template <int DIM, int k, typename Coordinate = float>
struct KDOP : public Details::KDOP_Directions<DIM, k, Coordinate>
{
  static constexpr int n_directions =
      Details::KDOP_Directions<DIM, k, Coordinate>::n_directions;
  Kokkos::Array<Coordinate, n_directions> _min_values;
  Kokkos::Array<Coordinate, n_directions> _max_values;

  KOKKOS_FUNCTION KDOP()
  {
    for (int i = 0; i < n_directions; ++i)
    {
      _min_values[i] =
          Details::KokkosExt::ArithmeticTraits::finite_max<Coordinate>::value;
      _max_values[i] =
          Details::KokkosExt::ArithmeticTraits::finite_min<Coordinate>::value;
    }
  }

  KOKKOS_FUNCTION explicit operator Box<DIM, Coordinate>() const
  {
    Box<DIM, Coordinate> box;
    expand(box, *this);
    return box;
  }
};
} // namespace Experimental
} // namespace ArborX

template <int DIM, int k, typename Coordinate>
struct ArborX::GeometryTraits::dimension<
    ArborX::Experimental::KDOP<DIM, k, Coordinate>>
{
  static constexpr int value = DIM;
};
template <int DIM, int k, typename Coordinate>
struct ArborX::GeometryTraits::tag<
    ArborX::Experimental::KDOP<DIM, k, Coordinate>>
{
  using type = KDOPTag;
};
template <int DIM, int k, typename Coordinate>
struct ArborX::GeometryTraits::coordinate_type<
    ArborX::Experimental::KDOP<DIM, k, Coordinate>>
{
  using type = Coordinate;
};

#endif
