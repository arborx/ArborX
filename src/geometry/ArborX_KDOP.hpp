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
#include <ArborX_Vector.hpp>
#include <details/ArborX_Algorithms.hpp>
#include <kokkos_ext/ArborX_KokkosExtArithmeticTraits.hpp>

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

namespace ArborX::Details::Dispatch
{

template <typename KDOP1, typename KDOP2>
struct expand<KDOPTag, KDOPTag, KDOP1, KDOP2>
{
  KOKKOS_FUNCTION static void apply(KDOP1 &that, KDOP2 const &other)
  {
    using Kokkos::max;
    using Kokkos::min;

    constexpr int n_directions = KDOP1::n_directions;
    static_assert(KDOP2::n_directions == n_directions);
    for (int i = 0; i < n_directions; ++i)
    {
      that._min_values[i] = min(that._min_values[i], other._min_values[i]);
      that._max_values[i] = max(that._max_values[i], other._max_values[i]);
    }
  }
};

template <typename KDOP, typename Point>
struct expand<KDOPTag, PointTag, KDOP, Point>
{
  KOKKOS_FUNCTION static void apply(KDOP &kdop, Point const &point)
  {
    using Kokkos::max;
    using Kokkos::min;

    constexpr int DIM = GeometryTraits::dimension_v<Point>;
    constexpr int n_directions = KDOP::n_directions;
    for (int i = 0; i < n_directions; ++i)
    {
      auto const &dir = kdop.directions()[i];
      auto proj_i = point[0] * dir[0];
      for (int d = 1; d < DIM; ++d)
        proj_i += point[d] * dir[d];

      kdop._min_values[i] = min(kdop._min_values[i], proj_i);
      kdop._max_values[i] = max(kdop._max_values[i], proj_i);
    }
  }
};

template <typename KDOP, typename Box>
struct expand<KDOPTag, BoxTag, KDOP, Box>
{
  KOKKOS_FUNCTION static void apply(KDOP &kdop, Box const &box)
  {
    constexpr int DIM = GeometryTraits::dimension_v<KDOP>;
    static_assert(DIM == 2 || DIM == 3);

    // NOTE if any of the ranges is invalid, the code below would actually
    // expand the KDOP which is not what we want.
    // We may revisit this later and decide passing a valid box becomes a
    // precondition but this would be a breaking change (going from a wide to a
    // narrow contract).
    for (int d = 0; d < DIM; ++d)
      if (box.minCorner()[d] > box.maxCorner()[d])
        return;

    using Point = std::decay_t<decltype(box.minCorner())>;
    if constexpr (DIM == 3)
    {
      auto const xmin = box.minCorner()[0];
      auto const ymin = box.minCorner()[1];
      auto const zmin = box.minCorner()[2];
      auto const xmax = box.maxCorner()[0];
      auto const ymax = box.maxCorner()[1];
      auto const zmax = box.maxCorner()[2];
      for (auto const &point : {
               Point{xmin, ymin, zmin},
               Point{xmin, ymax, zmin},
               Point{xmin, ymin, zmax},
               Point{xmin, ymax, zmax},
               Point{xmax, ymin, zmin},
               Point{xmax, ymax, zmin},
               Point{xmax, ymin, zmax},
               Point{xmax, ymax, zmax},
           })
      {
        Details::expand(kdop, point);
      }
    }
    else
    {
      auto const xmin = box.minCorner()[0];
      auto const ymin = box.minCorner()[1];
      auto const xmax = box.maxCorner()[0];
      auto const ymax = box.maxCorner()[1];
      for (auto const &point : {
               Point{xmin, ymin},
               Point{xmin, ymax},
               Point{xmax, ymin},
               Point{xmax, ymax},
           })
      {
        Details::expand(kdop, point);
      }
    }
  }
};

template <typename Box, typename KDOP>
struct expand<BoxTag, KDOPTag, Box, KDOP>
{
  KOKKOS_FUNCTION static void apply(Box &box, KDOP const &kdop)
  {
    constexpr int DIM = GeometryTraits::dimension_v<KDOP>;

    // WARNING implicit requirement on KDOP first DIM directions
    Box other;
    for (int d = 0; d < DIM; ++d)
    {
      other.minCorner()[d] = kdop._min_values[d];
      other.maxCorner()[d] = kdop._max_values[d];
    }
    Details::expand(box, other);
  }
};

template <typename KDOP, typename Box>
struct intersects<KDOPTag, BoxTag, KDOP, Box>
{
  KOKKOS_FUNCTION static constexpr bool apply(KDOP const &kdop, Box const &box)
  {
    KDOP other{};
    Details::expand(other, box);
    return Details::intersects(kdop, other);
  }
};

template <typename Box, typename KDOP>
struct intersects<BoxTag, KDOPTag, Box, KDOP>
{
  KOKKOS_FUNCTION static constexpr bool apply(Box const &box, KDOP const &kdop)
  {
    return Details::intersects(kdop, box);
  }
};

template <typename Point, typename KDOP>
struct intersects<PointTag, KDOPTag, Point, KDOP>
{
  KOKKOS_FUNCTION static constexpr bool apply(Point const &point,
                                              KDOP const &kdop)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Point>;
    constexpr int n_directions = KDOP::n_directions;
    for (int i = 0; i < n_directions; ++i)
    {
      auto const &dir = kdop.directions()[i];
      auto proj_i = point[0] * dir[0];
      for (int d = 1; d < DIM; ++d)
        proj_i += point[d] * dir[d];

      if (proj_i < kdop._min_values[i] || proj_i > kdop._max_values[i])
        return false;
    }
    return true;
  }
};

template <typename KDOP, typename Point>
struct intersects<KDOPTag, PointTag, KDOP, Point>
{
  KOKKOS_FUNCTION static constexpr bool apply(KDOP const &kdop,
                                              Point const &point)
  {
    return Details::intersects(point, kdop);
  }
};

template <typename KDOP1, typename KDOP2>
struct intersects<KDOPTag, KDOPTag, KDOP1, KDOP2>
{
  KOKKOS_FUNCTION static constexpr bool apply(KDOP1 const &kdop,
                                              KDOP2 const &other)
  {
    constexpr int n_directions = KDOP1::n_directions;
    static_assert(KDOP2::n_directions == n_directions);
    for (int i = 0; i < kdop.n_directions; ++i)
    {
      if (other._max_values[i] < kdop._min_values[i] ||
          other._min_values[i] > kdop._max_values[i])
      {
        return false;
      }
    }
    return true;
  }
};

template <typename KDOP>
struct centroid<KDOPTag, KDOP>
{
  KOKKOS_FUNCTION static auto apply(KDOP const &kdop)
  {
    constexpr int DIM = GeometryTraits::dimension_v<KDOP>;
    using Coordinate = GeometryTraits::coordinate_type_t<KDOP>;

    // WARNING implicit requirement on KDOP first DIM directions
    Point<DIM, Coordinate> point;
    for (int d = 0; d < DIM; ++d)
      point[d] = (kdop._min_values[d] + kdop._max_values[d]) / 2;
    return point;
  }
};

} // namespace ArborX::Details::Dispatch

#endif
