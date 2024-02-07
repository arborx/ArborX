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
#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_Point.hpp>
#include <ArborX_Sphere.hpp>

#include <Kokkos_Array.hpp>
#include <Kokkos_Macros.hpp>

namespace ArborX
{
namespace Details
{
template <typename Coordinate>
struct Direction
{
  Coordinate _data[3];
};

template <int k, typename Coordinate>
struct KDOP_Directions;

template <typename Coordinate>
struct KDOP_Directions<6, Coordinate>
{
protected:
  static constexpr int n_directions = 3;
  static KOKKOS_FUNCTION auto const &directions()
  {
    static constexpr Kokkos::Array<Direction<Coordinate>, n_directions>
        directions = {
            Direction<Coordinate>{1, 0, 0},
            Direction<Coordinate>{0, 1, 0},
            Direction<Coordinate>{0, 0, 1},
        };
    return directions;
  }
};

template <typename Coordinate>
struct KDOP_Directions<14, Coordinate>
{
protected:
  static constexpr int n_directions = 7;
  static KOKKOS_FUNCTION auto const &directions()
  {
    static constexpr Kokkos::Array<Direction<Coordinate>, n_directions>
        directions = {
            Direction<Coordinate>{1, 0, 0},
            Direction<Coordinate>{0, 1, 0},
            Direction<Coordinate>{0, 0, 1},
            // corners
            Direction<Coordinate>{1, 1, 1},
            Direction<Coordinate>{1, -1, 1},
            Direction<Coordinate>{1, 1, -1},
            Direction<Coordinate>{1, -1, -1},
        };
    return directions;
  }
};

template <typename Coordinate>
struct KDOP_Directions<18, Coordinate>
{
protected:
  static constexpr int n_directions = 9;
  static KOKKOS_FUNCTION auto const &directions()
  {
    static constexpr Kokkos::Array<Direction<Coordinate>, n_directions>
        directions = {
            Direction<Coordinate>{1, 0, 0},
            Direction<Coordinate>{0, 1, 0},
            Direction<Coordinate>{0, 0, 1},
            // edges
            Direction<Coordinate>{1, 1, 0},
            Direction<Coordinate>{1, 0, 1},
            Direction<Coordinate>{0, 1, 1},
            Direction<Coordinate>{1, -1, 0},
            Direction<Coordinate>{1, 0, -1},
            Direction<Coordinate>{0, 1, -1},
        };
    return directions;
  }
};

template <typename Coordinate>
struct KDOP_Directions<26, Coordinate>
{
protected:
  static constexpr int n_directions = 13;
  static KOKKOS_FUNCTION auto const &directions()
  {
    static constexpr Kokkos::Array<Direction<Coordinate>, n_directions>
        directions = {
            Direction<Coordinate>{1, 0, 0},
            Direction<Coordinate>{0, 1, 0},
            Direction<Coordinate>{0, 0, 1},
            // edges
            Direction<Coordinate>{1, 1, 0},
            Direction<Coordinate>{1, 0, 1},
            Direction<Coordinate>{0, 1, 1},
            Direction<Coordinate>{1, -1, 0},
            Direction<Coordinate>{1, 0, -1},
            Direction<Coordinate>{0, 1, -1},
            // corners
            Direction<Coordinate>{1, 1, 1},
            Direction<Coordinate>{1, -1, 1},
            Direction<Coordinate>{1, 1, -1},
            Direction<Coordinate>{1, -1, -1},
        };
    return directions;
  }
};
template <typename Coordinate>
KOKKOS_INLINE_FUNCTION auto project(Point const &p,
                                    Direction<Coordinate> const &d)
{
  static_assert(
      std::is_same_v<GeometryTraits::coordinate_type<Point>::type, Coordinate>);

  Coordinate r = 0;
  for (int i = 0; i < 3; ++i)
  {
    r += p[i] * d._data[i];
  }
  return r;
}
} // namespace Details

namespace Experimental
{

template <int k, typename Coordinate = float>
struct KDOP : private Details::KDOP_Directions<k, Coordinate>
{
  static constexpr int n_directions =
      Details::KDOP_Directions<k, Coordinate>::n_directions;
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
  KOKKOS_FUNCTION KDOP &operator+=(Point const &p)
  {
    using Details::KokkosExt::max;
    using Details::KokkosExt::min;
    for (int i = 0; i < n_directions; ++i)
    {
      auto const proj_i = Details::project(p, this->directions()[i]);
      _min_values[i] = min(_min_values[i], proj_i);
      _max_values[i] = max(_max_values[i], proj_i);
    }
    return *this;
  }
  KOKKOS_FUNCTION KDOP &operator+=(Box const &b)
  {
    // NOTE if any of the ranges is invalid, the code below would actually
    // expand the KDOP which is not what we want.
    // We may revisit this later and decide passing a valid box becomes a
    // precondition but this would be a breaking change (going from a wide to a
    // narrow contract).
    for (int i = 0; i < 3; ++i)
    {
      if (b.minCorner()[i] > b.maxCorner()[i])
      {
        return *this;
      }
    }

    auto const xmin = b.minCorner()[0];
    auto const ymin = b.minCorner()[1];
    auto const zmin = b.minCorner()[2];
    auto const xmax = b.maxCorner()[0];
    auto const ymax = b.maxCorner()[1];
    auto const zmax = b.maxCorner()[2];
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
      *this += point;
    }
    return *this;
  }
  KOKKOS_FUNCTION KDOP &operator+=(KDOP const &other)
  {
    using Details::KokkosExt::max;
    using Details::KokkosExt::min;
    for (int i = 0; i < n_directions; ++i)
    {
      _min_values[i] = min(_min_values[i], other._min_values[i]);
      _max_values[i] = max(_max_values[i], other._max_values[i]);
    }
    return *this;
  }
  KOKKOS_FUNCTION explicit operator Box() const
  {
    // WARNING implicit requirement on KDOP first three directions
    Box b{};
    for (int i = 0; i < 3; ++i)
    {
      b.minCorner()[i] = _min_values[i];
      b.maxCorner()[i] = _max_values[i];
    }
    return b;
  }
  KOKKOS_FUNCTION bool intersects(Point const &point) const
  {
    for (int i = 0; i < n_directions; ++i)
    {
      auto const proj_i = Details::project(point, this->directions()[i]);
      if (proj_i < _min_values[i] || proj_i > _max_values[i])
      {
        return false;
      }
    }
    return true;
  }
  KOKKOS_FUNCTION bool intersects(Box const &box) const
  {
    KDOP other{};
    other += box;
    return intersects(other);
  }
  KOKKOS_FUNCTION bool intersects(KDOP<k> const &other) const
  {
    for (int i = 0; i < n_directions; ++i)
    {
      if (other._max_values[i] < _min_values[i] ||
          other._min_values[i] > _max_values[i])
      {
        return false;
      }
    }
    return true;
  }
};

template <int k>
KOKKOS_INLINE_FUNCTION void expand(KDOP<k> &that, KDOP<k> const &other)
{
  that += other;
}

template <int k>
KOKKOS_INLINE_FUNCTION void expand(KDOP<k> &that, Point const &point)
{
  that += point;
}

template <int k>
KOKKOS_INLINE_FUNCTION void expand(KDOP<k> &that, Box const &box)
{
  that += box;
}

template <int k>
KOKKOS_INLINE_FUNCTION void expand(Box &a, KDOP<k> const &b)
{
  ArborX::Details::expand(a, (Box)b);
}

// NOTE intersects(predicate_geometry, bounding_volume)
template <int k>
KOKKOS_INLINE_FUNCTION bool intersects(Box const &a, KDOP<k> const &b)
{
  return b.intersects(a);
}

template <int k>
KOKKOS_INLINE_FUNCTION bool intersects(KDOP<k> const &a, Box const &b)
{
  return a.intersects(b);
}

template <int k>
KOKKOS_INLINE_FUNCTION bool intersects(Point const &p, KDOP<k> const &x)
{
  return x.intersects(p);
}

template <int k>
KOKKOS_INLINE_FUNCTION bool intersects(KDOP<k> const &a, KDOP<k> const &b)
{
  return a.intersects(b);
}

} // namespace Experimental
} // namespace ArborX

template <typename KDOP>
struct ArborX::Details::Dispatch::centroid<ArborX::GeometryTraits::KDOPTag,
                                           KDOP>
{
  KOKKOS_FUNCTION static auto apply(KDOP const &kdop)
  {
    // FIXME approximation
    using Box = ArborX::Box;
    return centroid<BoxTag, Box>::apply((Box)kdop);
  }
};

template <int k, typename Coordinate>
struct ArborX::GeometryTraits::dimension<
    ArborX::Experimental::KDOP<k, Coordinate>>
{
  static constexpr int value = 3;
};
template <int k, typename Coordinate>
struct ArborX::GeometryTraits::tag<ArborX::Experimental::KDOP<k, Coordinate>>
{
  using type = KDOPTag;
};
template <int k, typename Coordinate>
struct ArborX::GeometryTraits::coordinate_type<
    ArborX::Experimental::KDOP<k, Coordinate>>
{
  using type = Coordinate;
};

#endif
