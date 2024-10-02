/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_PREDICATE_HELPERS_HPP
#define ARBORX_PREDICATE_HELPERS_HPP

#include <ArborX_GeometryTraits.hpp>
#include <ArborX_Sphere.hpp>
#include <detail/ArborX_AccessTraits.hpp>
#include <detail/ArborX_Predicates.hpp>

namespace ArborX
{
namespace Experimental
{

template <typename UserPrimitives>
class PrimitivesIntersect
{
  using Primitives = Details::AccessValues<UserPrimitives, PrimitivesTag>;
  // FIXME:
  // using Geometry = typename Primitives::value_type;
  // static_assert(GeometryTraits::is_valid_geometry<Geometry>{});

public:
  Primitives _primitives;
};

template <typename UserPrimitives>
class PrimitivesOrderedIntersect
{
  using Primitives = Details::AccessValues<UserPrimitives, PrimitivesTag>;

public:
  Primitives _primitives;
};

template <typename UserPrimitives>
class PrimitivesWithRadius
{
  using Primitives = Details::AccessValues<UserPrimitives, PrimitivesTag>;
  using Point = typename Primitives::value_type;
  static_assert(GeometryTraits::is_point<Point>::value);
  using Coordinate = typename GeometryTraits::coordinate_type<Point>::type;

public:
  Primitives _primitives;
  Coordinate _r;

  PrimitivesWithRadius(UserPrimitives const &user_primitives, Coordinate r)
      : _primitives(user_primitives)
      , _r(r)
  {}
};

template <class UserPrimitives>
class PrimitivesNearestK
{
  using Primitives = Details::AccessValues<UserPrimitives, PrimitivesTag>;
  // FIXME:
  // using Geometry = typename Primitives::value_type;
  // static_assert(GeometryTraits::is_valid_geometry<Geometry>{});

public:
  Primitives _primitives;
  int _k;
};

template <typename Primitives>
auto make_intersects(Primitives const &primitives)
{
  Details::check_valid_access_traits(PrimitivesTag{}, primitives,
                                     Details::DoNotCheckGetReturnType());
  return PrimitivesIntersect<Primitives>{primitives};
}

template <typename Primitives, typename Coordinate>
auto make_intersects(Primitives const &primitives, Coordinate r)
{
  Details::check_valid_access_traits(PrimitivesTag{}, primitives);
  return PrimitivesWithRadius<Primitives>(primitives, r);
}

template <typename Primitives>
auto make_ordered_intersects(Primitives const &primitives)
{
  Details::check_valid_access_traits(PrimitivesTag{}, primitives,
                                     Details::DoNotCheckGetReturnType());
  return PrimitivesOrderedIntersect<Primitives>{primitives};
}

template <typename Primitives>
auto make_nearest(Primitives const &primitives, int k)
{
  Details::check_valid_access_traits(PrimitivesTag{}, primitives,
                                     Details::DoNotCheckGetReturnType());
  return PrimitivesNearestK<Primitives>{primitives, k};
}

} // namespace Experimental

template <class Primitives>
struct AccessTraits<Experimental::PrimitivesIntersect<Primitives>,
                    PredicatesTag>
{
private:
  using Self = Experimental::PrimitivesIntersect<Primitives>;

public:
  using memory_space = typename Primitives::memory_space;
  using size_type = typename memory_space::size_type;

  static KOKKOS_FUNCTION size_type size(Self const &x)
  {
    return x._primitives.size();
  }
  static KOKKOS_FUNCTION auto get(Self const &x, size_type i)
  {
    return intersects(x._primitives(i));
  }
};

template <class Primitives>
struct AccessTraits<Experimental::PrimitivesOrderedIntersect<Primitives>,
                    PredicatesTag>
{
private:
  using Self = Experimental::PrimitivesOrderedIntersect<Primitives>;

public:
  using memory_space = typename Primitives::memory_space;
  using size_type = typename memory_space::size_type;

  static KOKKOS_FUNCTION size_type size(Self const &x)
  {
    return x._primitives.size();
  }
  static KOKKOS_FUNCTION auto get(Self const &x, size_type i)
  {
    return ordered_intersects(x._primitives(i));
  }
};

template <class Primitives>
struct AccessTraits<Experimental::PrimitivesWithRadius<Primitives>,
                    PredicatesTag>
{
private:
  using Self = Experimental::PrimitivesWithRadius<Primitives>;

public:
  using memory_space = typename Primitives::memory_space;
  using size_type = typename memory_space::size_type;

  static KOKKOS_FUNCTION size_type size(Self const &x)
  {
    return x._primitives.size();
  }
  static KOKKOS_FUNCTION auto get(Self const &x, size_type i)
  {
    auto const &point = x._primitives(i);
    using Point = std::decay_t<decltype(point)>;
    constexpr int dim = GeometryTraits::dimension_v<Point>;
    using Coordinate = typename GeometryTraits::coordinate_type<Point>::type;
    return intersects(Sphere(
        Details::convert<::ArborX::Point<dim, Coordinate>>(point), x._r));
  }
};

template <class Primitives>
struct AccessTraits<Experimental::PrimitivesNearestK<Primitives>, PredicatesTag>
{
private:
  using Self = Experimental::PrimitivesNearestK<Primitives>;

public:
  using memory_space = typename Primitives::memory_space;
  using size_type = typename memory_space::size_type;

  static KOKKOS_FUNCTION size_type size(Self const &x)
  {
    return x._primitives.size();
  }
  static KOKKOS_FUNCTION auto get(Self const &x, size_type i)
  {
    return nearest(x._primitives(i), x._k);
  }
};

} // namespace ArborX

#endif
