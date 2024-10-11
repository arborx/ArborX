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
#ifndef ARBORX_PREDICATE_HPP
#define ARBORX_PREDICATE_HPP

#include <algorithms/ArborX_Distance.hpp>
#include <algorithms/ArborX_Intersects.hpp>

#include <Kokkos_Macros.hpp>

#include <type_traits>

namespace ArborX
{
namespace Details
{
struct NearestPredicateTag
{};
struct SpatialPredicateTag
{};
struct OrderedSpatialPredicateTag
{};

// nvcc has problems with using std::interal_constant here.
template <typename PredicateTag>
struct is_valid_predicate_tag
{
  static constexpr bool value =
      std::is_same<PredicateTag, SpatialPredicateTag>{} ||
      std::is_same<PredicateTag, NearestPredicateTag>{} ||
      std::is_same<PredicateTag, OrderedSpatialPredicateTag>{};
};
} // namespace Details

template <typename Geometry>
struct Nearest
{
  using Tag = Details::NearestPredicateTag;

  KOKKOS_DEFAULTED_FUNCTION
  Nearest() = default;

  KOKKOS_FUNCTION
  Nearest(Geometry const &geometry, int k)
      : _geometry(geometry)
      , _k(k)
  {}

  template <class OtherGeometry>
  KOKKOS_FUNCTION auto distance(OtherGeometry const &other) const
  {
    using Details::distance;
    return distance(_geometry, other);
  }

  Geometry _geometry;
  int _k = 0;
};

template <typename Geometry>
struct Intersects
{
  using Tag = Details::SpatialPredicateTag;

  KOKKOS_DEFAULTED_FUNCTION Intersects() = default;

  KOKKOS_FUNCTION Intersects(Geometry const &geometry)
      : _geometry(geometry)
  {}

  template <typename OtherGeometry>
  KOKKOS_FUNCTION bool operator()(OtherGeometry const &other) const
  {
    using Details::intersects;
    return intersects(_geometry, other);
  }

  Geometry _geometry;
};

namespace Experimental
{
template <typename Geometry>
struct OrderedSpatial
{
  using Tag = Details::OrderedSpatialPredicateTag;

  KOKKOS_DEFAULTED_FUNCTION
  OrderedSpatial() = default;

  KOKKOS_FUNCTION
  OrderedSpatial(Geometry const &geometry)
      : _geometry(geometry)
  {}

  template <class OtherGeometry>
  KOKKOS_FUNCTION auto distance(OtherGeometry const &other) const
  {
    using Details::distance;
    return distance(_geometry, other);
  }

  Geometry _geometry;
};
} // namespace Experimental

template <typename Geometry>
KOKKOS_INLINE_FUNCTION Nearest<Geometry> nearest(Geometry const &geometry,
                                                 int k = 1)
{
  return Nearest<Geometry>(geometry, k);
}

template <typename Geometry>
KOKKOS_INLINE_FUNCTION Intersects<Geometry> intersects(Geometry const &geometry)
{
  return Intersects<Geometry>(geometry);
}

template <typename Geometry>
KOKKOS_INLINE_FUNCTION int getK(Nearest<Geometry> const &pred)
{
  return pred._k;
}

namespace Experimental
{
template <typename Geometry>
KOKKOS_INLINE_FUNCTION OrderedSpatial<Geometry>
ordered_intersects(Geometry const &geometry)
{
  return OrderedSpatial<Geometry>(geometry);
}
} // namespace Experimental

template <typename Geometry>
KOKKOS_INLINE_FUNCTION Geometry const &
getGeometry(Nearest<Geometry> const &pred)
{
  return pred._geometry;
}
template <typename Geometry>
KOKKOS_INLINE_FUNCTION Geometry getGeometry(Nearest<Geometry> &&pred)
{
  return pred._geometry;
}

template <typename Geometry>
KOKKOS_INLINE_FUNCTION Geometry const &
getGeometry(Intersects<Geometry> const &pred)
{
  return pred._geometry;
}
template <typename Geometry>
KOKKOS_INLINE_FUNCTION Geometry getGeometry(Intersects<Geometry> &&pred)
{
  return pred._geometry;
}

template <typename Geometry>
KOKKOS_INLINE_FUNCTION Geometry const &
getGeometry(Experimental::OrderedSpatial<Geometry> const &pred)
{
  return pred._geometry;
}
template <typename Geometry>
KOKKOS_INLINE_FUNCTION Geometry
getGeometry(Experimental::OrderedSpatial<Geometry> &&pred)
{
  return pred._geometry;
}

template <typename Predicate, typename Data>
struct PredicateWithAttachment : Predicate
{
  KOKKOS_DEFAULTED_FUNCTION PredicateWithAttachment() = default;
  KOKKOS_INLINE_FUNCTION PredicateWithAttachment(Predicate const &pred,
                                                 Data const &data)
      : Predicate{pred}
      , _data{data}
  {}
  KOKKOS_INLINE_FUNCTION PredicateWithAttachment(Predicate &&pred, Data &&data)
      : Predicate(std::forward<Predicate>(pred))
      , _data(std::forward<Data>(data))
  {}
  Data _data;
};

template <typename Predicate, typename Data>
KOKKOS_INLINE_FUNCTION Data const &
getData(PredicateWithAttachment<Predicate, Data> const &pred) noexcept
{
  return pred._data;
}

template <typename Predicate, typename Data>
KOKKOS_INLINE_FUNCTION Predicate const &
getPredicate(PredicateWithAttachment<Predicate, Data> const &pred) noexcept
{
  return static_cast<Predicate const &>(pred); // slicing
}

template <typename Predicate, typename Data>
KOKKOS_INLINE_FUNCTION constexpr auto attach(Predicate &&pred, Data &&data)
{
  return PredicateWithAttachment<std::decay_t<Predicate>, std::decay_t<Data>>{
      std::forward<Predicate>(pred), std::forward<Data>(data)};
}

} // namespace ArborX

#endif
