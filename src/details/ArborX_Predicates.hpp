/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
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

#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_DetailsNode.hpp>

namespace ArborX
{
namespace Details
{
struct NearestPredicateTag
{
};
struct SpatialPredicateTag
{
};
} // namespace Details

template <typename Geometry>
struct Nearest
{
  using Tag = Details::NearestPredicateTag;

  KOKKOS_INLINE_FUNCTION
  Nearest() = default;

  KOKKOS_INLINE_FUNCTION
  Nearest(Geometry const &geometry, int k)
      : _geometry(geometry)
      , _k(k)
  {
  }

  Geometry _geometry;
  int _k = 0;
};

template <typename Geometry>
struct Intersects
{
  using Tag = Details::SpatialPredicateTag;

  KOKKOS_DEFAULTED_FUNCTION Intersects() = default;

  KOKKOS_INLINE_FUNCTION Intersects(Geometry const &geometry)
      : _geometry(geometry)
  {
  }

  template <typename Other>
  KOKKOS_INLINE_FUNCTION bool operator()(Other const &other) const
  {
    return Details::intersects(_geometry, other);
  }

  Geometry _geometry;
};

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

template <typename Geometry>
KOKKOS_INLINE_FUNCTION Geometry const &
getGeometry(Nearest<Geometry> const &pred)
{
  return pred._geometry;
}

template <typename Geometry>
KOKKOS_INLINE_FUNCTION Geometry const &
getGeometry(Intersects<Geometry> const &pred)
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
  {
  }
  KOKKOS_INLINE_FUNCTION PredicateWithAttachment(Predicate &&pred, Data &&data)
      : Predicate(std::forward<Predicate>(pred))
      , _data(std::forward<Data>(data))
  {
  }
  Data _data;
};

template <typename Predicate, typename Data>
KOKKOS_INLINE_FUNCTION Data const &
getData(PredicateWithAttachment<Predicate, Data> const &pred)
{
  return pred._data;
}

template <typename Predicate, typename Data>
KOKKOS_INLINE_FUNCTION constexpr auto attach(Predicate &&pred, Data &&data)
{
  return PredicateWithAttachment<std::decay_t<Predicate>, std::decay_t<Data>>{
      std::forward<Predicate>(pred), std::forward<Data>(data)};
}

} // namespace ArborX

#endif
