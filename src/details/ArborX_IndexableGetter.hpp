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

#ifndef ARBORX_INDEXABLE_GETTER_HPP
#define ARBORX_INDEXABLE_GETTER_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsNode.hpp> // PairIndexVolume
#include <ArborX_GeometryTraits.hpp>

namespace ArborX::Details
{

struct DefaultIndexableGetter
{
  KOKKOS_DEFAULTED_FUNCTION
  DefaultIndexableGetter() = default;

  template <typename Geometry,
            std::enable_if_t<GeometryTraits::is_point<Geometry>{} ||
                             GeometryTraits::is_box<Geometry>{}>>
  KOKKOS_FUNCTION auto const &operator()(Geometry const &geometry) const
  {
    return geometry;
  }

  template <typename Geometry>
  KOKKOS_FUNCTION Geometry const &
  operator()(PairIndexVolume<Geometry> const &pair) const
  {
    return pair.bounding_volume;
  }
};

template <typename Primitives, typename BoundingVolume>
class LegacyValues
{
  Primitives _primitives;
  using Access = AccessTraits<Primitives, PrimitivesTag>;

public:
  using memory_space = typename Access::memory_space;
  using value_type = Details::PairIndexVolume<BoundingVolume>;
  using size_type =
      Kokkos::detected_t<Details::AccessTraitsSizeArchetypeExpression, Access,
                         Primitives>;

  LegacyValues(Primitives const &primitives)
      : _primitives(primitives)
  {}

  KOKKOS_FUNCTION
  decltype(auto) operator()(size_type i) const
  {
    if constexpr (std::is_same_v<BoundingVolume,
                                 typename AccessTraitsHelper<Access>::type>)
    {
      return value_type{(unsigned)i, Access::get(_primitives, i)};
    }
    else
    {
      BoundingVolume bounding_volume{};
      expand(bounding_volume, Access::get(_primitives, i));
      return value_type{(unsigned)i, bounding_volume};
    }
  }

  KOKKOS_FUNCTION
  size_type size() const { return Access::size(_primitives); }
};

template <typename Primitives>
struct Indexables
{
  Primitives _primitives;
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  using memory_space = typename Access::memory_space;

  KOKKOS_FUNCTION decltype(auto) operator()(int i) const
  {
    return Access::get(_primitives, i);
  }

  KOKKOS_FUNCTION auto size() const { return Access::size(_primitives); }
};

} // namespace ArborX::Details

#endif
