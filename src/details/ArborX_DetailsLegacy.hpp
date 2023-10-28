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

#ifndef ARBORX_DETAILS_LEGACY_HPP
#define ARBORX_DETAILS_LEGACY_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsNode.hpp>

namespace ArborX::Details
{

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
  auto operator()(size_type i) const
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

template <typename Callback>
struct LegacyCallbackWrapper
{
  Callback _callback;

  template <typename Predicate, typename Geometry>
  KOKKOS_FUNCTION auto operator()(Predicate const &predicate,
                                  PairIndexVolume<Geometry> const &value) const
  {
    return _callback(predicate, value.index);
  }

  template <typename Predicate, typename Geometry, typename Output>
  KOKKOS_FUNCTION auto operator()(Predicate const &predicate,
                                  PairIndexVolume<Geometry> const &value,
                                  Output const &out) const
  {
    return _callback(predicate, value.index, out);
  }
};

struct LegacyDefaultCallback
{
  template <typename Query, typename Geometry, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Query const &,
                                  PairIndexVolume<Geometry> const &value,
                                  OutputFunctor const &output) const
  {
    output(value.index);
  }
};

struct LegacyDefaultTemplateValue
{};

} // namespace ArborX::Details

template <>
struct ArborX::GeometryTraits::dimension<
    ArborX::Details::LegacyDefaultTemplateValue>
{
  static constexpr int value = 3;
};
template <>
struct ArborX::GeometryTraits::tag<ArborX::Details::LegacyDefaultTemplateValue>
{
  using type = BoxTag;
};
template <>
struct ArborX::GeometryTraits::coordinate_type<
    ArborX::Details::LegacyDefaultTemplateValue>
{
  using type = float;
};

template <typename Primitives, typename BoundingVolume>
struct ArborX::AccessTraits<
    ArborX::Details::LegacyValues<Primitives, BoundingVolume>,
    ArborX::PrimitivesTag>
{
  using Values = ArborX::Details::LegacyValues<Primitives, BoundingVolume>;

  using memory_space = typename Values::memory_space;
  using size_type = typename Values::size_type;
  using value_type = typename Values::value_type;

  KOKKOS_FUNCTION static size_type size(Values const &values)
  {
    return values.size();
  }
  KOKKOS_FUNCTION static decltype(auto) get(Values const &values, size_type i)
  {
    return values(i);
  }
};

#endif
