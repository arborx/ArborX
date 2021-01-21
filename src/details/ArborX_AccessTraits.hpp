/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_ACCESS_TRAITS_HPP
#define ARBORX_ACCESS_TRAITS_HPP

#include <ArborX_DetailsConcepts.hpp>
#include <ArborX_DetailsTags.hpp>
#include <ArborX_Point.hpp>
#include <ArborX_Predicates.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{

struct PrimitivesTag
{
};

struct PredicatesTag
{
};

template <typename T, typename Tag, typename Enable = void>
struct AccessTraits
{
  using not_specialized = void; // tag to detect existence of a specialization
};

template <typename Traits>
using AccessTraitsNotSpecializedArchetypeAlias =
    typename Traits::not_specialized;

template <typename View, typename Tag>
struct AccessTraits<
    View, Tag, std::enable_if_t<Kokkos::is_view<View>{} && View::rank == 1>>
{
  // Returns a const reference
  KOKKOS_FUNCTION static typename View::const_value_type &get(View const &v,
                                                              int i)
  {
    return v(i);
  }

  KOKKOS_FUNCTION
  static typename View::size_type size(View const &v) { return v.extent(0); }

  using memory_space = typename View::memory_space;
};

template <typename View, typename Tag>
struct AccessTraits<
    View, Tag, std::enable_if_t<Kokkos::is_view<View>{} && View::rank == 2>>
{
  // Returns by value
  KOKKOS_FUNCTION static Point get(View const &v, int i)
  {
    return {{v(i, 0), v(i, 1), v(i, 2)}};
  }

  KOKKOS_FUNCTION
  static typename View::size_type size(View const &v) { return v.extent(0); }

  using memory_space = typename View::memory_space;
};

namespace Details
{

// archetypal alias for a 'memory_space' type member in access traits
template <typename Traits>
using AccessTraitsMemorySpaceArchetypeAlias = typename Traits::memory_space;

// archetypal expression for 'size()' static member function in access traits
template <typename Traits>
using AccessTraitsSizeArchetypeExpression = decltype(
    Traits::size(std::declval<first_template_parameter_t<Traits> const &>()));

// archetypal expression for 'get()' static member function in access traits
template <typename Traits>
using AccessTraitsGetArchetypeExpression = decltype(
    Traits::get(std::declval<first_template_parameter_t<Traits> const &>(), 0));

template <typename Access>
struct AccessTraitsHelper
{
  // Deduce return type of get()
  using type =
      std::decay_t<detected_t<AccessTraitsGetArchetypeExpression, Access>>;
  using tag = typename Tag<type>::type;
};

template <typename Predicates>
void check_valid_access_traits(PredicatesTag, Predicates const &)
{
  using Access = AccessTraits<Predicates, PredicatesTag>;
  static_assert(
      !is_detected<AccessTraitsNotSpecializedArchetypeAlias, Access>{},
      "Must specialize 'AccessTraits<Predicates,PredicatesTag>'");

  static_assert(is_detected<AccessTraitsMemorySpaceArchetypeAlias, Access>{},
                "AccessTraits<Predicates,PredicatesTag> must define "
                "'memory_space' member type");
  static_assert(
      Kokkos::is_memory_space<
          detected_t<AccessTraitsMemorySpaceArchetypeAlias, Access>>{},
      "'memory_space' member type must be a valid Kokkos memory space");

  static_assert(is_detected<AccessTraitsSizeArchetypeExpression, Access>{},
                "AccessTraits<Predicates,PredicatesTag> must define "
                "'size()' static member function");
  static_assert(
      std::is_integral<
          detected_t<AccessTraitsSizeArchetypeExpression, Access>>{},
      "size() static member function return type is not an integral type");

  static_assert(is_detected<AccessTraitsGetArchetypeExpression, Access>{},
                "AccessTraits<Predicates,PredicatesTag> must define "
                "'get()' static member function");

  using Tag = typename AccessTraitsHelper<Access>::tag;
  static_assert(std::is_same<Tag, NearestPredicateTag>{} ||
                    std::is_same<Tag, SpatialPredicateTag>{},
                "Invalid tag for the predicates");
}

template <typename Primitives>
void check_valid_access_traits(PrimitivesTag, Primitives const &)
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  static_assert(
      !is_detected<AccessTraitsNotSpecializedArchetypeAlias, Access>{},
      "Must specialize 'AccessTraits<Primitives,PrimitivesTag>'");

  static_assert(is_detected<AccessTraitsMemorySpaceArchetypeAlias, Access>{},
                "AccessTraits<Primitives,PrimitivesTag> must define "
                "'memory_space' member type");
  static_assert(
      Kokkos::is_memory_space<
          detected_t<AccessTraitsMemorySpaceArchetypeAlias, Access>>{},
      "'memory_space' member type must be a valid Kokkos memory space");

  static_assert(is_detected<AccessTraitsSizeArchetypeExpression, Access>{},
                "AccessTraits<Primitives,PrimitivesTag> must define "
                "'size()' static member function");
  static_assert(
      std::is_integral<
          detected_t<AccessTraitsSizeArchetypeExpression, Access>>{},
      "size() static member function return type is not an integral type");

  static_assert(is_detected<AccessTraitsGetArchetypeExpression, Access>{},
                "AccessTraits<Primitives,PrimitivesTag> must define "
                "'get()' static member function");
  using T =
      std::decay_t<detected_t<AccessTraitsGetArchetypeExpression, Access>>;
  static_assert(std::is_same<T, Point>{} || std::is_same<T, Box>{},
                "AccessTraits<Primitives,PrimitivesTag>::get() return type "
                "must decay to Point or to Box");
}

} // namespace Details

namespace Traits
{
using ::ArborX::PredicatesTag;
using ::ArborX::PrimitivesTag;
template <typename T, typename Tag, typename Enable = void>
struct Access
{
  using not_specialized = void;
};
} // namespace Traits
template <typename T, typename Tag>
struct AccessTraits<
    T, Tag,
    std::enable_if_t<!Details::is_detected<
        AccessTraitsNotSpecializedArchetypeAlias, Traits::Access<T, Tag>>{}>>
    : Traits::Access<T, Tag>
{
};
} // namespace ArborX

#endif
