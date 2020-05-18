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
#ifndef ARBORX_ACCESS_TRAITS_HPP
#define ARBORX_ACCESS_TRAITS_HPP

#include <ArborX_DetailsConcepts.hpp> // is_complete
#include <ArborX_DetailsTags.hpp>
#include <ArborX_Point.hpp>
#include <ArborX_Predicates.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{
namespace Traits
{

struct PrimitivesTag
{
};

struct PredicatesTag
{
};

// Only a declaration so that existence of a specialization can be detected
template <typename T, typename Tag, typename Enable = void>
struct Access;

template <typename View, typename Tag>
struct Access<View, Tag,
              typename std::enable_if<Kokkos::is_view<View>::value &&
                                      View::rank == 1>::type>
{
  // Returns a const reference
  KOKKOS_FUNCTION static typename View::const_value_type &get(View const &v,
                                                              int i)
  {
    return v(i);
  }

  static typename View::size_type size(View const &v) { return v.extent(0); }

  using memory_space = typename View::memory_space;
};

template <typename View, typename Tag>
struct Access<View, Tag,
              typename std::enable_if<Kokkos::is_view<View>::value &&
                                      View::rank == 2>::type>
{
  // Returns by value
  KOKKOS_FUNCTION static Point get(View const &v, int i)
  {
    return {v(i, 0), v(i, 1), v(i, 2)};
  }

  static typename View::size_type size(View const &v) { return v.extent(0); }

  using memory_space = typename View::memory_space;
};

} // namespace Traits
} // namespace ArborX

namespace ArborX
{
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

} // namespace Details

namespace Traits
{
template <typename Access,
          typename = std::enable_if_t<Details::is_complete<Access>{}>>
struct Helper
{
  // Deduce return type of get()
  using type = std::decay_t<
      Details::detected_t<Details::AccessTraitsGetArchetypeExpression, Access>>;
  using tag = typename Details::Tag<type>::type;
};
} // namespace Traits

namespace Details
{

template <typename Predicates>
void check_valid_access_traits(Traits::PredicatesTag, Predicates const &)
{
  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
  static_assert(
      is_complete<Access>{},
      "Must specialize 'Traits::Access<Predicates,Traits::PredicatesTag>'");

  static_assert(is_detected<AccessTraitsMemorySpaceArchetypeAlias, Access>{},
                "Traits::Access<Predicates,Traits::PredicatesTag> must define "
                "'memory_space' member type");
  static_assert(
      Kokkos::is_memory_space<
          detected_t<AccessTraitsMemorySpaceArchetypeAlias, Access>>{},
      "'memory_space' member type must be a valid Kokkos memory space");

  static_assert(is_detected<AccessTraitsSizeArchetypeExpression, Access>{},
                "Traits::Access<Predicates,Traits::PredicatesTag> must define "
                "'size()' static member function");
  static_assert(
      std::is_integral<
          detected_t<AccessTraitsSizeArchetypeExpression, Access>>{},
      "size() static member function return type is not an integral type");

  static_assert(is_detected<AccessTraitsGetArchetypeExpression, Access>{},
                "Traits::Access<Predicates,Traits::PredicatesTag> must define "
                "'get()' static member function");

  using Tag = typename Traits::Helper<Access>::tag;
  static_assert(std::is_same<Tag, NearestPredicateTag>{} ||
                    std::is_same<Tag, SpatialPredicateTag>{},
                "Invalid tag for the predicates");
}

template <typename Primitives>
void check_valid_access_traits(Traits::PrimitivesTag, Primitives const &)
{
  using Access = Traits::Access<Primitives, Traits::PrimitivesTag>;
  static_assert(
      is_complete<Access>{},
      "Must specialize 'Traits::Access<Primitives,Traits::PrimitivesTag>'");

  static_assert(is_detected<AccessTraitsMemorySpaceArchetypeAlias, Access>{},
                "Traits::Access<Primitives,Traits::PrimitivesTag> must define "
                "'memory_space' member type");
  static_assert(
      Kokkos::is_memory_space<
          detected_t<AccessTraitsMemorySpaceArchetypeAlias, Access>>{},
      "'memory_space' member type must be a valid Kokkos memory space");

  static_assert(is_detected<AccessTraitsSizeArchetypeExpression, Access>{},
                "Traits::Access<Primitives,Traits::PrimitivesTag> must define "
                "'size()' static member function");
  static_assert(
      std::is_integral<
          detected_t<AccessTraitsSizeArchetypeExpression, Access>>{},
      "size() static member function return type is not an integral type");

  static_assert(is_detected<AccessTraitsGetArchetypeExpression, Access>{},
                "Traits::Access<Primitives,Traits::PrimitivesTag> must define "
                "'get()' static member function");
  using T =
      std::decay_t<detected_t<AccessTraitsGetArchetypeExpression, Access>>;
  static_assert(
      std::is_same<T, Point>{} || std::is_same<T, Box>{},
      "Traits::Access<Primitives,Traits::PrimitivesTag>::get() return type "
      "must decay to Point or to Box");
}

} // namespace Details
} // namespace ArborX

#endif
