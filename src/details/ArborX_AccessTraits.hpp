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

template <typename T, typename TTag>
using has_access_traits = typename is_complete<Traits::Access<T, TTag>>::type;

template <typename T, typename = void>
struct has_memory_space : std::false_type
{
};

template <typename T>
struct has_memory_space<
    T,
    std::enable_if_t<Kokkos::is_memory_space<typename T::memory_space>::value>>
    : std::true_type
{
};

template <typename Traits>
struct result_of_get
{
  using type = decltype(Traits::get(
      std::declval<first_template_parameter_t<Traits> const &>(), 0));
};

template <typename Traits>
using decay_result_of_get_t =
    std::decay_t<typename result_of_get<Traits>::type>;

template <typename Traits, typename = void>
struct has_get : std::false_type
{
};

template <typename Traits>
struct has_get<
    Traits,
    std::void_t<decltype(Traits::get(
        std::declval<first_template_parameter_t<Traits> const &>(), 0))>>
    : std::true_type
{
};

template <typename Traits, typename = void>
struct has_size : std::false_type
{
};

template <typename Traits>
struct has_size<
    Traits,
    std::enable_if_t<std::is_integral<decltype(Traits::size(
        std::declval<first_template_parameter_t<Traits> const &>()))>::value>>
    : std::true_type
{
};
} // namespace Details

namespace Traits
{
template <typename Access>
struct Helper
{
  using type = Details::decay_result_of_get_t<Access>;
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
  static_assert(
      has_memory_space<Access>{},
      "Traits::Access<Predicates,Traits::PredicatesTag> must define "
      "'memory_space' member type that is a valid Kokkos memory space");
  static_assert(has_size<Access>{},
                "Traits::Access<Predicates,Traits::PredicatesTag> must define "
                "'size()' member function");
  static_assert(
      has_get<Access>{},
      "Traits::Access<Predicates,Traits::PredicatesTag> must define 'get()' "
      "member function");
  using Tag = typename Tag<decay_result_of_get_t<Access>>::type;
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
  static_assert(
      has_memory_space<Access>{},
      "Traits::Access<Primitives,Traits::PrimitivesTag> must define "
      "'memory_space' member type that is a valid Kokkos memory space");
  static_assert(has_size<Access>{},
                "Traits::Access<Primitives,Traits::PrimitivesTag> must define "
                "'size()' member function");
  static_assert(has_get<Access>{},
                "Traits::Access<Primitives,Traits::PrimitivesTag> must define "
                "'get()' member function");
  static_assert(
      std::is_same<decay_result_of_get_t<Access>, Point>{} ||
          std::is_same<decay_result_of_get_t<Access>, Box>{},
      "Traits::Access<Primitives,Traits::PrimitivesTag>::get() return type "
      "must decay to Point or to Box");
}

} // namespace Details
} // namespace ArborX

#endif
