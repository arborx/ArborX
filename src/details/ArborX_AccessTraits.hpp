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

#ifndef ARBORX_ACCESS_TRAITS_HPP
#define ARBORX_ACCESS_TRAITS_HPP

#include <ArborX_GeometryTraits.hpp>
#include <ArborX_Point.hpp>
#include <ArborX_Predicates.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{

struct PrimitivesTag
{};

struct PredicatesTag
{};

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
  template <std::size_t... Is>
  KOKKOS_FUNCTION static auto getPoint(std::index_sequence<Is...>,
                                       View const &v, int i)
  {
    return Point<sizeof...(Is)>{v(i, Is)...};
  }

  // Returns by value
  KOKKOS_FUNCTION static auto get(View const &v, int i)
  {
    constexpr int dim = View::static_extent(1);
    if constexpr (dim > 0) // dimension known at compile time
      return getPoint(std::make_index_sequence<dim>(), v, i);
    else
      return Point{v(i, 0), v(i, 1), v(i, 2)};
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
template <typename Traits, typename X>
using AccessTraitsSizeArchetypeExpression =
    decltype(Traits::size(std::declval<X const &>()));

// archetypal expression for 'get()' static member function in access traits
template <typename Traits, typename X>
using AccessTraitsGetArchetypeExpression =
    decltype(Traits::get(std::declval<X const &>(), 0));

template <typename P>
using PredicateTagArchetypeAlias = typename P::Tag;

template <typename Predicates>
void check_valid_access_traits(PredicatesTag, Predicates const &)
{
  using Access = AccessTraits<Predicates, PredicatesTag>;
  static_assert(
      !Kokkos::is_detected<AccessTraitsNotSpecializedArchetypeAlias, Access>{},
      "Must specialize 'AccessTraits<Predicates,PredicatesTag>'");

  static_assert(
      Kokkos::is_detected<AccessTraitsMemorySpaceArchetypeAlias, Access>{},
      "AccessTraits<Predicates,PredicatesTag> must define 'memory_space' "
      "member type");
  static_assert(
      Kokkos::is_memory_space<
          Kokkos::detected_t<AccessTraitsMemorySpaceArchetypeAlias, Access>>{},
      "'memory_space' member type must be a valid Kokkos memory space");

  static_assert(
      Kokkos::is_detected<AccessTraitsSizeArchetypeExpression, Access,
                          Predicates>{},
      "AccessTraits<Predicates,PredicatesTag> must define 'size()' static "
      "member function");
  static_assert(
      std::is_integral<Kokkos::detected_t<AccessTraitsSizeArchetypeExpression,
                                          Access, Predicates>>{},
      "size() static member function return type is not an integral type");

  static_assert(
      Kokkos::is_detected<AccessTraitsGetArchetypeExpression, Access,
                          Predicates>{},
      "AccessTraits<Predicates,PredicatesTag> must define 'get()' static "
      "member function");

  using Predicate =
      std::decay_t<Kokkos::detected_t<AccessTraitsGetArchetypeExpression,
                                      Access, Predicates>>;
  using Tag = Kokkos::detected_t<PredicateTagArchetypeAlias, Predicate>;
  static_assert(is_valid_predicate_tag<Tag>::value,
                "Invalid tag for the predicates");
}

struct DoNotCheckGetReturnType : std::false_type
{};

template <typename Primitives, typename CheckGetReturnType = std::true_type>
void check_valid_access_traits(PrimitivesTag, Primitives const &,
                               CheckGetReturnType = {})
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  static_assert(
      !Kokkos::is_detected<AccessTraitsNotSpecializedArchetypeAlias, Access>{},
      "Must specialize 'AccessTraits<Primitives,PrimitivesTag>'");

  static_assert(
      Kokkos::is_detected<AccessTraitsMemorySpaceArchetypeAlias, Access>{},
      "AccessTraits<Primitives,PrimitivesTag> must define 'memory_space' "
      "member type");
  static_assert(
      Kokkos::is_memory_space<
          Kokkos::detected_t<AccessTraitsMemorySpaceArchetypeAlias, Access>>{},
      "'memory_space' member type must be a valid Kokkos memory space");

  static_assert(
      Kokkos::is_detected<AccessTraitsSizeArchetypeExpression, Access,
                          Primitives>{},
      "AccessTraits<Primitives,PrimitivesTag> must define 'size()' static "
      "member function");
  static_assert(
      std::is_integral<Kokkos::detected_t<AccessTraitsSizeArchetypeExpression,
                                          Access, Primitives>>{},
      "size() static member function return type is not an integral type");

  static_assert(
      Kokkos::is_detected<AccessTraitsGetArchetypeExpression, Access,
                          Primitives>{},
      "AccessTraits<Primitives,PrimitivesTag> must define 'get()' static "
      "member function");
  using T = std::decay_t<Kokkos::detected_t<AccessTraitsGetArchetypeExpression,
                                            Access, Primitives>>;
  if constexpr (CheckGetReturnType())
  {
    static_assert(GeometryTraits::is_point_v<T> || GeometryTraits::is_box_v<T>,
                  "AccessTraits<Primitives,PrimitivesTag>::get() return type "
                  "must decay to a point or a box type");
  }
}

template <typename Values, typename Tag>
class AccessValuesI
{
private:
  using Access = AccessTraits<Values, Tag>;
  Values _values;

public:
  explicit AccessValuesI(Values values)
      : _values(std::move(values))
  {}
  using memory_space = typename Access::memory_space;
  using value_type = std::decay_t<
      Kokkos::detected_t<AccessTraitsGetArchetypeExpression, Access, Values>>;

  KOKKOS_FUNCTION
  decltype(auto) operator()(int i) const { return Access::get(_values, i); }

  KOKKOS_FUNCTION
  auto size() const { return Access::size(_values); }

  using self_type = AccessValuesI<Values, Tag>;
};

template <typename D, typename... P, typename Tag>
class AccessValuesI<Kokkos::View<D, P...>, Tag> : public Kokkos::View<D, P...>
{
public:
  using self_type = Kokkos::View<D, P...>;
};

template <typename Values, typename Tag1, typename Tag2>
class AccessValuesI<AccessValuesI<Values, Tag1>, Tag2>
    : public AccessValuesI<Values, Tag1>
{
  static_assert(std::is_same_v<Tag1, Tag2>);
};

template <typename Values, typename Tag>
using AccessValues = typename AccessValuesI<Values, Tag>::self_type;

} // namespace Details

template <typename Values, typename Tag>
struct AccessTraits<Details::AccessValuesI<Values, Tag>, Tag>
{
  using AccessValues = Details::AccessValuesI<Values, Tag>;

  using memory_space = typename AccessValues::memory_space;

  KOKKOS_FUNCTION static decltype(auto) get(AccessValues const &w, int i)
  {
    return w(i);
  }

  KOKKOS_FUNCTION
  static decltype(auto) size(AccessValues const &w) { return w.size(); }
};

} // namespace ArborX

#endif
