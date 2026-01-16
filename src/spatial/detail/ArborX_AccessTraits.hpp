/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
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
#include <detail/ArborX_Predicates.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{

template <typename T, typename Enable = void>
struct AccessTraits
{
  using not_specialized = void; // tag to detect existence of a specialization
};

template <typename Traits>
using AccessTraitsNotSpecializedArchetypeAlias =
    typename Traits::not_specialized;

template <typename View>
struct AccessTraits<
    View, std::enable_if_t<Kokkos::is_view<View>{} && View::rank() == 1>>
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

template <typename View>
struct AccessTraits<
    View, std::enable_if_t<Kokkos::is_view<View>{} && View::rank() == 2>>
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

template <typename T>
struct AccessTraits<std::vector<T>>
{
  using memory_space = typename Kokkos::HostSpace;

  using Vector = std::vector<T>;

  static T &get(Vector const &v, int i) { return v[i]; }
  static typename Vector::size_type size(Vector const &v) { return v.size(); }
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

struct DoNotCheckReturnTypeTag
{};

struct CheckReturnTypeTag
{};

template <typename Values, typename Tag = DoNotCheckReturnTypeTag>
void check_valid_access_traits(Values const &, Tag = {})
{
  using Access = AccessTraits<Values>;
  static_assert(
      !Kokkos::is_detected<AccessTraitsNotSpecializedArchetypeAlias, Access>{},
      "Must specialize 'AccessTraits<Values>'");

  static_assert(
      Kokkos::is_detected<AccessTraitsMemorySpaceArchetypeAlias, Access>{},
      "AccessTraits<Values> must define 'memory_space' member type");
  static_assert(
      Kokkos::is_memory_space<
          Kokkos::detected_t<AccessTraitsMemorySpaceArchetypeAlias, Access>>{},
      "'memory_space' member type must be a valid Kokkos memory space");

  static_assert(
      Kokkos::is_detected<AccessTraitsSizeArchetypeExpression, Access,
                          Values>{},
      "AccessTraits<Values> must define 'size()' static member function");
  static_assert(
      std::is_integral<Kokkos::detected_t<AccessTraitsSizeArchetypeExpression,
                                          Access, Values>>{},
      "size() static member function return type is not an integral type");

  static_assert(
      Kokkos::is_detected<AccessTraitsGetArchetypeExpression, Access, Values>{},
      "AccessTraits<Values> must define 'get()' static member function");
  static_assert(
      !std::is_void_v<Kokkos::detected_t<AccessTraitsGetArchetypeExpression,
                                         Access, Values>>,
      "get() static member function return type must not be void");

  if constexpr (std::is_same_v<Tag, CheckReturnTypeTag>)
  {
    using Predicate = std::decay_t<
        Kokkos::detected_t<AccessTraitsGetArchetypeExpression, Access, Values>>;
    using PredicateTag =
        Kokkos::detected_t<PredicateTagArchetypeAlias, Predicate>;
    static_assert(is_valid_predicate_tag<PredicateTag>::value,
                  "Invalid tag for the predicates");
  }
}

template <typename Values>
class AccessValuesI
{
private:
  using Access = AccessTraits<Values>;
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
  std::size_t size() const { return Access::size(_values); }

  using self_type = AccessValuesI<Values>;
};

template <typename D, typename... P>
class AccessValuesI<Kokkos::View<D, P...>> : public Kokkos::View<D, P...>
{
public:
  using self_type = Kokkos::View<D, P...>;
};

template <typename Values>
class AccessValuesI<AccessValuesI<Values>> : public AccessValuesI<Values>
{};

template <typename Values>
using AccessValues = typename AccessValuesI<Values>::self_type;

} // namespace Details

template <typename Values>
struct AccessTraits<Details::AccessValuesI<Values>>
{
  using AccessValues = Details::AccessValuesI<Values>;

  using memory_space = typename AccessValues::memory_space;

  KOKKOS_FUNCTION static decltype(auto) get(AccessValues const &w, int i)
  {
    return w(i);
  }

  KOKKOS_FUNCTION
  static auto size(AccessValues const &w) { return w.size(); }
};

} // namespace ArborX

#endif
