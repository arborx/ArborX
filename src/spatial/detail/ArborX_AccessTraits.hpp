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

namespace Details
{

namespace Concepts
{

template <typename T>
concept AccessTraits = requires() {
  typename ArborX::AccessTraits<T>::memory_space;
  requires Kokkos::is_memory_space_v<
      typename ArborX::AccessTraits<T>::memory_space>;
} && requires(T const &v) {
  {
    AccessTraits<T>::size(v)
  } -> std::integral;
  // Cannot check return type of get() here as we need to test for non-void, but
  // there's no not_same_as concept, and !std::same_as<void> does not work
  AccessTraits<T>::get(v, 0);
} && !requires(T const &v) {
  {
    AccessTraits<T>::get(v, 0)
  } -> std::same_as<void>;
};

template <typename T>
concept HasTag = requires() { typename std::decay_t<T>::Tag; };

template <typename T>
concept Primitives = AccessTraits<T>;

template <typename T>
concept Predicates = AccessTraits<T> && requires(T const &v) {
  {
    ArborX::AccessTraits<T>::get(v, 0)
  } -> HasTag;
  requires Details::is_valid_predicate_tag<typename std::decay_t<
      decltype(ArborX::AccessTraits<T>::get(v, 0))>::Tag>::value;
};

} // namespace Concepts

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
  using value_type = std::decay_t<decltype(Access::get(_values, 0))>;

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
