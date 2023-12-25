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
#ifndef ARBORX_RANGE_TRAITS_HPP
#define ARBORX_RANGE_TRAITS_HPP

#include <ArborX_Predicates.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{

template <typename Values>
struct RangeTraitsArchetypeExpression
{
  using memory_space = typename Values::memory_space;
  using size_type = typename Values::size_type;

  KOKKOS_FUNCTION static auto get(Values const &values, size_type index);
  KOKKOS_FUNCTION static auto size(Values const &values);
};

template <typename Container, typename Enable = void>
struct RangeTraits
{
  using not_specialized = void;
};

template <typename Traits>
using RangeTraitsNotSpecializedArchetypeAlias =
    typename Traits::not_specialized;

// Specialization for 1D Kokkos View
template <typename View>
struct RangeTraits<View,
                   std::enable_if_t<Kokkos::is_view<View>{} && View::rank == 1>>
{
  using memory_space = typename View::memory_space;
  using size_type = typename View::size_type;

  KOKKOS_FUNCTION static typename View::const_value_type &get(View const &v,
                                                              size_type i)
  {
    return v(i);
  }
  KOKKOS_FUNCTION static auto size(View const &v) { return v.extent(0); }
};

namespace Details
{

// archetypal alias for a 'memory_space' type member in range traits
template <typename Traits>
using RangeTraitsMemorySpaceArchetypeAlias = typename Traits::memory_space;

// archetypal expression for 'size()' static member function in range traits
template <typename Traits, typename Values>
using RangeTraitsSizeArchetypeExpression =
    decltype(Traits::size(std::declval<Values const &>()));

// archetypal expression for 'get()' static member function in range traits
template <typename Traits, typename Values>
using RangeTraitsGetArchetypeExpression =
    decltype(Traits::get(std::declval<Values const &>(), 0));

template <typename Values>
void check_valid_range_traits(Values const &)
{
  using Range = RangeTraits<Values>;
  static_assert(
      !Kokkos::is_detected<RangeTraitsNotSpecializedArchetypeAlias, Range>{},
      "Must specialize 'RangeTraits<Values>'");

  static_assert(
      Kokkos::is_detected<RangeTraitsMemorySpaceArchetypeAlias, Range>{},
      "RangeTraits<Values> must define 'memory_space' member type");
  static_assert(
      Kokkos::is_memory_space<
          Kokkos::detected_t<RangeTraitsMemorySpaceArchetypeAlias, Range>>{},
      "'memory_space' member type must be a valid Kokkos memory space");

  static_assert(
      Kokkos::is_detected<RangeTraitsSizeArchetypeExpression, Range, Values>{},
      "RangeTraits<Values> must define 'size()' static member function");
  static_assert(
      std::is_integral<Kokkos::detected_t<RangeTraitsSizeArchetypeExpression,
                                          Range, Values>>{},
      "size() static member function return type is not an integral type");

  static_assert(
      Kokkos::is_detected<RangeTraitsGetArchetypeExpression, Range, Values>{},
      "RangeTraits<Values> must define 'get()' static member function");
}

template <typename Values>
class RangeValues
{
private:
  using Range = RangeTraits<Values>;

public:
  Values _values;

  using memory_space = typename Range::memory_space;
  using value_type = std::decay_t<
      Kokkos::detected_t<RangeTraitsGetArchetypeExpression, Range, Values>>;

  KOKKOS_FUNCTION
  decltype(auto) operator()(int i) const { return Range::get(_values, i); }

  KOKKOS_FUNCTION
  auto size() const { return Range::size(_values); }
};

} // namespace Details

template <typename Values>
class RangeTraits<Details::RangeValues<Values>>
{
private:
  using RangeValues = Details::RangeValues<Values>;

public:
  using memory_space = typename RangeValues::memory_space;

  KOKKOS_FUNCTION static decltype(auto) get(RangeValues const &values, int i)
  {
    return values(i);
  }

  KOKKOS_FUNCTION
  static auto size(RangeValues const &values) { return values.size(); }
};

} // namespace ArborX

#endif
