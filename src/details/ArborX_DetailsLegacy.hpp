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
#include <ArborX_Callbacks.hpp>
#include <ArborX_PairValueIndex.hpp>
#include <ArborX_RangeTraits.hpp>

namespace ArborX::Details
{

template <typename Primitives, typename BoundingVolume>
class LegacyValues
{
  Primitives _primitives;
  using Access = AccessTraits<Primitives, PrimitivesTag>;

public:
  using memory_space = typename Access::memory_space;
  using index_type = unsigned;
  using value_type = PairValueIndex<BoundingVolume, index_type>;
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
      return value_type{Access::get(_primitives, i), (index_type)i};
    }
    else
    {
      BoundingVolume bounding_volume{};
      expand(bounding_volume, Access::get(_primitives, i));
      return value_type{bounding_volume, (index_type)i};
    }
  }

  KOKKOS_FUNCTION
  size_type size() const { return Access::size(_primitives); }
};

template <typename Callback>
struct LegacyCallbackWrapper
{
  Callback _callback;

  template <typename Predicate, typename Value>
  KOKKOS_FUNCTION auto operator()(Predicate const &predicate,
                                  PairValueIndex<Value> const &value) const
  {
    return _callback(predicate, value.index);
  }

  template <typename Predicate, typename Value, typename Output>
  KOKKOS_FUNCTION auto operator()(Predicate const &predicate,
                                  PairValueIndex<Value> const &value,
                                  Output const &out) const
  {
    return _callback(predicate, value.index, out);
  }
};

struct LegacyDefaultCallback
{
  template <typename Query, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Query const &,
                                  PairValueIndex<Value> const &value,
                                  OutputFunctor const &output) const
  {
    output(value.index);
  }
};

template <typename Value, typename Callback, typename Predicates,
          typename OutputView>
void check_valid_callback_accesstraits(Callback const &callback,
                                       Predicates const &, OutputView const &)
{
  check_generic_lambda_support(callback);

  using Access = AccessTraits<Predicates, PredicatesTag>;
  using PredicateTag = typename AccessTraitsHelper<Access>::tag;
  using Predicate = typename AccessTraitsHelper<Access>::type;

  static_assert(!(std::is_same<PredicateTag, NearestPredicateTag>{} &&
                  Kokkos::is_detected<
                      Legacy_NearestPredicateInlineCallbackArchetypeExpression,
                      Callback, Predicate, OutputFunctorHelper<OutputView>>{}),
                R"error(Callback signature has changed for nearest predicates.
See https://github.com/arborx/ArborX/pull/366 for more details.
Sorry!)error");

  static_assert(is_valid_predicate_tag<PredicateTag>::value &&
                    Kokkos::is_detected<InlineCallbackArchetypeExpression,
                                        Callback, Predicate, Value,
                                        OutputFunctorHelper<OutputView>>{},
                "Callback 'operator()' does not have the correct signature");

  static_assert(
      std::is_void<Kokkos::detected_t<InlineCallbackArchetypeExpression,
                                      Callback, Predicate, Value,
                                      OutputFunctorHelper<OutputView>>>{},
      "Callback 'operator()' return type must be void");
}

template <typename Value, typename Callback, typename Predicates>
void check_valid_callback_accesstraits(Callback const &callback,
                                       Predicates const &)
{
  check_generic_lambda_support(callback);

  using Access = AccessTraits<Predicates, PredicatesTag>;
  using PredicateTag = typename AccessTraitsHelper<Access>::tag;
  using Predicate = typename AccessTraitsHelper<Access>::type;

  static_assert(is_valid_predicate_tag<PredicateTag>::value,
                "The predicate tag is not valid");

  static_assert(Kokkos::is_detected<Experimental_CallbackArchetypeExpression,
                                    Callback, Predicate, Value>{},
                "Callback 'operator()' does not have the correct signature");

  static_assert(
      !(std::is_same<PredicateTag, SpatialPredicateTag>{} ||
        std::is_same<PredicateTag,
                     Experimental::OrderedSpatialPredicateTag>{}) ||
          (std::is_same<
               CallbackTreeTraversalControl,
               Kokkos::detected_t<Experimental_CallbackArchetypeExpression,
                                  Callback, Predicate, Value>>{} ||
           std::is_void<
               Kokkos::detected_t<Experimental_CallbackArchetypeExpression,
                                  Callback, Predicate, Value>>{}),
      "Callback 'operator()' return type must be void or "
      "ArborX::CallbackTreeTraversalControl");

  static_assert(
      !std::is_same<PredicateTag, NearestPredicateTag>{} ||
          std::is_void<
              Kokkos::detected_t<Experimental_CallbackArchetypeExpression,
                                 Callback, Predicate, Value>>{},
      "Callback 'operator()' return type must be void");
}

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
struct ArborX::RangeTraits<
    ArborX::Details::LegacyValues<Primitives, BoundingVolume>>
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

template <typename X, typename Tag>
class ArborX::RangeTraits<ArborX::Details::AccessValues<X, Tag>>
{
private:
  using Values = ArborX::Details::AccessValues<X, Tag>;

public:
  using memory_space = typename Values::memory_space;

  KOKKOS_FUNCTION static decltype(auto) get(Values const &values, int i)
  {
    return values(i);
  }

  KOKKOS_FUNCTION
  static auto size(Values const &values) { return values.size(); }
};

#endif
