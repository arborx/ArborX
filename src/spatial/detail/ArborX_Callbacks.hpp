/****************************************************************************
 * Copyright (c) 2017-2023 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_CALLBACKS_HPP
#define ARBORX_CALLBACKS_HPP

#include <detail/ArborX_AccessTraits.hpp>
#include <detail/ArborX_PairValueIndex.hpp>
#include <detail/ArborX_Predicates.hpp> // is_valid_predicate_tag

#include <Kokkos_DetectionIdiom.hpp>
#include <Kokkos_Macros.hpp>

#include <type_traits>

namespace ArborX
{

enum class CallbackTreeTraversalControl
{
  early_exit,
  normal_continuation
};

namespace Details
{

struct PostCallbackTag
{};

struct DefaultCallback
{
  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const &, Value const &value,
                                  OutputFunctor const &out) const
  {
    out(value);
  }
};

struct LegacyDefaultCallback
{
  template <typename Query, typename Value, typename Index,
            typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Query const &,
                                  PairValueIndex<Value, Index> const &value,
                                  OutputFunctor const &output) const
  {
    // APIv1 callback has the signature operator()(Query, int)
    // As we store PairValueIndex with potentially non int index (like
    // unsigned), we explicitly cast it here.
    output((int)value.index);
  }
};

// archetypal alias for a 'tag' type member in user callbacks
template <typename Callback>
using CallbackTagArchetypeAlias = typename Callback::tag;

template <typename Callback>
struct is_tagged_post_callback
    : Kokkos::is_detected_exact<PostCallbackTag, CallbackTagArchetypeAlias,
                                Callback>::type
{};

// output functor to pass to the callback during detection
template <typename T>
struct Sink
{
  KOKKOS_FUNCTION void operator()(T const &) const {}
};

template <typename OutputView>
using OutputFunctorHelper = Sink<typename OutputView::value_type>;

template <class Callback>
void check_generic_lambda_support(Callback const &)
{
#ifdef __NVCC__
  // Without it would get a segmentation fault and no diagnostic whatsoever
  static_assert(
      !__nv_is_extended_host_device_lambda_closure_type(Callback),
      "__host__ __device__ extended lambdas cannot be generic lambdas");
#endif
}

template <typename Value, typename Callback, typename Predicates,
          typename OutputView>
void check_valid_callback(Callback const &callback, Predicates const &,
                          OutputView const &)
{
  check_generic_lambda_support(callback);

  using Predicate =
      typename AccessValues<Predicates, PredicatesTag>::value_type;
  using PredicateTag = typename Predicate::Tag;

  static_assert(!(std::is_same_v<PredicateTag, NearestPredicateTag> &&
                  std::is_invocable_v<Callback const &, Predicate, int, float,
                                      OutputFunctorHelper<OutputView>>),
                R"error(Callback signature has changed for nearest predicates.
See https://github.com/arborx/ArborX/pull/366 for more details.
Sorry!)error");

  static_assert(is_valid_predicate_tag<PredicateTag>::value &&
                    std::is_invocable_v<Callback const &, Predicate, Value,
                                        OutputFunctorHelper<OutputView>>,
                "Callback 'operator()' does not have the correct signature");

  static_assert(
      std::is_void_v<std::invoke_result_t<Callback const &, Predicate, Value,
                                          OutputFunctorHelper<OutputView>>>,
      "Callback 'operator()' return type must be void");
}

template <typename Callback, typename Predicate, typename Primitive>
KOKKOS_FUNCTION bool invoke_callback_and_check_early_exit(Callback &&callback,
                                                          Predicate &&predicate,
                                                          Primitive &&primitive)
{
  if constexpr (std::is_same_v<CallbackTreeTraversalControl,
                               std::invoke_result_t<Callback &&, Predicate &&,
                                                    Primitive &&>>)
  {
    // Invoke a callback that may return a hint to interrupt the tree traversal
    // and return true for early exit, or false for normal continuation.
    return ((Callback &&)callback)((Predicate &&)predicate,
                                   (Primitive &&)primitive) ==
           CallbackTreeTraversalControl::early_exit;
  }
  else
  {
    // Invoke a callback that does not return a hint.  Always return false to
    // signify that the tree traversal should continue normally.
    ((Callback &&)callback)((Predicate &&)predicate, (Primitive &&)primitive);
    return false;
  }
}

template <typename Value, typename Callback, typename Predicates>
void check_valid_callback(Callback const &callback, Predicates const &)
{
  check_generic_lambda_support(callback);

  using Predicate =
      typename AccessValues<Predicates, PredicatesTag>::value_type;
  using PredicateTag = typename Predicate::Tag;

  static_assert(is_valid_predicate_tag<PredicateTag>::value,
                "The predicate tag is not valid");

  static_assert(std::is_invocable_v<Callback const &, Predicate, Value>,
                "Callback 'operator()' does not have the correct signature");

  static_assert(
      !(std::is_same_v<PredicateTag, SpatialPredicateTag> ||
        std::is_same_v<PredicateTag, OrderedSpatialPredicateTag>) ||
          (std::is_same_v<
               CallbackTreeTraversalControl,
               std::invoke_result_t<Callback const &, Predicate, Value>> ||
           std::is_void_v<
               std::invoke_result_t<Callback const &, Predicate, Value>>),
      "Callback 'operator()' return type must be void or "
      "ArborX::CallbackTreeTraversalControl");

  static_assert(
      !std::is_same_v<PredicateTag, NearestPredicateTag> ||
          std::is_void_v<
              std::invoke_result_t<Callback const &, Predicate, Value>>,
      "Callback 'operator()' return type must be void");
}

} // namespace Details
} // namespace ArborX

#endif
