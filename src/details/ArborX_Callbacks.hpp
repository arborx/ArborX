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
#ifndef ARBORX_CALLBACKS_HPP
#define ARBORX_CALLBACKS_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsConcepts.hpp>

#include <Kokkos_Macros.hpp>

#include <utility> // declval

namespace ArborX
{
namespace Details
{

struct InlineCallbackTag
{
};

struct PostCallbackTag
{
};

struct CallbackDefaultSpatialPredicate
{
  using tag = InlineCallbackTag;
  template <typename Query, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Query const &, int index,
                                  OutputFunctor const &output) const
  {
    output(index);
  }
};

struct CallbackDefaultNearestPredicate
{
  using tag = InlineCallbackTag;
  template <typename Query, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Query const &, int index, float,
                                  OutputFunctor const &output) const
  {
    output(index);
  }
};

struct CallbackDefaultNearestPredicateWithDistance
{
  using tag = InlineCallbackTag;
  template <typename Query, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Query const &, int index, float distance,
                                  OutputFunctor const &output) const
  {
    output({index, distance});
  }
};

// archetypal expression for user callbacks
template <typename Callback, typename Predicate, typename Out>
using NearestPredicateInlineCallbackArchetypeExpression =
    decltype(std::declval<Callback const &>()(
        std::declval<Predicate const &>(), 0, 0., std::declval<Out const &>()));

template <typename Callback, typename Predicate, typename Out>
using SpatialPredicateInlineCallbackArchetypeExpression =
    decltype(std::declval<Callback const &>()(std::declval<Predicate const &>(),
                                              0, std::declval<Out const &>()));

// archetypal alias for a 'tag' type member in user callbacks
template <typename Callback>
using CallbackTagArchetypeAlias = typename Callback::tag;

template <typename Callback>
struct is_tagged_post_callback
    : std::is_same<detected_t<CallbackTagArchetypeAlias, Callback>,
                   PostCallbackTag>::type
{
};

// output functor to pass to the callback during detection
template <typename T>
struct Sink
{
  void operator()(T const &) const {}
};

template <typename OutputView>
using OutputFunctorHelper = Sink<typename OutputView::value_type>;

template <typename Callback, typename Predicates, typename OutputView>
void check_valid_callback(Callback const &, Predicates const &,
                          OutputView const &)
{
#ifdef __NVCC__
  // Without it would get a segmentation fault and no diagnostic whatsoever
  static_assert(
      !__nv_is_extended_host_device_lambda_closure_type(Callback),
      "__host__ __device__ extended lambdas cannot be generic lambdas");
#endif

  using Access = AccessTraits<Predicates, PredicatesTag>;
  using PredicateTag = typename AccessTraitsHelper<Access>::tag;
  using Predicate = typename AccessTraitsHelper<Access>::type;

  static_assert(
      (std::is_same<PredicateTag, SpatialPredicateTag>{} &&
       is_detected<SpatialPredicateInlineCallbackArchetypeExpression, Callback,
                   Predicate, OutputFunctorHelper<OutputView>>{}) ||
          (std::is_same<PredicateTag, NearestPredicateTag>{} &&
           is_detected<NearestPredicateInlineCallbackArchetypeExpression,
                       Callback, Predicate, OutputFunctorHelper<OutputView>>{}),
      "Callback 'operator()' does not have the correct signature");
}

} // namespace Details
} // namespace ArborX

#endif
