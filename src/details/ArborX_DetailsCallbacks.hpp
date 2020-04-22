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
#ifndef ARBORX_DETAILS_CALLBACKS_HPP
#define ARBORX_DETAILS_CALLBACKS_HPP

#include <ArborX_DetailsConcepts.hpp>
#include <ArborX_Traits.hpp>

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
  template <typename Query, typename Insert>
  KOKKOS_FUNCTION void operator()(Query const &, int index,
                                  Insert const &insert) const
  {
    insert(index);
  }
};

struct CallbackDefaultNearestPredicate
{
  using tag = InlineCallbackTag;
  template <typename Query, typename Insert>
  KOKKOS_FUNCTION void operator()(Query const &, int index, float,
                                  Insert const &insert) const
  {
    insert(index);
  }
};

struct CallbackDefaultNearestPredicateWithDistance
{
  using tag = InlineCallbackTag;
  template <typename Query, typename Insert>
  KOKKOS_FUNCTION void operator()(Query const &, int index, float distance,
                                  Insert const &insert) const
  {
    insert({index, distance});
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

// output functor to pass to the callback during detection
template <typename T>
struct Sink
{
  void operator()(T const &) const {}
};

template <typename Predicates>
using PredicatesHelper =
    decay_result_of_get_t<Traits::Access<Predicates, Traits::PredicatesTag>>;

template <typename OutputView>
using OutputFunctorHelper = Sink<typename OutputView::value_type>;

template <typename Callback, typename Predicates, typename OutputView>
void check_valid_callback(Callback const &, Predicates const &,
                          OutputView const &)
{
}

} // namespace Details
} // namespace ArborX

#endif
