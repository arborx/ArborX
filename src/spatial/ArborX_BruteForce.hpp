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

#ifndef ARBORX_BRUTE_FORCE_HPP
#define ARBORX_BRUTE_FORCE_HPP

#include <ArborX_Box.hpp>
#include <ArborX_CrsGraphWrapper.hpp>
#include <detail/ArborX_AccessTraits.hpp>
#include <detail/ArborX_AttachIndices.hpp>
#include <detail/ArborX_BruteForceImpl.hpp>
#include <detail/ArborX_Callbacks.hpp>
#include <detail/ArborX_CrsGraphWrapperImpl.hpp>
#include <detail/ArborX_IndexableGetter.hpp>
#include <detail/ArborX_PairValueIndex.hpp>
#include <detail/ArborX_PredicateHelpers.hpp>
#include <kokkos_ext/ArborX_KokkosExtAccessibilityTraits.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

namespace ArborX
{

template <typename MemorySpace, typename Value,
          typename IndexableGetter = Experimental::DefaultIndexableGetter,
          typename BoundingVolume = Box<
              GeometryTraits::dimension_v<
                  std::decay_t<std::invoke_result_t<IndexableGetter, Value>>>,
              typename GeometryTraits::coordinate_type_t<
                  std::decay_t<std::invoke_result_t<IndexableGetter, Value>>>>>
class BruteForce
{
public:
  using memory_space = MemorySpace;
  static_assert(Kokkos::is_memory_space<MemorySpace>::value);
  using size_type = typename MemorySpace::size_type;
  using bounding_volume_type = BoundingVolume;
  using value_type = Value;

  BruteForce() = default;

  template <typename ExecutionSpace, Details::Concepts::Primitives Values>
  BruteForce(ExecutionSpace const &space, Values const &values,
             IndexableGetter const &indexable_getter = IndexableGetter());

  KOKKOS_FUNCTION
  size_type size() const noexcept { return _size; }

  KOKKOS_FUNCTION
  bool empty() const noexcept { return size() == 0; }

  KOKKOS_FUNCTION
  bounding_volume_type bounds() const noexcept { return _bounds; }

  template <typename ExecutionSpace, Details::Concepts::Predicates Predicates,
            typename Callback, typename Ignore = int>
  void query(ExecutionSpace const &space, Predicates const &predicates,
             Callback const &callback, Ignore = Ignore()) const;

  template <typename ExecutionSpace,
            Details::Concepts::Predicates UserPredicates,
            typename CallbackOrView, typename View, typename... Args>
  std::enable_if_t<Kokkos::is_view_v<std::decay_t<View>>>
  query(ExecutionSpace const &space, UserPredicates const &user_predicates,
        CallbackOrView &&callback_or_view, View &&view, Args &&...args) const
  {
    Kokkos::Profiling::ScopedRegion guard("ArborX::BruteForce::query_crs");

    Details::CrsGraphWrapperImpl::
        check_valid_callback_if_first_argument_is_not_a_view<value_type>(
            callback_or_view, user_predicates, view);

    using Predicates = Details::AccessValues<UserPredicates>;
    using Tag = typename Predicates::value_type::Tag;

    Details::CrsGraphWrapperImpl::queryDispatch(
        Tag{}, *this, space, Predicates{user_predicates},
        std::forward<CallbackOrView>(callback_or_view),
        std::forward<View>(view), std::forward<Args>(args)...);
  }

  KOKKOS_FUNCTION auto const &indexable_get() const
  {
    return _indexable_getter;
  }

private:
  size_type _size{0};
  bounding_volume_type _bounds;
  Kokkos::View<value_type *, memory_space> _values;
  IndexableGetter _indexable_getter;
};

template <typename ExecutionSpace, Details::Concepts::Primitives Values>
KOKKOS_DEDUCTION_GUIDE BruteForce(ExecutionSpace, Values)
    -> BruteForce<typename Details::AccessValues<Values>::memory_space,
                  typename Details::AccessValues<Values>::value_type>;

template <typename ExecutionSpace, Details::Concepts::Primitives Values,
          typename IndexableGetter>
KOKKOS_DEDUCTION_GUIDE BruteForce(ExecutionSpace, Values, IndexableGetter)
    -> BruteForce<typename Details::AccessValues<Values>::memory_space,
                  typename Details::AccessValues<Values>::value_type,
                  IndexableGetter>;

template <typename MemorySpace, typename Value, typename IndexableGetter,
          typename BoundingVolume>
template <typename ExecutionSpace, Details::Concepts::Primitives UserValues>
BruteForce<MemorySpace, Value, IndexableGetter, BoundingVolume>::BruteForce(
    ExecutionSpace const &space, UserValues const &user_values,
    IndexableGetter const &indexable_getter)
    : _size(AccessTraits<UserValues>::size(user_values))
    , _values(Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                                 "ArborX::BruteForce::values"),
              _size)
    , _indexable_getter(indexable_getter)
{
  static_assert(Details::KokkosExt::is_accessible_from<MemorySpace,
                                                       ExecutionSpace>::value);

  using Values = Details::AccessValues<UserValues>;
  Values values{user_values}; // NOLINT

  static_assert(
      Details::KokkosExt::is_accessible_from<typename Values::memory_space,
                                             ExecutionSpace>::value,
      "Values must be accessible from the execution space");

  Kokkos::Profiling::ScopedRegion guard("ArborX::BruteForce::BruteForce");

  if (empty())
  {
    return;
  }

  Details::BruteForceImpl::initializeBoundingVolumesAndReduceBoundsOfTheScene(
      space, values, _indexable_getter, _values, _bounds);
}

template <typename MemorySpace, typename Value, typename IndexableGetter,
          typename BoundingVolume>
template <typename ExecutionSpace, Details::Concepts::Predicates UserPredicates,
          typename Callback, typename Ignore>
void BruteForce<MemorySpace, Value, IndexableGetter, BoundingVolume>::query(
    ExecutionSpace const &space, UserPredicates const &user_predicates,
    Callback const &callback, Ignore) const
{
  static_assert(Details::KokkosExt::is_accessible_from<MemorySpace,
                                                       ExecutionSpace>::value);
  Details::check_valid_callback<value_type>(callback, user_predicates);

  using Predicates = Details::AccessValues<UserPredicates>;
  static_assert(
      Details::KokkosExt::is_accessible_from<typename Predicates::memory_space,
                                             ExecutionSpace>::value,
      "Predicates must be accessible from the execution space");

  Predicates predicates{user_predicates}; // NOLINT

  using Tag = typename Predicates::value_type::Tag;

  Details::BruteForceImpl::query(
      Tag{}, space, predicates, _values,
      Details::Indexables{_values, _indexable_getter}, callback);
}

} // namespace ArborX

#endif
