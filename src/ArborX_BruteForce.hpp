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

#ifndef ARBORX_BRUTE_FORCE_HPP
#define ARBORX_BRUTE_FORCE_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_Box.hpp>
#include <ArborX_CrsGraphWrapper.hpp>
#include <ArborX_DetailsBruteForceImpl.hpp>
#include <ArborX_DetailsCrsGraphWrapperImpl.hpp>
#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>
#include <ArborX_DetailsLegacy.hpp>
#include <ArborX_IndexableGetter.hpp>
#include <ArborX_PairValueIndex.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

namespace ArborX
{

template <typename MemorySpace,
          typename Value = Details::LegacyDefaultTemplateValue,
          typename IndexableGetter = Details::DefaultIndexableGetter,
          typename BoundingVolume = ExperimentalHyperGeometry::Box<
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

  template <typename ExecutionSpace, typename Values>
  BruteForce(ExecutionSpace const &space, Values const &values,
             IndexableGetter const &indexable_getter = IndexableGetter());

  KOKKOS_FUNCTION
  size_type size() const noexcept { return _size; }

  KOKKOS_FUNCTION
  bool empty() const noexcept { return size() == 0; }

  KOKKOS_FUNCTION
  bounding_volume_type bounds() const noexcept { return _bounds; }

  template <typename ExecutionSpace, typename Predicates, typename Callback,
            typename Ignore = int>
  void query(ExecutionSpace const &space, Predicates const &predicates,
             Callback const &callback, Ignore = Ignore()) const;

  template <typename ExecutionSpace, typename UserPredicates,
            typename CallbackOrView, typename View, typename... Args>
  std::enable_if_t<Kokkos::is_view_v<std::decay_t<View>>>
  query(ExecutionSpace const &space, UserPredicates const &user_predicates,
        CallbackOrView &&callback_or_view, View &&view, Args &&...args) const
  {
    Kokkos::Profiling::ScopedRegion guard("ArborX::BruteForce::query_crs");

    Details::CrsGraphWrapperImpl::
        check_valid_callback_if_first_argument_is_not_a_view<value_type>(
            callback_or_view, user_predicates, view);

    using Predicates = Details::AccessValues<UserPredicates, PredicatesTag>;
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

template <typename MemorySpace>
class BruteForce<MemorySpace, Details::LegacyDefaultTemplateValue,
                 Details::DefaultIndexableGetter,
                 ExperimentalHyperGeometry::Box<3, float>>
    : public BruteForce<MemorySpace, PairValueIndex<Box>,
                        Details::DefaultIndexableGetter, Box>
{
  using base_type = BruteForce<MemorySpace, PairValueIndex<Box>,
                               Details::DefaultIndexableGetter, Box>;

public:
  using bounding_volume_type = typename base_type::bounding_volume_type;

  BruteForce() = default;

  template <typename ExecutionSpace, typename Primitives>
  BruteForce(ExecutionSpace const &space, Primitives const &primitives)
      : base_type(
            space,
            // Validate the primitives before calling the base constructor
            (Details::check_valid_access_traits(PrimitivesTag{}, primitives),
             Details::LegacyValues<Primitives, bounding_volume_type>{
                 primitives}),
            Details::DefaultIndexableGetter())
  {}

  template <typename ExecutionSpace, typename Predicates, typename Callback,
            typename Ignore = int>
  void query(ExecutionSpace const &space, Predicates const &predicates,
             Callback const &callback, Ignore = Ignore()) const
  {
    Details::check_valid_callback<int>(callback, predicates);
    base_type::query(space, predicates,
                     Details::LegacyCallbackWrapper<Callback>{callback});
  }

  template <typename ExecutionSpace, typename Predicates, typename View,
            typename... Args>
  std::enable_if_t<Kokkos::is_view_v<std::decay_t<View>>>
  query(ExecutionSpace const &space, Predicates const &predicates, View &&view,
        Args &&...args) const
  {
    base_type::query(space, predicates, Details::LegacyDefaultCallback{},
                     std::forward<View>(view), std::forward<Args>(args)...);
  }

  template <typename ExecutionSpace, typename Predicates, typename Callback,
            typename OutputView, typename OffsetView, typename... Args>
  std::enable_if_t<!Kokkos::is_view_v<std::decay_t<Callback>>>
  query(ExecutionSpace const &space, Predicates const &predicates,
        Callback &&callback, OutputView &&out, OffsetView &&offset,
        Args &&...args) const
  {
    if constexpr (!Details::is_tagged_post_callback<
                      std::decay_t<Callback>>::value)
    {
      Details::check_valid_callback<int>(callback, predicates, out);
      base_type::query(space, predicates,
                       Details::LegacyCallbackWrapper<std::decay_t<Callback>>{
                           std::forward<Callback>(callback)},
                       std::forward<OutputView>(out),
                       std::forward<OffsetView>(offset),
                       std::forward<Args>(args)...);
    }
    else
    {
      Kokkos::Profiling::ScopedRegion guard("ArborX::BruteForce::query_crs");

      Kokkos::View<int *, MemorySpace> indices(
          "ArborX::CrsGraphWrapper::query::indices", 0);
      base_type::query(space, predicates, Details::LegacyDefaultCallback{},
                       indices, std::forward<OffsetView>(offset),
                       std::forward<Args>(args)...);
      callback(predicates, std::forward<OffsetView>(offset), indices,
               std::forward<OutputView>(out));
    }
  }
};

template <typename MemorySpace, typename Value, typename IndexableGetter,
          typename BoundingVolume>
template <typename ExecutionSpace, typename UserValues>
BruteForce<MemorySpace, Value, IndexableGetter, BoundingVolume>::BruteForce(
    ExecutionSpace const &space, UserValues const &user_values,
    IndexableGetter const &indexable_getter)
    : _size(AccessTraits<UserValues, PrimitivesTag>::size(user_values))
    , _values(Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                                 "ArborX::BruteForce::values"),
              _size)
    , _indexable_getter(indexable_getter)
{
  static_assert(Details::KokkosExt::is_accessible_from<MemorySpace,
                                                       ExecutionSpace>::value);
  // FIXME redo with RangeTraits
  Details::check_valid_access_traits<UserValues>(
      PrimitivesTag{}, user_values, Details::DoNotCheckGetReturnType());

  using Values = Details::AccessValues<UserValues, PrimitivesTag>;
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
template <typename ExecutionSpace, typename UserPredicates, typename Callback,
          typename Ignore>
void BruteForce<MemorySpace, Value, IndexableGetter, BoundingVolume>::query(
    ExecutionSpace const &space, UserPredicates const &user_predicates,
    Callback const &callback, Ignore) const
{
  static_assert(Details::KokkosExt::is_accessible_from<MemorySpace,
                                                       ExecutionSpace>::value);
  Details::check_valid_access_traits(PredicatesTag{}, user_predicates);
  Details::check_valid_callback<value_type>(callback, user_predicates);

  using Predicates = Details::AccessValues<UserPredicates, PredicatesTag>;
  static_assert(
      Details::KokkosExt::is_accessible_from<typename Predicates::memory_space,
                                             ExecutionSpace>::value,
      "Predicates must be accessible from the execution space");

  Predicates predicates{user_predicates}; // NOLINT

  using Tag = typename Predicates::value_type::Tag;
  static_assert(std::is_same<Tag, Details::SpatialPredicateTag>{},
                "nearest query not implemented yet");

  Kokkos::Profiling::pushRegion("ArborX::BruteForce::query::spatial");

  Details::BruteForceImpl::query(
      space, predicates, _values,
      Details::Indexables<decltype(_values), IndexableGetter>{
          _values, _indexable_getter},
      callback);

  Kokkos::Profiling::popRegion();
}

} // namespace ArborX

#endif
