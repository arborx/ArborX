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
#include <ArborX_DetailsKokkosExtScopedProfileRegion.hpp>
#include <ArborX_DetailsLegacy.hpp>
#include <ArborX_IndexableGetter.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{

template <typename MemorySpace, typename Value,
          typename IndexableGetter = Details::DefaultIndexableGetter,
          typename BoundingVolume = ExperimentalHyperGeometry::Box<
              GeometryTraits::dimension_v<
                  std::decay_t<std::invoke_result_t<IndexableGetter, Value>>>,
              typename GeometryTraits::coordinate_type<std::decay_t<
                  std::invoke_result_t<IndexableGetter, Value>>>::type>>
class BasicBruteForce
{
public:
  using memory_space = MemorySpace;
  static_assert(Kokkos::is_memory_space<MemorySpace>::value);
  using size_type = typename MemorySpace::size_type;
  using bounding_volume_type = BoundingVolume;
  using value_type = Value;

  BasicBruteForce() = default;

  template <typename ExecutionSpace, typename Values>
  BasicBruteForce(ExecutionSpace const &space, Values const &values,
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

  template <typename ExecutionSpace, typename Predicates,
            typename CallbackOrView, typename View, typename... Args>
  std::enable_if_t<Kokkos::is_view_v<std::decay_t<View>>>
  query(ExecutionSpace const &space, Predicates const &predicates,
        CallbackOrView &&callback_or_view, View &&view, Args &&...args) const
  {
    KokkosExt::ScopedProfileRegion guard("ArborX::BruteForce::query_crs");

    Details::CrsGraphWrapperImpl::
        check_valid_callback_if_first_argument_is_not_a_view<value_type>(
            callback_or_view, predicates, view);

    using Access = AccessTraits<Predicates, PredicatesTag>;
    using Tag = typename Details::AccessTraitsHelper<Access>::tag;

    Details::CrsGraphWrapperImpl::queryDispatch(
        Tag{}, *this, space, predicates,
        std::forward<CallbackOrView>(callback_or_view),
        std::forward<View>(view), std::forward<Args>(args)...);
  }

private:
  size_type _size{0};
  bounding_volume_type _bounds;
  Kokkos::View<value_type *, memory_space> _values;
  IndexableGetter _indexable_getter;
};

template <typename MemorySpace, typename BoundingVolume = Box>
class BruteForce
    : public BasicBruteForce<MemorySpace, Details::PairIndexVolume<Box>,
                             Details::DefaultIndexableGetter, BoundingVolume>
{
  using base_type =
      BasicBruteForce<MemorySpace, Details::PairIndexVolume<Box>,
                      Details::DefaultIndexableGetter, BoundingVolume>;

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
      KokkosExt::ScopedProfileRegion guard("ArborX::BruteForce::query_crs");

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
template <typename ExecutionSpace, typename Values>
BasicBruteForce<MemorySpace, Value, IndexableGetter, BoundingVolume>::
    BasicBruteForce(ExecutionSpace const &space, Values const &user_values,
                    IndexableGetter const &indexable_getter)
    : _size(AccessTraits<Values, PrimitivesTag>::size(user_values))
    , _values(Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                                 "ArborX::BruteForce::values"),
              _size)
    , _indexable_getter(indexable_getter)
{
  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value);
  // FIXME redo with RangeTraits
  Details::check_valid_access_traits<Values>(
      PrimitivesTag{}, user_values, Details::DoNotCheckGetReturnType());
  using Access = AccessTraits<Values, PrimitivesTag>;
  static_assert(KokkosExt::is_accessible_from<typename Access::memory_space,
                                              ExecutionSpace>::value,
                "Values must be accessible from the execution space");

  KokkosExt::ScopedProfileRegion guard("ArborX::BruteForce::BruteForce");

  if (empty())
  {
    return;
  }

  Details::AccessValues<Values> values{user_values};

  Details::BruteForceImpl::initializeBoundingVolumesAndReduceBoundsOfTheScene(
      space, values, _indexable_getter, _values, _bounds);
}

template <typename MemorySpace, typename Value, typename IndexableGetter,
          typename BoundingVolume>
template <typename ExecutionSpace, typename Predicates, typename Callback,
          typename Ignore>
void BasicBruteForce<MemorySpace, Value, IndexableGetter,
                     BoundingVolume>::query(ExecutionSpace const &space,
                                            Predicates const &predicates,
                                            Callback const &callback,
                                            Ignore) const
{
  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value);
  Details::check_valid_access_traits(PredicatesTag{}, predicates);
  using Access = AccessTraits<Predicates, PredicatesTag>;
  static_assert(KokkosExt::is_accessible_from<typename Access::memory_space,
                                              ExecutionSpace>::value,
                "Predicates must be accessible from the execution space");
  using Tag = typename Details::AccessTraitsHelper<Access>::tag;
  static_assert(std::is_same<Tag, Details::SpatialPredicateTag>{},
                "nearest query not implemented yet");
  Details::check_valid_callback<Value>(callback, predicates);

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
