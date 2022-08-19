/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
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
#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{

template <typename MemorySpace, typename BoundingVolume = Box>
class BruteForce
{
public:
  using memory_space = MemorySpace;
  static_assert(Kokkos::is_memory_space<MemorySpace>::value);
  using size_type = typename MemorySpace::size_type;
  using bounding_volume_type = BoundingVolume;

  BruteForce() = default;

  template <typename ExecutionSpace, typename Primitives>
  BruteForce(ExecutionSpace const &space, Primitives const &primitives);

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
  std::enable_if_t<Kokkos::is_view<std::decay_t<View>>{}>
  query(ExecutionSpace const &space, Predicates const &predicates,
        CallbackOrView &&callback_or_view, View &&view, Args &&...args) const
  {
    ArborX::query(*this, space, predicates,
                  std::forward<CallbackOrView>(callback_or_view),
                  std::forward<View>(view), std::forward<Args>(args)...);
  }

private:
  size_type _size{0};
  bounding_volume_type _bounds;
  Kokkos::View<bounding_volume_type *, memory_space> _bounding_volumes;
};

template <typename MemorySpace, typename BoundingVolume>
template <typename ExecutionSpace, typename Primitives>
BruteForce<MemorySpace, BoundingVolume>::BruteForce(
    ExecutionSpace const &space, Primitives const &primitives)
    : _size(AccessTraits<Primitives, PrimitivesTag>::size(primitives))
    , _bounding_volumes(
          Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                             "ArborX::BruteForce::bounding_volumes"),
          _size)
{
  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value);
  Details::check_valid_access_traits(PrimitivesTag{}, primitives);
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  static_assert(KokkosExt::is_accessible_from<typename Access::memory_space,
                                              ExecutionSpace>::value,
                "Primitives must be accessible from the execution space");

  Kokkos::Profiling::pushRegion("ArborX::BruteForce::BruteForce");

  Details::BruteForceImpl::initializeBoundingVolumesAndReduceBoundsOfTheScene(
      space, primitives, _bounding_volumes, _bounds);

  Kokkos::Profiling::popRegion();
}

template <typename MemorySpace, typename BoundingVolume>
template <typename ExecutionSpace, typename Predicates, typename Callback,
          typename Ignore>
void BruteForce<MemorySpace, BoundingVolume>::query(
    ExecutionSpace const &space, Predicates const &predicates,
    Callback const &callback, Ignore) const
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
  Details::check_valid_callback(callback, predicates);

  Kokkos::Profiling::pushRegion("ArborX::BruteForce::query::spatial");

  Details::BruteForceImpl::query(space, _bounding_volumes, predicates,
                                 callback);

  Kokkos::Profiling::popRegion();
}

} // namespace ArborX

#endif
