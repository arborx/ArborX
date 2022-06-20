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

#ifndef ARBORX_LINEAR_BVH_HPP
#define ARBORX_LINEAR_BVH_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_Box.hpp>
#include <ArborX_Callbacks.hpp>
#include <ArborX_CrsGraphWrapper.hpp>
#include <ArborX_DetailsBatchedQueries.hpp>
#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>
#include <ArborX_DetailsKokkosExtScopedProfileRegion.hpp>
#include <ArborX_DetailsNode.hpp>
#include <ArborX_DetailsPermutedData.hpp>
#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_DetailsTreeConstruction.hpp>
#include <ArborX_DetailsTreeTraversal.hpp>
#include <ArborX_SpaceFillingCurves.hpp>
#include <ArborX_TraversalPolicy.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{

namespace Details
{
struct HappyTreeFriends;
} // namespace Details

template <typename MemorySpace, typename BoundingVolume = Box,
          typename Enable = void>
class BasicBoundingVolumeHierarchy
{
public:
  using memory_space = MemorySpace;
  static_assert(Kokkos::is_memory_space<MemorySpace>::value, "");
  using size_type = typename MemorySpace::size_type;
  using bounding_volume_type = BoundingVolume;

  BasicBoundingVolumeHierarchy() = default; // build an empty tree

  template <typename ExecutionSpace, typename Primitives,
            typename SpaceFillingCurve = Experimental::Morton64>
  BasicBoundingVolumeHierarchy(
      ExecutionSpace const &space, Primitives const &primitives,
      SpaceFillingCurve const &curve = SpaceFillingCurve());

  KOKKOS_FUNCTION
  size_type size() const noexcept { return _size; }

  KOKKOS_FUNCTION
  bool empty() const noexcept { return size() == 0; }

  KOKKOS_FUNCTION
  bounding_volume_type bounds() const noexcept { return _bounds; }

  template <typename ExecutionSpace, typename Predicates, typename Callback>
  void query(ExecutionSpace const &space, Predicates const &predicates,
             Callback const &callback,
             Experimental::TraversalPolicy const &policy =
                 Experimental::TraversalPolicy()) const;

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
  friend struct Details::HappyTreeFriends;

  using node_type = Details::NodeWithLeftChildAndRope<bounding_volume_type>;

  Kokkos::View<node_type *, MemorySpace> getInternalNodes()
  {
    assert(!empty());
    return Kokkos::subview(_internal_and_leaf_nodes,
                           std::make_pair(size_type{0}, size() - 1));
  }

  Kokkos::View<node_type *, MemorySpace> getLeafNodes()
  {
    assert(!empty());
    return Kokkos::subview(_internal_and_leaf_nodes,
                           std::make_pair(size() - 1, 2 * size() - 1));
  }
  Kokkos::View<node_type const *, MemorySpace> getLeafNodes() const
  {
    assert(!empty());
    return Kokkos::subview(_internal_and_leaf_nodes,
                           std::make_pair(size() - 1, 2 * size() - 1));
  }

  KOKKOS_FUNCTION
  bounding_volume_type const *getRootBoundingVolumePtr() const
  {
    // Need address of the root node's bounding box to copy it back on the host,
    // but can't access _internal_and_leaf_nodes elements from the constructor
    // since the data is on the device.
    assert(Details::HappyTreeFriends::getRoot(*this) == 0 &&
           "workaround below assumes root is stored as first element");
    return &_internal_and_leaf_nodes.data()->bounding_volume;
  }

  size_t _size;
  bounding_volume_type _bounds;
  Kokkos::View<node_type *, MemorySpace> _internal_and_leaf_nodes;
};

template <typename DeviceType>
class BasicBoundingVolumeHierarchy<
    DeviceType, std::enable_if_t<Kokkos::is_device<DeviceType>::value>>
    : public BasicBoundingVolumeHierarchy<typename DeviceType::memory_space>
{
  using base_type =
      BasicBoundingVolumeHierarchy<typename DeviceType::memory_space>;

public:
  using device_type = DeviceType;

  // clang-format off
  [[deprecated("ArborX::BoundingVolumeHierarchy templated on a device type "
               "is deprecated, use it templated on a memory space instead.")]]
  BasicBoundingVolumeHierarchy() = default;
  template <typename Primitives>
  [[deprecated("ArborX::BoundingVolumeHierarchy templated on a device type "
               "is deprecated, use it templated on a memory space instead.")]]
  BasicBoundingVolumeHierarchy(Primitives const &primitives)
      : base_type(
            typename DeviceType::execution_space{}, primitives)
  {
  }
  // clang-format on
  template <typename FirstArgumentType, typename... Args>
  std::enable_if_t<!Kokkos::is_execution_space<FirstArgumentType>::value>
  query(FirstArgumentType &&arg1, Args &&...args) const
  {
    base_type::query(typename DeviceType::execution_space{},
                     std::forward<FirstArgumentType>(arg1),
                     std::forward<Args>(args)...);
  }

private:
  template <typename Tree, typename ExecutionSpace, typename Predicates,
            typename CallbackOrView, typename View, typename... Args>
  friend void ArborX::query(Tree const &tree, ExecutionSpace const &space,
                            Predicates const &predicates,
                            CallbackOrView &&callback_or_view, View &&view,
                            Args &&...args);

  template <typename FirstArgumentType, typename... Args>
  std::enable_if_t<Kokkos::is_execution_space<FirstArgumentType>::value>
  query(FirstArgumentType const &space, Args &&...args) const
  {
    base_type::query(space, std::forward<Args>(args)...);
  }
};

template <typename MemorySpace>
using BoundingVolumeHierarchy = BasicBoundingVolumeHierarchy<MemorySpace>;

template <typename MemorySpace>
using BVH = BoundingVolumeHierarchy<MemorySpace>;

template <typename MemorySpace, typename BoundingVolume, typename Enable>
template <typename ExecutionSpace, typename Primitives,
          typename SpaceFillingCurve>
BasicBoundingVolumeHierarchy<MemorySpace, BoundingVolume, Enable>::
    BasicBoundingVolumeHierarchy(ExecutionSpace const &space,
                                 Primitives const &primitives,
                                 SpaceFillingCurve const &curve)
    : _size(AccessTraits<Primitives, PrimitivesTag>::size(primitives))
    , _internal_and_leaf_nodes(
          Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                             "ArborX::BVH::internal_and_leaf_nodes"),
          _size > 0 ? 2 * _size - 1 : 0)
{
  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value, "");
  Details::check_valid_access_traits(PrimitivesTag{}, primitives);
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  static_assert(KokkosExt::is_accessible_from<typename Access::memory_space,
                                              ExecutionSpace>::value,
                "Primitives must be accessible from the execution space");
  Details::check_valid_space_filling_curve(curve);

  KokkosExt::ScopedProfileRegion guard("ArborX::BVH::BVH");

  if (empty())
  {
    return;
  }

  Kokkos::Profiling::pushRegion(
      "ArborX::BVH::BVH::calculate_scene_bounding_box");

  // determine the bounding box of the scene
  Box bbox{};
  Details::TreeConstruction::calculateBoundingBoxOfTheScene(space, primitives,
                                                            bbox);

  Kokkos::Profiling::popRegion();

  if (size() == 1)
  {
    Details::TreeConstruction::initializeSingleLeafNode(
        space, primitives, _internal_and_leaf_nodes);
    Kokkos::deep_copy(
        space,
        Kokkos::View<BoundingVolume, Kokkos::HostSpace,
                     Kokkos::MemoryUnmanaged>(&_bounds),
        Kokkos::View<BoundingVolume const, MemorySpace,
                     Kokkos::MemoryUnmanaged>(getRootBoundingVolumePtr()));
    return;
  }

  Kokkos::Profiling::pushRegion("ArborX::BVH::BVH::compute_linear_ordering");

  // map primitives from multidimensional domain to one-dimensional interval
  using LinearOrderingValueType = Kokkos::detected_t<
      Details::SpaceFillingCurveProjectionArchetypeExpression,
      SpaceFillingCurve, Point>;
  Kokkos::View<LinearOrderingValueType *, MemorySpace> linear_ordering_indices(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::BVH::BVH::linear_ordering"),
      size());
  Details::TreeConstruction::projectOntoSpaceFillingCurve(
      space, primitives, curve, bbox, linear_ordering_indices);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::BVH::BVH::sort_linearized_order");

  // compute the ordering of the primitives along the space-filling curve
  auto permutation_indices =
      Details::sortObjects(space, linear_ordering_indices);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::BVH::BVH::generate_hierarchy");

  // generate bounding volume hierarchy
  Details::TreeConstruction::generateHierarchy(
      space, primitives, permutation_indices, linear_ordering_indices,
      getLeafNodes(), getInternalNodes());

  Kokkos::deep_copy(
      space,
      Kokkos::View<BoundingVolume, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
          &_bounds),
      Kokkos::View<BoundingVolume const, MemorySpace, Kokkos::MemoryUnmanaged>(
          getRootBoundingVolumePtr()));

  Kokkos::Profiling::popRegion();
}

template <typename MemorySpace, typename BoundingVolume, typename Enable>
template <typename ExecutionSpace, typename Predicates, typename Callback>
void BasicBoundingVolumeHierarchy<MemorySpace, BoundingVolume, Enable>::query(
    ExecutionSpace const &space, Predicates const &predicates,
    Callback const &callback, Experimental::TraversalPolicy const &policy) const
{
  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value, "");
  Details::check_valid_access_traits(PredicatesTag{}, predicates);
  using Access = AccessTraits<Predicates, PredicatesTag>;
  static_assert(KokkosExt::is_accessible_from<typename Access::memory_space,
                                              ExecutionSpace>::value,
                "Predicates must be accessible from the execution space");
  Details::check_valid_callback(callback, predicates);

  using Tag = typename Details::AccessTraitsHelper<Access>::tag;
  auto profiling_prefix =
      std::string("ArborX::BVH::query::") +
      (std::is_same<Tag, Details::SpatialPredicateTag>{} ? "spatial"
                                                         : "nearest");

  Kokkos::Profiling::pushRegion(profiling_prefix);

  if (policy._sort_predicates)
  {
    Kokkos::Profiling::pushRegion(profiling_prefix + "::compute_permutation");
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
    auto permute = Details::BatchedQueries<DeviceType>::
        sortPredicatesAlongSpaceFillingCurve(space, Experimental::Morton32(),
                                             static_cast<Box>(bounds()),
                                             predicates);
    Kokkos::Profiling::popRegion();

    using PermutedPredicates =
        Details::PermutedData<Predicates, decltype(permute)>;
    Details::traverse(space, *this, PermutedPredicates{predicates, permute},
                      callback);
  }
  else
  {
    Details::traverse(space, *this, predicates, callback);
  }

  Kokkos::Profiling::popRegion();
}

} // namespace ArborX

#endif
