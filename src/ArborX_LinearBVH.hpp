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
#include <ArborX_HyperBox.hpp>
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
  static_assert(Kokkos::is_memory_space<MemorySpace>::value);
  using size_type = typename MemorySpace::size_type;
  using bounding_volume_type = BoundingVolume;
  using value_type = Details::PairIndexVolume<bounding_volume_type>;

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

  using leaf_node_type = Details::LeafNode<value_type>;
  using internal_node_type = Details::InternalNode<bounding_volume_type>;

  KOKKOS_FUNCTION
  bounding_volume_type const *getRootBoundingVolumePtr() const
  {
    int const n = size();
    // Need address of the root node's bounding box to copy it back on the host,
    // but can't access node elements from the constructor since the data is on
    // the device.
    assert((n == 1 || Details::HappyTreeFriends::getRoot(*this) == n) &&
           "workaround below assumes root is stored as first element");
    return (n > 1 ? &_internal_nodes.data()->bounding_volume
                  : &_leaf_nodes.data()->value.bounding_volume);
  }

  size_type _size{0};
  bounding_volume_type _bounds;
  Kokkos::View<leaf_node_type *, MemorySpace> _leaf_nodes;
  Kokkos::View<internal_node_type *, MemorySpace> _internal_nodes;
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
    , _leaf_nodes(Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                                     "ArborX::BVH::leaf_nodes"),
                  _size)
    , _internal_nodes(Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                                         "ArborX::BVH::internal_nodes"),
                      _size > 1 ? _size - 1 : 0)
{
  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value);
  Details::check_valid_access_traits(PrimitivesTag{}, primitives);
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  static_assert(KokkosExt::is_accessible_from<typename Access::memory_space,
                                              ExecutionSpace>::value,
                "Primitives must be accessible from the execution space");
  constexpr int DIM = GeometryTraits::dimension_v<BoundingVolume>;

  Details::check_valid_space_filling_curve<DIM>(curve);

  KokkosExt::ScopedProfileRegion guard("ArborX::BVH::BVH");

  if (empty())
  {
    return;
  }

  Kokkos::Profiling::pushRegion(
      "ArborX::BVH::BVH::calculate_scene_bounding_box");

  // determine the bounding box of the scene
  ExperimentalHyperGeometry::Box<DIM> bbox{};
  Details::TreeConstruction::calculateBoundingBoxOfTheScene(space, primitives,
                                                            bbox);

  Kokkos::Profiling::popRegion();

  if (size() == 1)
  {
    Details::TreeConstruction::initializeSingleLeafNode(space, primitives,
                                                        _leaf_nodes);
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
      SpaceFillingCurve, decltype(bbox),
      std::decay_t<decltype(Access::get(primitives, 0))>>;
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
      _leaf_nodes, _internal_nodes);

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
    Callback const &legacy_callback,
    Experimental::TraversalPolicy const &policy) const
{
  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value);
  Details::check_valid_access_traits(PredicatesTag{}, predicates);
  using Access = AccessTraits<Predicates, PredicatesTag>;
  static_assert(KokkosExt::is_accessible_from<typename Access::memory_space,
                                              ExecutionSpace>::value,
                "Predicates must be accessible from the execution space");
  Details::check_valid_callback(legacy_callback, predicates);
  Details::LegacyCallbackWrapper<Callback, value_type> callback{
      legacy_callback};

  using Tag = typename Details::AccessTraitsHelper<Access>::tag;
  std::string profiling_prefix = "ArborX::BVH::query::";
  if constexpr (std::is_same_v<Tag, Details::SpatialPredicateTag>)
  {
    profiling_prefix += "spatial";
  }
  else if constexpr (std::is_same_v<Tag, Details::NearestPredicateTag>)
  {
    profiling_prefix += "nearest";
  }
  else if constexpr (std::is_same_v<Tag,
                                    Experimental::OrderedSpatialPredicateTag>)
  {
    profiling_prefix += "ordered_spatial";
  }
  else
  {
    static_assert(std::is_void_v<Tag>, "ArborX implementation bug");
  }

  Kokkos::Profiling::pushRegion(profiling_prefix);

  if (policy._sort_predicates)
  {
    Kokkos::Profiling::pushRegion(profiling_prefix + "::compute_permutation");
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
    ExperimentalHyperGeometry::Box<
        GeometryTraits::dimension_v<bounding_volume_type>>
        scene_bounding_box{};
    using namespace Details;
    expand(scene_bounding_box, bounds());
    auto permute = Details::BatchedQueries<DeviceType>::
        sortPredicatesAlongSpaceFillingCurve(space, Experimental::Morton32(),
                                             scene_bounding_box, predicates);
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
