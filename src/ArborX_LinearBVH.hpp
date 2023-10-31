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
#include <ArborX_DetailsCrsGraphWrapperImpl.hpp>
#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>
#include <ArborX_DetailsKokkosExtScopedProfileRegion.hpp>
#include <ArborX_DetailsLegacy.hpp>
#include <ArborX_DetailsNode.hpp>
#include <ArborX_DetailsPermutedData.hpp>
#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_DetailsTreeConstruction.hpp>
#include <ArborX_DetailsTreeTraversal.hpp>
#include <ArborX_HyperBox.hpp>
#include <ArborX_IndexableGetter.hpp>
#include <ArborX_SpaceFillingCurves.hpp>
#include <ArborX_TraversalPolicy.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{

namespace Experimental
{
struct PerThread
{};
} // namespace Experimental

namespace Details
{
struct HappyTreeFriends;
} // namespace Details

template <typename MemorySpace, typename Value,
          typename IndexableGetter = Details::DefaultIndexableGetter,
          typename BoundingVolume = ExperimentalHyperGeometry::Box<
              GeometryTraits::dimension_v<
                  std::decay_t<std::invoke_result_t<IndexableGetter, Value>>>,
              typename GeometryTraits::coordinate_type<std::decay_t<
                  std::invoke_result_t<IndexableGetter, Value>>>::type>>
class BasicBoundingVolumeHierarchy
{
public:
  using memory_space = MemorySpace;
  static_assert(Kokkos::is_memory_space<MemorySpace>::value);
  using size_type = typename MemorySpace::size_type;
  using bounding_volume_type = BoundingVolume;
  using value_type = Value;

  BasicBoundingVolumeHierarchy() = default; // build an empty tree

  template <typename ExecutionSpace, typename Values,
            typename SpaceFillingCurve = Experimental::Morton64>
  BasicBoundingVolumeHierarchy(
      ExecutionSpace const &space, Values const &values,
      IndexableGetter const &indexable_getter = IndexableGetter(),
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
  std::enable_if_t<Kokkos::is_view_v<std::decay_t<View>>>
  query(ExecutionSpace const &space, Predicates const &predicates,
        CallbackOrView &&callback_or_view, View &&view, Args &&...args) const
  {
    KokkosExt::ScopedProfileRegion guard("ArborX::BVH::query_crs");

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

  template <typename Predicate, typename Callback>
  KOKKOS_FUNCTION void query(Experimental::PerThread,
                             Predicate const &predicate,
                             Callback const &callback) const
  {
    ArborX::Details::TreeTraversal<BasicBoundingVolumeHierarchy,
                                   /* Predicates Dummy */ std::true_type,
                                   Callback, typename Predicate::Tag>
        tree_traversal(*this, callback);
    tree_traversal(predicate);
  }

private:
  friend struct Details::HappyTreeFriends;

  using indexable_type =
      std::decay_t<std::invoke_result_t<IndexableGetter, Value>>;
  using leaf_node_type = Details::LeafNode<value_type>;
  using internal_node_type = Details::InternalNode<bounding_volume_type>;

  size_type _size{0};
  bounding_volume_type _bounds;
  Kokkos::View<leaf_node_type *, MemorySpace> _leaf_nodes;
  Kokkos::View<internal_node_type *, MemorySpace> _internal_nodes;
  IndexableGetter _indexable_getter;
};

template <typename MemorySpace>
class BoundingVolumeHierarchy
    : public BasicBoundingVolumeHierarchy<MemorySpace,
                                          Details::PairIndexVolume<Box>,
                                          Details::DefaultIndexableGetter, Box>
{
  using base_type =
      BasicBoundingVolumeHierarchy<MemorySpace, Details::PairIndexVolume<Box>,
                                   Details::DefaultIndexableGetter, Box>;

public:
  using bounding_volume_type = typename base_type::bounding_volume_type;

  BoundingVolumeHierarchy() = default; // build an empty tree

  template <typename ExecutionSpace, typename Primitives,
            typename SpaceFillingCurve = Experimental::Morton64>
  BoundingVolumeHierarchy(ExecutionSpace const &space,
                          Primitives const &primitives,
                          SpaceFillingCurve const &curve = SpaceFillingCurve())
      : base_type(
            space,
            // Validate the primitives before calling the base constructor
            (Details::check_valid_access_traits(PrimitivesTag{}, primitives),
             Details::LegacyValues<Primitives, bounding_volume_type>{
                 primitives}),
            Details::DefaultIndexableGetter(), curve)
  {}

  template <typename ExecutionSpace, typename Predicates, typename Callback>
  void query(ExecutionSpace const &space, Predicates const &predicates,
             Callback const &callback,
             Experimental::TraversalPolicy const &policy =
                 Experimental::TraversalPolicy()) const
  {
    Details::check_valid_callback<int>(callback, predicates);
    base_type::query(space, predicates,
                     Details::LegacyCallbackWrapper<Callback>{callback},
                     policy);
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
      KokkosExt::ScopedProfileRegion guard("ArborX::BVH::query_crs");

      Kokkos::View<int *, MemorySpace> indices(
          "ArborX::CrsGraphWrapper::query::indices", 0);
      base_type::query(space, predicates, Details::LegacyDefaultCallback{},
                       indices, std::forward<OffsetView>(offset),
                       std::forward<Args>(args)...);
      callback(predicates, std::forward<OffsetView>(offset), indices,
               std::forward<OutputView>(out));
    }
  }

  template <typename Predicate, typename Callback>
  KOKKOS_FUNCTION void query(Experimental::PerThread tag,
                             Predicate const &predicate,
                             Callback const &callback) const
  {
    base_type::query(tag, predicate, callback);
  }
};

template <typename MemorySpace>
using BVH = BoundingVolumeHierarchy<MemorySpace>;

template <typename MemorySpace, typename Value, typename IndexableGetter,
          typename BoundingVolume>
template <typename ExecutionSpace, typename Values, typename SpaceFillingCurve>
BasicBoundingVolumeHierarchy<MemorySpace, Value, IndexableGetter,
                             BoundingVolume>::
    BasicBoundingVolumeHierarchy(ExecutionSpace const &space,
                                 Values const &user_values,
                                 IndexableGetter const &indexable_getter,
                                 SpaceFillingCurve const &curve)
    : _size(AccessTraits<Values, PrimitivesTag>::size(user_values))
    , _leaf_nodes(Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                                     "ArborX::BVH::leaf_nodes"),
                  _size)
    , _internal_nodes(Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                                         "ArborX::BVH::internal_nodes"),
                      _size > 1 ? _size - 1 : 0)
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

  constexpr int DIM = GeometryTraits::dimension_v<BoundingVolume>;

  Details::check_valid_space_filling_curve<DIM>(curve);

  KokkosExt::ScopedProfileRegion guard("ArborX::BVH::BVH");

  if (empty())
  {
    return;
  }

  Details::AccessValues<Values> values{user_values};

  if (size() == 1)
  {
    Details::TreeConstruction::initializeSingleLeafTree(
        space, values, _indexable_getter, _leaf_nodes, _bounds);
    return;
  }

  Details::Indexables<decltype(values), IndexableGetter> indexables{
      values, indexable_getter};

  Kokkos::Profiling::pushRegion(
      "ArborX::BVH::BVH::calculate_scene_bounding_box");

  // determine the bounding box of the scene
  ExperimentalHyperGeometry::Box<
      DIM, typename GeometryTraits::coordinate_type<BoundingVolume>::type>
      bbox{};
  Details::TreeConstruction::calculateBoundingBoxOfTheScene(space, indexables,
                                                            bbox);
  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::BVH::BVH::compute_linear_ordering");

  // Map indexables from multidimensional domain to one-dimensional interval
  using LinearOrderingValueType = Kokkos::detected_t<
      Details::SpaceFillingCurveProjectionArchetypeExpression,
      SpaceFillingCurve, decltype(bbox), indexable_type>;
  Kokkos::View<LinearOrderingValueType *, MemorySpace> linear_ordering_indices(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::BVH::BVH::linear_ordering"),
      size());
  Details::TreeConstruction::projectOntoSpaceFillingCurve(
      space, indexables, curve, bbox, linear_ordering_indices);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::BVH::BVH::sort_linearized_order");

  // Compute the ordering of the indexables along the space-filling curve
  auto permutation_indices =
      Details::sortObjects(space, linear_ordering_indices);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::BVH::BVH::generate_hierarchy");

  // Generate bounding volume hierarchy
  Details::TreeConstruction::generateHierarchy(
      space, values, _indexable_getter, permutation_indices,
      linear_ordering_indices, _leaf_nodes, _internal_nodes, _bounds);

  Kokkos::Profiling::popRegion();
}

template <typename MemorySpace, typename Value, typename IndexableGetter,
          typename BoundingVolume>
template <typename ExecutionSpace, typename Predicates, typename Callback>
void BasicBoundingVolumeHierarchy<
    MemorySpace, Value, IndexableGetter,
    BoundingVolume>::query(ExecutionSpace const &space,
                           Predicates const &predicates,
                           Callback const &callback,
                           Experimental::TraversalPolicy const &policy) const
{
  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value);
  Details::check_valid_access_traits(PredicatesTag{}, predicates);
  using Access = AccessTraits<Predicates, PredicatesTag>;
  static_assert(KokkosExt::is_accessible_from<typename Access::memory_space,
                                              ExecutionSpace>::value,
                "Predicates must be accessible from the execution space");
  Details::check_valid_callback<value_type>(callback, predicates);

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
        GeometryTraits::dimension_v<bounding_volume_type>,
        typename GeometryTraits::coordinate_type<bounding_volume_type>::type>
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
