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

#ifndef ARBORX_LINEAR_BVH_HPP
#define ARBORX_LINEAR_BVH_HPP

#include <ArborX_Box.hpp>
#include <ArborX_DetailsBoundingVolumeHierarchyImpl.hpp>
#include <ArborX_DetailsConcepts.hpp>
#include <ArborX_DetailsKokkosExt.hpp>
#include <ArborX_DetailsNode.hpp>
#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_DetailsTags.hpp>
#include <ArborX_DetailsTreeConstruction.hpp>
#include <ArborX_Traits.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{
namespace Details
{
template <typename DeviceType>
struct TreeVisualization;
}

template <typename MemorySpace, typename Enable = void>
class BoundingVolumeHierarchy
{
public:
  using memory_space = MemorySpace;
  static_assert(Kokkos::is_memory_space<MemorySpace>::value, "");
  using size_type = typename MemorySpace::size_type;
  using bounding_volume_type = Box;

  BoundingVolumeHierarchy() = default; // build an empty tree

  template <typename ExecutionSpace, typename Primitives>
  BoundingVolumeHierarchy(ExecutionSpace const &space,
                          Primitives const &primitives);

  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept { return _size; }

  KOKKOS_INLINE_FUNCTION
  bool empty() const noexcept { return size() == 0; }

  bounding_volume_type bounds() const noexcept { return _bounds; }

  template <typename ExecutionSpace, typename Predicates, typename... Args>
  void query(ExecutionSpace const &space, Predicates const &predicates,
             Args &&... args) const
  {
    using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
    static_assert(
        Details::is_complete<Access>::value,
        "Must specialize 'Traits::Access<Predicates,Traits::PredicatesTag>'");
    static_assert(
        Details::has_memory_space<Access>::value,
        "Traits::Access<Predicates,Traits::PredicatesTag> must define "
        "'memory_space' member type that is a valid Kokkos memory space");
    static_assert(
        KokkosExt::is_accessible_from<typename Access::memory_space,
                                      ExecutionSpace>::value,
        "Traits::Access<Predicates,Traits::PredicatesTag>::memory_space must "
        "be accessible from the bounding volume hierarchy execution space");
    static_assert(
        Details::has_size<Access>::value,
        "Traits::Access<Predicates,Traits::PredicatesTag> must define "
        "'size()' member function");
    static_assert(
        Details::has_get<Access>::value,
        "Traits::Access<Predicates,Traits::PredicatesTag> must define 'get()' "
        "member function");
    using Tag =
        typename Details::Tag<Details::decay_result_of_get_t<Access>>::type;
    static_assert(std::is_same<Tag, Details::NearestPredicateTag>::value ||
                      std::is_same<Tag, Details::SpatialPredicateTag>::value,
                  "Invalid tag for the predicates");

    Details::BoundingVolumeHierarchyImpl::queryDispatch(
        Tag{}, *this, space, predicates, std::forward<Args>(args)...);
  }

private:
  template <typename DeviceType>
  friend struct Details::TreeTraversal;
  template <typename DeviceType>
  friend struct Details::TreeVisualization;
  using Node = Details::Node;

  Kokkos::View<Node *, MemorySpace> getInternalNodes()
  {
    assert(!empty());
    return Kokkos::subview(_internal_and_leaf_nodes,
                           std::make_pair(size_type{0}, size() - 1));
  }

  Kokkos::View<Node *, MemorySpace> getLeafNodes()
  {
    assert(!empty());
    return Kokkos::subview(_internal_and_leaf_nodes,
                           std::make_pair(size() - 1, 2 * size() - 1));
  }

  KOKKOS_INLINE_FUNCTION
  Node const *getRoot() const { return _internal_and_leaf_nodes.data(); }

  KOKKOS_INLINE_FUNCTION
  Node *getRoot() { return _internal_and_leaf_nodes.data(); }

  KOKKOS_INLINE_FUNCTION
  Node const *getNodePtr(int i) const { return &_internal_and_leaf_nodes(i); }

  KOKKOS_INLINE_FUNCTION
  bounding_volume_type const &getBoundingVolume(Node const *node) const
  {
    return node->bounding_box;
  }

  KOKKOS_INLINE_FUNCTION
  bounding_volume_type &getBoundingVolume(Node *node)
  {
    return node->bounding_box;
  }

  size_t _size;
  bounding_volume_type _bounds;
  Kokkos::View<Node *, MemorySpace> _internal_and_leaf_nodes;
};

template <typename DeviceType>
class BoundingVolumeHierarchy<
    DeviceType, std::enable_if_t<Kokkos::is_device<DeviceType>::value>>
    : public BoundingVolumeHierarchy<typename DeviceType::memory_space>
{
public:
  using device_type = DeviceType;
  BoundingVolumeHierarchy() = default;
  template <typename Primitives>
  BoundingVolumeHierarchy(Primitives const &primitives)
      : BoundingVolumeHierarchy<typename DeviceType::memory_space>(
            typename DeviceType::execution_space{}, primitives)
  {
  }
  template <typename... Args>
  void query(Args &&... args) const
  {
    BoundingVolumeHierarchy<typename DeviceType::memory_space>::query(
        typename DeviceType::execution_space{}, std::forward<Args>(args)...);
  }
};

template <typename MemorySpace>
using BVH = BoundingVolumeHierarchy<MemorySpace>;

template <typename MemorySpace, typename Enable>
template <typename ExecutionSpace, typename Primitives>
BoundingVolumeHierarchy<MemorySpace, Enable>::BoundingVolumeHierarchy(
    ExecutionSpace const &space, Primitives const &primitives)
    : _size(Traits::Access<Primitives, Traits::PrimitivesTag>::size(primitives))
    , _internal_and_leaf_nodes(
          Kokkos::ViewAllocateWithoutInitializing("internal_and_leaf_nodes"),
          _size > 0 ? 2 * _size - 1 : 0)
{
  Kokkos::Profiling::pushRegion("ArborX:BVH:construction");

  using Access = Traits::Access<Primitives, Traits::PrimitivesTag>;
  static_assert(
      Details::is_complete<Access>::value,
      "Must specialize 'Traits::Access<Primitives,Traits::PrimitivesTag>'");
  static_assert(
      Details::has_memory_space<Access>::value,
      "Traits::Access<Primitives,Traits::PrimitivesTag> must define "
      "'memory_space' member type that is a valid Kokkos memory space");
  static_assert(
      KokkosExt::is_accessible_from<typename Access::memory_space,
                                    ExecutionSpace>::value,
      "Traits::Access<Primitives,Traits::PrimitivesTag>::memory_space must be "
      "accessible from the bounding volume hierarchy execution space");
  static_assert(Details::has_size<Access>::value,
                "Traits::Access<Primitives,Traits::PrimitivesTag> must define "
                "'size()' member function");
  static_assert(Details::has_get<Access>::value,
                "Traits::Access<Primitives,Traits::PrimitivesTag> must define "
                "'get()' member function");
  static_assert(
      std::is_same<Details::decay_result_of_get_t<Access>, Point>::value ||
          std::is_same<Details::decay_result_of_get_t<Access>, Box>::value,
      "Traits::Access<Primitives,Traits::PrimitivesTag>::get() return type "
      "must decay to Point or to Box");

  if (empty())
  {
    return;
  }

  Kokkos::Profiling::pushRegion("ArborX:BVH:calculate_scene_bounding_box");

  // determine the bounding box of the scene
  Details::TreeConstruction::calculateBoundingBoxOfTheScene(space, primitives,
                                                            _bounds);

  Kokkos::Profiling::popRegion();

  if (size() == 1)
  {
    Kokkos::View<unsigned int *, MemorySpace> permutation_indices(
        Kokkos::view_alloc("permute", space), 1);
    Details::TreeConstruction::initializeLeafNodes(
        space, primitives, permutation_indices, getLeafNodes());
    return;
  }

  Kokkos::Profiling::pushRegion("ArborX:BVH:assign_morton_codes");

  // calculate Morton codes of all objects
  Kokkos::View<unsigned int *, MemorySpace> morton_indices(
      Kokkos::ViewAllocateWithoutInitializing("morton"), size());
  Details::TreeConstruction::assignMortonCodes(space, primitives,
                                               morton_indices, _bounds);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX:BVH:sort_morton_codes");

  // compute the ordering of primitives along Z-order space-filling curve
  auto permutation_indices = Details::sortObjects(space, morton_indices);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX:BVH:init_leaves");

  // initialize leaves using the computed ordering
  Details::TreeConstruction::initializeLeafNodes(
      space, primitives, permutation_indices, getLeafNodes());

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX:BVH:generate_hierarchy");

  // generate bounding volume hierarchy
  Kokkos::View<int *, MemorySpace> parents(
      Kokkos::ViewAllocateWithoutInitializing("parents"), 2 * size() - 1);
  Details::TreeConstruction::generateHierarchy(
      space, morton_indices, getLeafNodes(), getInternalNodes(), parents);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX:BVH:calculate_bounding_volumes");

  // calculate bounding volume for each internal node by walking the
  // hierarchy toward the root
  Details::TreeConstruction::calculateInternalNodesBoundingVolumes(
      space, getLeafNodes(), getInternalNodes(), parents);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
}

} // namespace ArborX

#endif
