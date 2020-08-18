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

#include <ArborX_AccessTraits.hpp>
#include <ArborX_Box.hpp>
#include <ArborX_DetailsBoundingVolumeHierarchyImpl.hpp>
#include <ArborX_DetailsConcepts.hpp>
#include <ArborX_DetailsKokkosExt.hpp>
#include <ArborX_DetailsNode.hpp>
#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_DetailsTreeConstruction.hpp>

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

  KOKKOS_FUNCTION
  size_type size() const noexcept { return _size; }

  KOKKOS_FUNCTION
  bool empty() const noexcept { return size() == 0; }

  KOKKOS_FUNCTION
  bounding_volume_type bounds() const noexcept { return _bounds; }

  template <typename ExecutionSpace, typename Predicates, typename... Args>
  void query(ExecutionSpace const &space, Predicates const &predicates,
             Args &&... args) const
  {
    Details::check_valid_access_traits(PredicatesTag{}, predicates);
    using Access = AccessTraits<Predicates, PredicatesTag>;
    static_assert(KokkosExt::is_accessible_from<typename Access::memory_space,
                                                ExecutionSpace>::value,
                  "Predicates must be accessible from the execution space");

    Details::BoundingVolumeHierarchyImpl::query(space, *this, predicates,
                                                std::forward<Args>(args)...);
  }

private:
  template <typename BVH, typename Predicates, typename Callback,
            typename /*Enable*/>
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

  KOKKOS_FUNCTION
  Node const *getRoot() const { return _internal_and_leaf_nodes.data(); }

  KOKKOS_FUNCTION
  Node *getRoot() { return _internal_and_leaf_nodes.data(); }

  KOKKOS_FUNCTION
  Node const *getNodePtr(int i) const { return &_internal_and_leaf_nodes(i); }

  KOKKOS_FUNCTION
  bounding_volume_type const &getBoundingVolume(Node const *node) const
  {
    return node->bounding_box;
  }

  KOKKOS_FUNCTION
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
    : _size(AccessTraits<Primitives, PrimitivesTag>::size(primitives))
    , _internal_and_leaf_nodes(
          Kokkos::ViewAllocateWithoutInitializing("internal_and_leaf_nodes"),
          _size > 0 ? 2 * _size - 1 : 0)
{
  Kokkos::Profiling::pushRegion("ArborX:BVH:construction");

  Details::check_valid_access_traits(PrimitivesTag{}, primitives);
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  static_assert(KokkosExt::is_accessible_from<typename Access::memory_space,
                                              ExecutionSpace>::value,
                "Primitives must be accessible from the execution space");

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
  Details::TreeConstruction::generateHierarchy(
      space, morton_indices, getLeafNodes(), getInternalNodes());

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
}

} // namespace ArborX

#endif
