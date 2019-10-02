/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
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
#include <ArborX_DetailsTreeConstruction.hpp>
#include <ArborX_Traits.hpp>

#include <Kokkos_Macros.hpp>
#include <Kokkos_View.hpp>

namespace ArborX
{
namespace Details
{
template <typename DeviceType>
struct TreeVisualization;
}

template <typename DeviceType>
class BoundingVolumeHierarchy
{
public:
  using device_type = DeviceType;
  using bounding_volume_type = Box;
  using size_type = typename DeviceType::memory_space::size_type;

  BoundingVolumeHierarchy() = default; // build an empty tree

  template <typename Primitives>
  BoundingVolumeHierarchy(Primitives const &primitives);

  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept { return _size; }

  KOKKOS_INLINE_FUNCTION
  bool empty() const noexcept { return size() == 0; }

  bounding_volume_type bounds() const noexcept { return _bounds; }

  template <typename Predicates, typename... Args>
  void query(Predicates const &predicates, Args &&... args) const
  {
    // FIXME placeholder for concept check

    using Tag =
        typename Details::TagHelper<Predicates, Traits::PredicatesTag>::type;
    static_assert(std::is_same<Tag, Details::NearestPredicateTag>::value ||
                      std::is_same<Tag, Details::SpatialPredicateTag>::value,
                  "Invalid tag for the predicates");
    Details::BoundingVolumeHierarchyImpl<DeviceType>::queryDispatch(
        Tag{}, *this, predicates, std::forward<Args>(args)...);
  }

private:
  friend struct Details::TreeTraversal<DeviceType>;
  friend struct Details::TreeVisualization<DeviceType>;

  Kokkos::View<Node *, DeviceType> getInternalNodes()
  {
    assert(!empty());
    return Kokkos::subview(_internal_and_leaf_nodes,
                           std::make_pair(size_type{0}, size() - 1));
  }

  Kokkos::View<Node *, DeviceType> getLeafNodes()
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
  bounding_volume_type const &getBoundingVolume(Node const *node) const
  {
    return node->bounding_box;
  }

  KOKKOS_INLINE_FUNCTION
  bounding_volume_type &getBoundingVolume(Node *node)
  {
    return node->bounding_box;
  }

  KOKKOS_INLINE_FUNCTION
  size_t getLeafPermutationIndex(Node const *leaf) const
  {
    static_assert(sizeof(size_t) == sizeof(Node *),
                  "Conversion is a bad idea if these sizes do not match");
    return reinterpret_cast<size_t>(leaf->children.second);
  }

  KOKKOS_INLINE_FUNCTION
  bool isLeaf(Node const *node) const
  {
    return (node->children.first == nullptr);
  }

  size_t _size;
  bounding_volume_type _bounds;
  Kokkos::View<Node *, DeviceType> _internal_and_leaf_nodes;
};

template <typename DeviceType>
using BVH = BoundingVolumeHierarchy<DeviceType>;

template <typename DeviceType>
template <typename Primitives>
BoundingVolumeHierarchy<DeviceType>::BoundingVolumeHierarchy(
    Primitives const &primitives)
    : _size(Traits::Access<Primitives, Traits::PrimitivesTag>::size(primitives))
    , _internal_and_leaf_nodes(
          Kokkos::ViewAllocateWithoutInitializing("internal_and_leaf_nodes"),
          _size > 0 ? 2 * _size - 1 : 0)
{
  Kokkos::Profiling::pushRegion("ArborX:BVH:construction");

  // FIXME placeholder for concept check

  if (empty())
  {
    return;
  }

  Kokkos::Profiling::pushRegion("ArborX:BVH:calculate_scene_bounding_box");

  // determine the bounding box of the scene
  Details::TreeConstruction<DeviceType>::calculateBoundingBoxOfTheScene(
      primitives, _bounds);

  Kokkos::Profiling::popRegion();

  if (size() == 1)
  {
    Kokkos::View<size_t *, DeviceType> permutation_indices("permute", 1);
    Details::TreeConstruction<DeviceType>::initializeLeafNodes(
        primitives, permutation_indices, getLeafNodes());
    return;
  }

  Kokkos::Profiling::pushRegion("ArborX:BVH:assign_morton_codes");

  // calculate morton code of all objects
  Kokkos::View<unsigned int *, DeviceType> morton_indices(
      Kokkos::ViewAllocateWithoutInitializing("morton"), size());
  Details::TreeConstruction<DeviceType>::assignMortonCodes(
      primitives, morton_indices, _bounds);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX:BVH:sort_morton_codes_and_init_leaves");

  // sort them along the Z-order space-filling curve
  auto permutation_indices = Details::sortObjects(morton_indices);
  Details::TreeConstruction<DeviceType>::initializeLeafNodes(
      primitives, permutation_indices, getLeafNodes());

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX:BVH:generate_hierarchy");

  // generate bounding volume hierarchy
  Kokkos::View<int *, DeviceType> parents(
      Kokkos::ViewAllocateWithoutInitializing("parents"), 2 * size() - 1);
  Details::TreeConstruction<DeviceType>::generateHierarchy(
      morton_indices, getLeafNodes(), getInternalNodes(), parents);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX:BVH:calculate_bounding_volumes");

  // calculate bounding volume for each internal node by walking the
  // hierarchy toward the root
  Details::TreeConstruction<DeviceType>::calculateInternalNodesBoundingVolumes(
      getLeafNodes(), getInternalNodes(), parents);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
}

// FIXME not sure where to put these
static_assert(Details::is_expandable<Box, Box>::value, "");
static_assert(Details::is_expandable<Box, Box const>::value, "");
static_assert(Details::is_expandable<Box, Point>::value, "");
static_assert(Details::is_expandable<Box, Point const>::value, "");
static_assert(Details::has_centroid<Box, Point>::value, "");
static_assert(Details::has_centroid<Box const, Point>::value, "");
static_assert(Details::has_centroid<Point, Point>::value, "");
static_assert(Details::has_centroid<Point const, Point>::value, "");

} // namespace ArborX

#endif
