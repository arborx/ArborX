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

#ifndef ARBORX_DETAILS_TREE_CONSTRUCTION_HPP
#define ARBORX_DETAILS_TREE_CONSTRUCTION_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_Box.hpp>
#include <ArborX_DetailsAlgorithms.hpp> // expand
#include <ArborX_DetailsKokkosExt.hpp>  // min, max
#include <ArborX_DetailsMortonCode.hpp> // morton3D
#include <ArborX_DetailsNode.hpp>
#include <ArborX_DetailsTags.hpp>
#include <ArborX_Macros.hpp>

#include <Kokkos_Core.hpp>

#include <cassert>

namespace ArborX
{
namespace Details
{
namespace TreeConstruction
{

template <typename Primitives>
class CalculateBoundingBoxOfTheSceneFunctor
{
public:
  using Access = AccessTraits<Primitives, PrimitivesTag>;

  CalculateBoundingBoxOfTheSceneFunctor(Primitives const &primitives)
      : _primitives(primitives)
  {
  }

  KOKKOS_INLINE_FUNCTION
  void init(Box &box) const { box = Box(); }

  KOKKOS_INLINE_FUNCTION
  void operator()(int const i, Box &box) const
  {
    expand(box, Access::get(_primitives, i));
  }

  KOKKOS_INLINE_FUNCTION
  void join(volatile Box &dst, volatile Box const &src) const
  {
    expand(dst, src);
  }

private:
  Primitives _primitives;
};

template <typename ExecutionSpace, typename Primitives>
inline void calculateBoundingBoxOfTheScene(ExecutionSpace const &space,
                                           Primitives const &primitives,
                                           Box &scene_bounding_box)
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  auto const n = Access::size(primitives);
  Kokkos::parallel_reduce(
      ARBORX_MARK_REGION("calculate_bounding_box_of_the_scene"),
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
      CalculateBoundingBoxOfTheSceneFunctor<Primitives>(primitives),
      scene_bounding_box);
}

template <typename ExecutionSpace, typename Primitives, typename MortonCodes>
inline void assignMortonCodesDispatch(BoxTag, ExecutionSpace const &space,
                                      Primitives const &primitives,
                                      MortonCodes morton_codes,
                                      Box const &scene_bounding_box)
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  auto const n = Access::size(primitives);
  Kokkos::parallel_for(ARBORX_MARK_REGION("assign_morton_codes"),
                       Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
                       KOKKOS_LAMBDA(int i) {
                         Point xyz;
                         centroid(Access::get(primitives, i), xyz);
                         translateAndScale(xyz, xyz, scene_bounding_box);
                         morton_codes(i) = morton3D(xyz[0], xyz[1], xyz[2]);
                       });
}

template <typename ExecutionSpace, typename Primitives, typename MortonCodes>
inline void assignMortonCodesDispatch(PointTag, ExecutionSpace const &space,
                                      Primitives const &primitives,
                                      MortonCodes morton_codes,
                                      Box const &scene_bounding_box)
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  auto const n = Access::size(primitives);
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("assign_morton_codes"),
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        Point xyz;
        translateAndScale(Access::get(primitives, i), xyz, scene_bounding_box);
        morton_codes(i) = morton3D(xyz[0], xyz[1], xyz[2]);
      });
}

template <typename ExecutionSpace, typename Primitives,
          typename... MortonCodesViewProperties>
inline void assignMortonCodes(
    ExecutionSpace const &space, Primitives const &primitives,
    Kokkos::View<unsigned int *, MortonCodesViewProperties...> morton_codes,
    Box const &scene_bounding_box)
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;

  auto const n = Access::size(primitives);
  ARBORX_ASSERT(morton_codes.extent(0) == n);

  using Tag = typename AccessTraitsHelper<Access>::tag;
  assignMortonCodesDispatch(Tag{}, space, primitives, morton_codes,
                            scene_bounding_box);
}

template <typename ExecutionSpace, typename Primitives, typename Indices,
          typename Nodes>
inline void initializeLeafNodesDispatch(BoxTag, ExecutionSpace const &space,
                                        Primitives const &primitives,
                                        Indices permutation_indices,
                                        Nodes leaf_nodes)
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  auto const n = Access::size(primitives);
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("initialize_leaf_nodes"),
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        leaf_nodes(i) =
            makeLeafNode(permutation_indices(i),
                         Access::get(primitives, permutation_indices(i)));
      });
}

template <typename ExecutionSpace, typename Primitives, typename Indices,
          typename Nodes>
inline void initializeLeafNodesDispatch(PointTag, ExecutionSpace const &space,
                                        Primitives const &primitives,
                                        Indices permutation_indices,
                                        Nodes leaf_nodes)
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  auto const n = Access::size(primitives);
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("initialize_leaf_nodes"),
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        leaf_nodes(i) =
            makeLeafNode(permutation_indices(i),
                         {Access::get(primitives, permutation_indices(i)),
                          Access::get(primitives, permutation_indices(i))});
      });
}

template <typename ExecutionSpace, typename Primitives,
          typename... PermutationIndicesViewProperties,
          typename... LeafNodesViewProperties>
inline void initializeLeafNodes(
    ExecutionSpace const &space, Primitives const &primitives,
    Kokkos::View<unsigned int const *, PermutationIndicesViewProperties...>
        permutation_indices,
    Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes)
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;

  auto const n = Access::size(primitives);
  ARBORX_ASSERT(permutation_indices.extent(0) == n);
  ARBORX_ASSERT(leaf_nodes.extent(0) == n);

  using Tag = typename AccessTraitsHelper<Access>::tag;
  initializeLeafNodesDispatch(Tag{}, space, primitives, permutation_indices,
                              leaf_nodes);
}

template <typename ExecutionSpace, typename Primitives,
          typename... PermutationIndicesViewProperties,
          typename... LeafNodesViewProperties>
inline void initializeLeafNodes(
    ExecutionSpace const &space, Primitives const &primitives,
    Kokkos::View<unsigned int *, PermutationIndicesViewProperties...>
        permutation_indices,
    Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes)
{
  initializeLeafNodes(
      space, primitives,
      Kokkos::View<unsigned int const *, PermutationIndicesViewProperties...>{
          permutation_indices},
      leaf_nodes);
}

template <typename MemorySpace>
class GenerateHierarchyFunctor
{
public:
  template <typename ExecutionSpace, typename... MortonCodesViewProperties,
            typename... InternalNodesViewProperties>
  GenerateHierarchyFunctor(
      ExecutionSpace const &space,
      Kokkos::View<unsigned int const *, MortonCodesViewProperties...>
          sorted_morton_codes,
      Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes,
      Kokkos::View<int, MemorySpace> root_node_index)
      : _sorted_morton_codes(sorted_morton_codes)
      , _internal_nodes(internal_nodes)
      , _ranges(Kokkos::ViewAllocateWithoutInitializing("ranges"),
                internal_nodes.extent(0))
      , _root_node_index(root_node_index)
  {
    Kokkos::deep_copy(space, _ranges, -1);
  }

  KOKKOS_FUNCTION
  int delta(int const i) const
  {
    // Per Apetrei:
    //   Because we already know where the highest differing bit is for each
    //   internal node, the delta function basically represents a distance
    //   metric between two keys. Unlike the delta used by Karras, we are
    //   interested in the index of the highest differing bit and not the length
    //   of the common prefix. In practice, logical xor can be used instead of
    //   finding the index of the highest differing bit as we can compare the
    //   numbers. The higher the index of the differing bit, the larger the
    //   number.
    return ((_sorted_morton_codes(i) != _sorted_morton_codes(i + 1))
                ? (_sorted_morton_codes(i) ^ _sorted_morton_codes(i + 1))
                : -i); // FIXME: not sure about this
  }

  KOKKOS_FUNCTION void operator()(int i) const
  {
    auto const n = _internal_nodes.extent_int(0);

    // For a leaf node, the range is just one index
    int range[2] = {i - n, i - n};
    int &range_left = range[0];
    int &range_right = range[1];

    // Walk toward the root and do process it even though technically its
    // bounding box has already been computed (bounding box of the scene)
    do
    {
      // Determine the parent and set parent->child connection
      int parent_index =
          (range_left == 0 ||
           (range_right != n && delta(range_right) < delta(range_left - 1)));
      bool is_left_child = parent_index == 1;

      // parent = range_left - 1 or parent = range_right
      // to avoid if/else, do this calculation
      int parent = range[parent_index] - (1 - parent_index);
      Node *node = &_internal_nodes(parent);

      if (is_left_child)
        node->children.first = i;
      else
        node->children.second = i;

      // Replace one of the boundaries with the one coming from the parent
      // It also serves as an atomic check whether a thread should continue
      range[parent_index] = Kokkos::atomic_compare_exchange(
          &_ranges(parent), -1, range[1 - parent_index]);

      // Use an atomic flag per internal node to terminate the first
      // thread that enters it, while letting the second one through.
      // This ensures that every node gets processed only once, and not
      // before both of its children are processed.
      if (range[parent_index] == -1)
        break;

      // Internal node bounding boxes are unitialized hence the
      // assignment operator below.
      // FIXME: accessing Node::bounding_box is not ideal but I was
      // reluctant to pass the bounding volume hierarchy to
      // generateHierarchy()
      Node const *first_child = &_internal_nodes(node->children.first);
      Node const *second_child = &_internal_nodes(node->children.second);
      node->bounding_box = first_child->bounding_box;
      expand(node->bounding_box, second_child->bounding_box);

      i = parent;

      if (range_right - range_left == n)
      {
        // root node
        _root_node_index() = i;
        break;
      }
    } while (true);

    // NOTE: could check that bounding box of the root node is indeed the
    // union of the two children.
  }

private:
  Kokkos::View<unsigned int const *, MemorySpace> _sorted_morton_codes;
  Kokkos::View<Node *, MemorySpace> _internal_nodes;
  // Use int instead of bool because CAS (Compare And Swap) on CUDA does not
  // support boolean
  Kokkos::View<int *, MemorySpace> _ranges;
  Kokkos::View<int, MemorySpace> _root_node_index;
};

template <typename ExecutionSpace, typename... MortonCodesViewProperties,
          typename... InternalNodesViewProperties>
int generateHierarchy(
    ExecutionSpace const &space,
    Kokkos::View<unsigned int const *, MortonCodesViewProperties...>
        sorted_morton_codes,
    Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes)
{
  using MemorySpace = typename decltype(internal_nodes)::memory_space;
  auto const n_internal_nodes = internal_nodes.extent(0);
  Kokkos::View<int, MemorySpace> root_node_index("root_node");

  Kokkos::parallel_for(
      ARBORX_MARK_REGION("generate_hierarchy"),
      Kokkos::RangePolicy<ExecutionSpace>(space, n_internal_nodes,
                                          2 * n_internal_nodes + 1),
      GenerateHierarchyFunctor<MemorySpace>(space, sorted_morton_codes,
                                            internal_nodes, root_node_index));
  auto root_node_index_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, root_node_index);

  return root_node_index_host();
}

template <typename ExecutionSpace, typename... MortonCodesViewProperties,
          typename... InternalNodesViewProperties>
int generateHierarchy(
    ExecutionSpace const &space,
    Kokkos::View<unsigned int *, MortonCodesViewProperties...>
        sorted_morton_codes,
    Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes)
{
  return generateHierarchy(
      space,
      Kokkos::View<unsigned int const *, MortonCodesViewProperties...>{
          sorted_morton_codes},
      internal_nodes);
}

} // namespace TreeConstruction
} // namespace Details
} // namespace ArborX

#endif
