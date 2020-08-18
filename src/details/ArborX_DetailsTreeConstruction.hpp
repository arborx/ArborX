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

namespace
{
// Ideally, this would be
//     static int constexpr UNTOUCHED_NODE = -1;
// inside the GenerateHierachyFunctor class. But prior to C++17, this would
// require to also have a definition outside of the class as it is odr-used.
// This is a workaround.
int constexpr UNTOUCHED_NODE = -1;
} // namespace

template <typename MemorySpace>
class GenerateHierarchyFunctor
{
public:
  template <typename ExecutionSpace, typename... MortonCodesViewProperties,
            typename... LeafNodesViewProperties,
            typename... InternalNodesViewProperties>
  GenerateHierarchyFunctor(
      ExecutionSpace const &space,
      Kokkos::View<unsigned int const *, MortonCodesViewProperties...>
          sorted_morton_codes,
      Kokkos::View<Node const *, LeafNodesViewProperties...> leaf_nodes,
      Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes)
      : _sorted_morton_codes(sorted_morton_codes)
      , _leaf_nodes(leaf_nodes)
      , _internal_nodes(internal_nodes)
      , _ranges(Kokkos::ViewAllocateWithoutInitializing("ranges"),
                internal_nodes.extent(0))
      , _num_internal_nodes(_internal_nodes.extent_int(0))
  {
    Kokkos::deep_copy(space, _ranges, UNTOUCHED_NODE);
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

    // This check is here simply to avoid code complications in the main
    // operator
    if (i < 0 || i >= _num_internal_nodes)
      return INT_MAX;

    // The Apetrei's paper does not mention dealing with duplicate indices. We
    // follow the original Karras idea in this situation:
    //   The case of duplicate Morton codes has to be handled explicitly, since
    //   our construction algorithm relies on the keys being unique. We
    //   accomplish this by augmenting each key with a bit representation of
    //   its index, i.e. k_i = k_i <+> i, where <+> indicates string
    //   concatenation.
    // In this case, if the Morton indices are the same, we want to compare is.
    // We also want the result in this situation to always be less than any
    // Morton comparison. Thus, we add INT_MIN to it.
    // We also avoid if/else statement by doing a "x + !x*<blah>" trick.
    auto x = _sorted_morton_codes(i) ^ _sorted_morton_codes(i + 1);
    return x + (!x) * (INT_MIN + (i ^ (i + 1)));
  }

  KOKKOS_FUNCTION Node *getNodePtr(int i) const
  {
    int const n = _num_internal_nodes;
    return i >= n ? const_cast<Node *>(&(_leaf_nodes(i - n)))
                  : &(_internal_nodes(i));
  }

  KOKKOS_FUNCTION void operator()(int i) const
  {
    auto const leaf_nodes_shift = _num_internal_nodes;

    Box bbox = getNodePtr(i)->bounding_box;

    int range_left = i - leaf_nodes_shift;
    int range_right = i - leaf_nodes_shift;
    int delta_left = delta(range_left - 1);
    int delta_right = delta(range_right);

    do
    {
      bool const is_left_child = delta_right < delta_left;
      if (is_left_child)
      {
        int parent = range_right;

        range_right = Kokkos::atomic_compare_exchange(
            &_ranges(parent), UNTOUCHED_NODE, range_left);
        if (range_right == UNTOUCHED_NODE)
          break;

        int other_child = parent + 1;

        if (other_child == range_right)
          other_child += leaf_nodes_shift;

        delta_right = delta(range_right);

        parent = delta_right < delta_left ? range_right : range_left;

        auto *parent_node = getNodePtr(parent);
        parent_node->children.first = i;
        parent_node->children.second = other_child;
        expand(bbox, getNodePtr(other_child)->bounding_box);
        parent_node->bounding_box = bbox;

        i = parent;
      }
      else
      {
        int parent = range_left - 1;

        range_left = Kokkos::atomic_compare_exchange(
            &_ranges(parent), UNTOUCHED_NODE, range_right);
        if (range_left == UNTOUCHED_NODE)
          break;

        int other_child = parent;

        if (other_child == range_left)
          other_child += leaf_nodes_shift;

        delta_left = delta(range_left - 1);

        parent = delta_right < delta_left ? range_right : range_left;

        auto *parent_node = getNodePtr(parent);
        parent_node->children.first = other_child;
        parent_node->children.second = i;
        expand(bbox, getNodePtr(other_child)->bounding_box);
        parent_node->bounding_box = bbox;

        i = parent;
      }
    } while (i != 0);
  }

private:
  Kokkos::View<unsigned int const *, MemorySpace> _sorted_morton_codes;
  Kokkos::View<Node const *, MemorySpace> _leaf_nodes;
  Kokkos::View<Node *, MemorySpace> _internal_nodes;
  Kokkos::View<int *, MemorySpace> _ranges;
  int _num_internal_nodes;
}; // namespace TreeConstruction

template <typename ExecutionSpace, typename... MortonCodesViewProperties,
          typename... LeafNodesViewProperties,
          typename... InternalNodesViewProperties>
void generateHierarchy(
    ExecutionSpace const &space,
    Kokkos::View<unsigned int const *, MortonCodesViewProperties...>
        sorted_morton_codes,
    Kokkos::View<Node const *, LeafNodesViewProperties...> leaf_nodes,
    Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes)
{
  using MemorySpace = typename decltype(internal_nodes)::memory_space;
  auto const n_internal_nodes = internal_nodes.extent(0);

  Kokkos::parallel_for(
      ARBORX_MARK_REGION("generate_hierarchy"),
      Kokkos::RangePolicy<ExecutionSpace>(space, n_internal_nodes,
                                          2 * n_internal_nodes + 1),
      GenerateHierarchyFunctor<MemorySpace>(space, sorted_morton_codes,
                                            leaf_nodes, internal_nodes));
}

template <typename ExecutionSpace, typename... MortonCodesViewProperties,
          typename... LeafNodesViewProperties,
          typename... InternalNodesViewProperties>
void generateHierarchy(
    ExecutionSpace const &space,
    Kokkos::View<unsigned int *, MortonCodesViewProperties...>
        sorted_morton_codes,
    Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes,
    Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes)
{
  generateHierarchy(
      space,
      Kokkos::View<unsigned int const *, MortonCodesViewProperties...>{
          sorted_morton_codes},
      Kokkos::View<Node const *, LeafNodesViewProperties...>{leaf_nodes},
      internal_nodes);
}

} // namespace TreeConstruction
} // namespace Details
} // namespace ArborX

#endif
