/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
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
#include <ArborX_DetailsNode.hpp>       // makeLeafNode
#include <ArborX_DetailsTags.hpp>

#include <Kokkos_Core.hpp>

#include <cassert>

namespace Kokkos
{ // reduction identity must be defined in Kokkos namespace
template <>
struct reduction_identity<ArborX::Box>
{
  KOKKOS_FUNCTION static ArborX::Box sum() { return {}; }
};
} // namespace Kokkos

namespace ArborX
{
namespace Details
{
namespace TreeConstruction
{

template <typename ExecutionSpace, typename Primitives>
inline void calculateBoundingBoxOfTheScene(ExecutionSpace const &space,
                                           Primitives const &primitives,
                                           Box &scene_bounding_box)
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  auto const n = Access::size(primitives);
  Kokkos::parallel_reduce(
      "ArborX::TreeConstruction::calculate_bounding_box_of_the_scene",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
      KOKKOS_LAMBDA(int i, Box &update) {
        update += Access::get(primitives, i);
      },
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
  Kokkos::parallel_for("ArborX::TreeConstruction::assign_morton_codes",
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
      "ArborX::TreeConstruction::assign_morton_codes",
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
  ARBORX_ASSERT(static_cast<decltype(n)>(morton_codes.extent(0)) == n);

  using Tag = typename AccessTraitsHelper<Access>::tag;
  assignMortonCodesDispatch(Tag{}, space, primitives, morton_codes,
                            scene_bounding_box);
}

template <typename ExecutionSpace, typename Primitives, typename Nodes>
inline void initializeSingleLeafNode(ExecutionSpace const &space,
                                     Primitives const &primitives,
                                     Nodes const &leaf_nodes)
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;

  ARBORX_ASSERT(leaf_nodes.extent(0) == 1);
  ARBORX_ASSERT(Access::size(primitives) == 1);

  using Node = typename Nodes::value_type;
  using BoundingVolume = typename Node::bounding_volume_type;

  Kokkos::parallel_for(
      "ArborX::TreeConstruction::initialize_single_leaf",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, 1), KOKKOS_LAMBDA(int) {
        BoundingVolume bounding_volume{};
        expand(bounding_volume, Access::get(primitives, 0));
        leaf_nodes(0) =
            makeLeafNode(typename Node::Tag{}, 0, std::move(bounding_volume));
      });
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

template <typename Primitives, typename MemorySpace, typename Node>
class GenerateHierarchy
{
public:
  template <typename ExecutionSpace,
            typename... PermutationIndicesViewProperties,
            typename... MortonCodesViewProperties,
            typename... LeafNodesViewProperties,
            typename... InternalNodesViewProperties>
  GenerateHierarchy(
      ExecutionSpace const &space, Primitives const &primitives,
      Kokkos::View<unsigned int const *, PermutationIndicesViewProperties...>
          permutation_indices,
      Kokkos::View<unsigned int const *, MortonCodesViewProperties...>
          sorted_morton_codes,
      Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes,
      Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes)
      : _primitives(primitives)
      , _permutation_indices(permutation_indices)
      , _sorted_morton_codes(sorted_morton_codes)
      , _leaf_nodes(leaf_nodes)
      , _internal_nodes(internal_nodes)
      , _ranges(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                   "ArborX::BVH::BVH::ranges"),
                internal_nodes.extent(0))
      , _num_internal_nodes(_internal_nodes.extent_int(0))
  {
    Kokkos::deep_copy(space, _ranges, UNTOUCHED_NODE);

    Kokkos::parallel_for(
        "ArborX::TreeConstruction::generate_hierarchy",
        Kokkos::RangePolicy<ExecutionSpace>(space, _num_internal_nodes,
                                            2 * _num_internal_nodes + 1),
        *this);
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
    return (i < n ? &(_internal_nodes(i)) : &(_leaf_nodes(i - n)));
  }

  template <typename Tag = typename Node::Tag>
  KOKKOS_FUNCTION std::enable_if_t<std::is_same<Tag, NodeWithTwoChildrenTag>{}>
  setRightChild(Node *node, int child_right) const
  {
    assert(!node->isLeaf());
    node->right_child = child_right;
  }

  template <typename Tag = typename Node::Tag>
  KOKKOS_FUNCTION
      std::enable_if_t<std::is_same<Tag, NodeWithLeftChildAndRopeTag>{}>
      setRightChild(Node *node, int) const
  {
    assert(!node->isLeaf());
    (void)node;
  }

  template <typename Tag = typename Node::Tag>
  KOKKOS_FUNCTION std::enable_if_t<std::is_same<Tag, NodeWithTwoChildrenTag>{}>
  setRope(Node *, int, int) const
  {
  }

  template <typename Tag = typename Node::Tag>
  KOKKOS_FUNCTION
      std::enable_if_t<std::is_same<Tag, NodeWithLeftChildAndRopeTag>{}>
      setRope(Node *node, int range_right, int delta_right) const
  {
    int rope;
    if (range_right != _num_internal_nodes)
    {
      // The way Karras indices constructed, the rope is going to be the right
      // child of the first internal node that we are in the left subtree of.
      // The determination of whether that node is internal or leaf requires an
      // additional delta() evaluation.
      rope = range_right + 1;
      if (delta_right < delta(range_right + 1))
        rope += _num_internal_nodes;
    }
    else
    {
      // The node is on the right-most path in the tree. The only reason we
      // need to set it is because nodes may have been allocated without
      // initializing.
      rope = ROPE_SENTINEL;
    }
    node->rope = rope;
  }

  KOKKOS_FUNCTION void operator()(int i) const
  {
    auto const leaf_nodes_shift = _num_internal_nodes;

    // Index in the orginal order primitives were given in.
    auto const original_index = _permutation_indices(i - leaf_nodes_shift);

    using BoundingVolume = typename Node::bounding_volume_type;
    BoundingVolume bounding_volume{};
    using Access = AccessTraits<Primitives, PrimitivesTag>;
    expand(bounding_volume, Access::get(_primitives, original_index));

    // Initialize leaf node
    auto *leaf_node = getNodePtr(i);
    *leaf_node =
        makeLeafNode(typename Node::Tag{}, original_index, bounding_volume);

    // For a leaf node, the range is just one index
    int range_left = i - leaf_nodes_shift;
    int range_right = range_left;

    int delta_left = delta(range_left - 1);
    int delta_right = delta(range_right);

    setRope(leaf_node, range_right, delta_right);

    // Walk toward the root and do process it even though technically its
    // bounding box has already been computed (bounding box of the scene)
    do
    {
      // Determine whether this node is left or right child of its parent
      bool const is_left_child = delta_right < delta_left;

      int left_child;
      int right_child;
      if (is_left_child)
      {
        // The main benefit of the Apetrei index (which is also called a split
        // in the Karras algorithm) is that each child can compute it based
        // just on the child's range. This is different from a Karras index,
        // where the index can only be computed based on the range of the
        // parent, and thus requires knowing the ranges of both children.
        int const apetrei_parent = range_right;

        // The range of the parent is the union of the ranges of children. Each
        // child updates one of these range values, the farthest from the
        // split. The first thread up stores the updated range value (which
        // also serves as a flag). The second thread up finishes constructing
        // the full parent range.
        range_right = Kokkos::atomic_compare_exchange(
            &_ranges(apetrei_parent), UNTOUCHED_NODE, range_left);

        // Use an atomic flag per internal node to terminate the first
        // thread that enters it, while letting the second one through.
        // This ensures that every node gets processed only once, and not
        // before both of its children are processed.
        if (range_right == UNTOUCHED_NODE)
          break;

        // This is slightly convoluted due to the fact that the indices of leaf
        // nodes have to be shifted. The determination whether the other child
        // is a leaf node depends on the position of the split (which is
        // apetrei index) to the range boundary.
        left_child = i;
        right_child = apetrei_parent + 1;
        if (right_child == range_right)
          right_child += leaf_nodes_shift;

        delta_right = delta(range_right);

        expand(bounding_volume, getNodePtr(right_child)->bounding_volume);
      }
      else
      {
        // The comments for this clause are identical to the ones above (in the
        // if clause), and thus ommitted for brevity.

        int const apetrei_parent = range_left - 1;

        range_left = Kokkos::atomic_compare_exchange(
            &_ranges(apetrei_parent), UNTOUCHED_NODE, range_right);
        if (range_left == UNTOUCHED_NODE)
          break;

        left_child = apetrei_parent;
        if (left_child == range_left)
          left_child += leaf_nodes_shift;
        right_child = i;

        delta_left = delta(range_left - 1);

        expand(bounding_volume, getNodePtr(left_child)->bounding_volume);
      }

      // Having the full range for the parent, we can compute the Karras index.
      int const karras_parent =
          delta_right < delta_left ? range_right : range_left;

      auto *parent_node = getNodePtr(karras_parent);
      parent_node->left_child = left_child;
      setRightChild(parent_node, right_child);
      setRope(parent_node, range_right, delta_right);
      parent_node->bounding_volume = bounding_volume;

      i = karras_parent;

    } while (i != 0);
  }

private:
  Primitives _primitives;
  Kokkos::View<unsigned int const *, MemorySpace> _permutation_indices;
  Kokkos::View<unsigned int const *, MemorySpace> _sorted_morton_codes;
  Kokkos::View<Node *, MemorySpace> _leaf_nodes;
  Kokkos::View<Node *, MemorySpace> _internal_nodes;
  Kokkos::View<int *, MemorySpace> _ranges;
  int _num_internal_nodes;
};

template <typename ExecutionSpace, typename Primitives,
          typename... PermutationIndicesViewProperties,
          typename... MortonCodesViewProperties, typename Node,
          typename... LeafNodesViewProperties,
          typename... InternalNodesViewProperties>
void generateHierarchy(
    ExecutionSpace const &space, Primitives const &primitives,
    Kokkos::View<unsigned int *, PermutationIndicesViewProperties...>
        permutation_indices,
    Kokkos::View<unsigned int *, MortonCodesViewProperties...>
        sorted_morton_codes,
    Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes,
    Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes)
{
  using ConstPermutationIndices =
      Kokkos::View<unsigned int const *, PermutationIndicesViewProperties...>;
  using ConstMortonCodes =
      Kokkos::View<unsigned int const *, MortonCodesViewProperties...>;

  using MemorySpace = typename decltype(internal_nodes)::memory_space;

  GenerateHierarchy<Primitives, MemorySpace, Node>(
      space, primitives, ConstPermutationIndices(permutation_indices),
      ConstMortonCodes(sorted_morton_codes), leaf_nodes, internal_nodes);
}

} // namespace TreeConstruction
} // namespace Details
} // namespace ArborX

#endif
