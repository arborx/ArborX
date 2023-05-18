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

#ifndef ARBORX_DETAILS_TREE_CONSTRUCTION_HPP
#define ARBORX_DETAILS_TREE_CONSTRUCTION_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsAlgorithms.hpp> // expand
#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_DetailsNode.hpp> // makeLeafNode
#include <ArborX_SpaceFillingCurves.hpp>

#include <Kokkos_Core.hpp>

#include <cassert>

namespace ArborX
{
namespace Details
{
namespace TreeConstruction
{

template <typename ExecutionSpace, typename Primitives, typename Box>
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
      Kokkos::Sum<Box>{scene_bounding_box});
}

template <typename ExecutionSpace, typename Primitives,
          typename SpaceFillingCurve, typename Box, typename LinearOrdering>
inline void projectOntoSpaceFillingCurve(ExecutionSpace const &space,
                                         Primitives const &primitives,
                                         SpaceFillingCurve const &curve,
                                         Box const &scene_bounding_box,
                                         LinearOrdering linear_ordering_indices)
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;

  size_t const n = Access::size(primitives);
  ARBORX_ASSERT(linear_ordering_indices.extent(0) == n);
  static_assert(
      std::is_same<typename LinearOrdering::value_type,
                   decltype(curve(scene_bounding_box,
                                  Access::get(primitives, 0)))>::value);

  Kokkos::parallel_for(
      "ArborX::TreeConstruction::project_primitives_onto_space_filling_curve",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        linear_ordering_indices(i) =
            curve(scene_bounding_box, Access::get(primitives, i));
      });
}

template <typename ExecutionSpace, typename Primitives, typename Nodes>
inline void initializeSingleLeafNode(ExecutionSpace const &space,
                                     Primitives const &primitives,
                                     Nodes const &leaf_nodes)
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  using Value = typename Nodes::value_type::value_type;
  using BoundingVolume = decltype(std::declval<Value>().bounding_volume);

  ARBORX_ASSERT(leaf_nodes.extent(0) == 1);
  ARBORX_ASSERT(Access::size(primitives) == 1);

  Kokkos::parallel_for(
      "ArborX::TreeConstruction::initialize_single_leaf",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, 1), KOKKOS_LAMBDA(int) {
        BoundingVolume bounding_volume{};
        expand(bounding_volume, Access::get(primitives, 0));
        leaf_nodes(0) = makeLeafNode(Value{(unsigned)0, bounding_volume});
      });
}

template <typename Primitives, typename PermutationIndices,
          typename LinearOrdering, typename LeafNodes, typename InternalNodes>
class GenerateHierarchy
{
  static constexpr int UNTOUCHED_NODE = -1;

  using MemorySpace = typename LeafNodes::memory_space;
  using LinearOrderingValueType = typename LinearOrdering::non_const_value_type;

public:
  template <typename ExecutionSpace>
  GenerateHierarchy(ExecutionSpace const &space, Primitives const &primitives,
                    PermutationIndices const &permutation_indices,
                    LinearOrdering const &sorted_morton_codes,
                    LeafNodes leaf_nodes, InternalNodes internal_nodes)
      : _primitives(primitives)
      , _permutation_indices(permutation_indices)
      , _sorted_morton_codes(sorted_morton_codes)
      , _leaf_nodes(leaf_nodes)
      , _internal_nodes(internal_nodes)
      , _ranges(Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                                   "ArborX::BVH::BVH::ranges"),
                internal_nodes.extent(0))
      , _num_internal_nodes(_internal_nodes.extent_int(0))
  {
    Kokkos::deep_copy(space, _ranges, UNTOUCHED_NODE);

    Kokkos::parallel_for(
        "ArborX::TreeConstruction::generate_hierarchy",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, leaf_nodes.extent(0)),
        *this);
  }

  using DeltaValueType = std::make_signed_t<LinearOrderingValueType>;

  KOKKOS_FUNCTION
  auto internalIndex(int const i) const { return i + _num_internal_nodes + 1; }

  KOKKOS_FUNCTION
  DeltaValueType delta(int const i) const
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

    constexpr auto max_value =
        KokkosExt::ArithmeticTraits::finite_max<DeltaValueType>::value;
    constexpr auto min_value =
        KokkosExt::ArithmeticTraits::finite_min<DeltaValueType>::value;

    // This check is here simply to avoid code complications in the main
    // operator
    if (i < 0 || i >= _num_internal_nodes)
      return max_value;

    // The Apetrei's paper does not mention dealing with duplicate indices. We
    // follow the original Karras idea in this situation:
    //   The case of duplicate Morton codes has to be handled explicitly, since
    //   our construction algorithm relies on the keys being unique. We
    //   accomplish this by augmenting each key with a bit representation of
    //   its index, i.e. k_i = k_i <+> i, where <+> indicates string
    //   concatenation.
    // In this case, if the Morton indices are the same, we want to compare is.
    // We also want the result in this situation to always be less than any
    // Morton comparison. Thus, we add LLONG_MIN to it.
    auto const x = _sorted_morton_codes(i) ^ _sorted_morton_codes(i + 1);

    return x + (!x) * (min_value + (i ^ (i + 1))) - 1;
    //                                            ^^^
    // When using 63 bits for Morton codes, the LLONG_MAX is actually a valid
    // code. As we want the return statement above to return a value always
    // greater than anything here, we downshift by 1.
  }

  template <typename Node>
  KOKKOS_FUNCTION void setRope(Node &node, int range_right,
                               DeltaValueType delta_right) const
  {
    int rope;
    if (range_right != _num_internal_nodes)
    {
      // The way Karras indices constructed, the rope is going to be the right
      // child of the first internal node that we are in the left subtree of.
      // The determination of whether that node is internal or leaf requires an
      // additional delta() evaluation.
      rope = (delta_right < delta(range_right + 1)
                  ? range_right + 1
                  : internalIndex(range_right + 1));
    }
    else
    {
      // The node is on the right-most path in the tree. The only reason we
      // need to set it is that nodes may have been allocated without
      // initializing.
      rope = ROPE_SENTINEL;
    }
    node.rope = rope;
  }

  KOKKOS_FUNCTION void operator()(int i) const
  {
    // Index in the original order primitives were given in.
    auto const original_index = _permutation_indices(i);

    using BoundingVolume =
        typename InternalNodes::value_type::bounding_volume_type;
    BoundingVolume bounding_volume{};
    using Access = AccessTraits<Primitives, PrimitivesTag>;
    expand(bounding_volume, Access::get(_primitives, original_index));

    // Initialize leaf node
    using Value = typename LeafNodes::value_type::value_type;
    auto &leaf_node = _leaf_nodes(i);
    leaf_node = makeLeafNode(Value{original_index, bounding_volume});

    // For a leaf node, the range is just one index
    int range_left = i;
    int range_right = i;

    auto delta_left = delta(range_left - 1);
    auto delta_right = delta(range_right);

    setRope(leaf_node, range_right, delta_right);

    // Walk toward the root and do process it even though technically its
    // bounding box has already been computed (bounding box of the scene)
    auto const root = internalIndex(0);
    do
    {
      // Determine whether this node is left or right child of its parent
      bool const is_left_child = (delta_right < delta_left);

      int left_child;
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
        int right_child = apetrei_parent + 1;
        bool const right_child_is_leaf = (right_child == range_right);

        delta_right = delta(range_right);

        // Memory synchronization below ensures write from other threads to the
        // child bounding volume memory location are visible from the current
        // thread.
        // NOTE we need acquire semantics at the device scope
        Kokkos::load_fence();
        expand(bounding_volume,
               right_child_is_leaf
                   ? _leaf_nodes(right_child).value.bounding_volume
                   : _internal_nodes(right_child).bounding_volume);
      }
      else
      {
        // The comments for this clause are identical to the ones above (in the
        // if clause), and thus omitted for brevity.

        int const apetrei_parent = range_left - 1;

        range_left = Kokkos::atomic_compare_exchange(
            &_ranges(apetrei_parent), UNTOUCHED_NODE, range_right);
        if (range_left == UNTOUCHED_NODE)
          break;

        left_child = apetrei_parent;
        bool const left_child_is_leaf = (left_child == range_left);

        delta_left = delta(range_left - 1);

        Kokkos::load_fence();
        expand(bounding_volume,
               left_child_is_leaf
                   ? _leaf_nodes(left_child).value.bounding_volume
                   : _internal_nodes(left_child).bounding_volume);

        if (!left_child_is_leaf)
          left_child = internalIndex(left_child);
      }

      // Having the full range for the parent, we can compute the Karras index.
      int const karras_parent =
          delta_right < delta_left ? range_right : range_left;

      auto &parent_node = _internal_nodes(karras_parent);
      parent_node.left_child = left_child;
      setRope(parent_node, range_right, delta_right);
      parent_node.bounding_volume = bounding_volume;

      i = internalIndex(karras_parent);
    } while (i != root);
  }

private:
  Primitives _primitives;
  PermutationIndices _permutation_indices;
  LinearOrdering _sorted_morton_codes;
  LeafNodes _leaf_nodes;
  InternalNodes _internal_nodes;
  Kokkos::View<int *, MemorySpace> _ranges;
  int _num_internal_nodes;
};

template <typename ExecutionSpace, typename Primitives,
          typename... PermutationIndicesViewProperties,
          typename LinearOrderingValueType,
          typename... LinearOrderingViewProperties, typename LeafNodes,
          typename InternalNodes>
void generateHierarchy(
    ExecutionSpace const &space, Primitives const &primitives,
    Kokkos::View<unsigned int *, PermutationIndicesViewProperties...>
        permutation_indices,
    Kokkos::View<LinearOrderingValueType *, LinearOrderingViewProperties...>
        sorted_morton_codes,
    LeafNodes leaf_nodes, InternalNodes internal_nodes)
{
  using ConstPermutationIndices =
      Kokkos::View<unsigned int const *, PermutationIndicesViewProperties...>;
  using ConstLinearOrdering = Kokkos::View<LinearOrderingValueType const *,
                                           LinearOrderingViewProperties...>;

  GenerateHierarchy(
      space, primitives, ConstPermutationIndices(permutation_indices),
      ConstLinearOrdering(sorted_morton_codes), leaf_nodes, internal_nodes);
}

} // namespace TreeConstruction
} // namespace Details
} // namespace ArborX

#endif
