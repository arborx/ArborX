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

#include <ArborX_Box.hpp>
#include <ArborX_DetailsAlgorithms.hpp> // expand
#include <ArborX_DetailsConcepts.hpp>   // decay_result_of_get_t
#include <ArborX_DetailsKokkosExt.hpp>  // clz, min, max, sgn
#include <ArborX_DetailsMortonCode.hpp> // morton3D
#include <ArborX_DetailsNode.hpp>
#include <ArborX_DetailsTags.hpp>
#include <ArborX_Macros.hpp>
#include <ArborX_Traits.hpp>

#include <Kokkos_Core.hpp>

#include <cassert>

namespace ArborX
{
namespace Details
{

/**
 * This structure contains all the functions used to build the BVH. All the
 * functions are static.
 */
template <typename DeviceType>
struct TreeConstruction
{
public:
  template <typename ExecutionSpace, typename Primitives>
  static void calculateBoundingBoxOfTheScene(ExecutionSpace const &space,
                                             Primitives const &primitives,
                                             Box &scene_bounding_box);

  // to assign the Morton code for a given object, we use the centroid point
  // of its bounding box, and express it relative to the bounding box of the
  // scene.
  template <typename ExecutionSpace, typename Primitives,
            typename... MortonCodesViewProperties>
  static void assignMortonCodes(
      ExecutionSpace const &space, Primitives const &primitives,
      Kokkos::View<unsigned int *, MortonCodesViewProperties...> morton_codes,
      Box const &scene_bounding_box);

  template <typename ExecutionSpace, typename Primitives,
            typename... PermutationIndicesViewProperties,
            typename... LeafNodesViewProperties>
  static void initializeLeafNodes(
      ExecutionSpace const &space, Primitives const &primitives,
      Kokkos::View<size_t const *, PermutationIndicesViewProperties...>
          permutation_indices,
      Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes);

  template <typename ExecutionSpace, typename Primitives,
            typename... PermutationIndicesViewProperties,
            typename... LeafNodesViewProperties>
  static void initializeLeafNodes(
      ExecutionSpace const &space, Primitives const &primitives,
      Kokkos::View<size_t *, PermutationIndicesViewProperties...>
          permutation_indices,
      Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes)
  {
    initializeLeafNodes(
        space, primitives,
        Kokkos::View<size_t const *, PermutationIndicesViewProperties...>{
            permutation_indices},
        leaf_nodes);
  }

  template <typename ExecutionSpace, typename... MortonCodesViewProperties,
            typename... LeafNodesViewProperties,
            typename... InternalNodesViewProperties,
            typename... ParentsViewProperties>
  static Node *generateHierarchy(
      ExecutionSpace const &space,
      Kokkos::View<unsigned int *, MortonCodesViewProperties...>
          sorted_morton_codes,
      Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes,
      Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes,
      Kokkos::View<int *, ParentsViewProperties...> parents);

  template <typename ExecutionSpace, typename... LeafNodesViewProperties,
            typename... InternalNodesViewProperties,
            typename... ParentsViewProperties>
  static void calculateInternalNodesBoundingVolumes(
      ExecutionSpace const &space,
      Kokkos::View<Node const *, LeafNodesViewProperties...> leaf_nodes,
      Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes,
      Kokkos::View<int const *, ParentsViewProperties...> parents);

  template <typename ExecutionSpace, typename... LeafNodesViewProperties,
            typename... InternalNodesViewProperties,
            typename... ParentsViewProperties>
  static void calculateInternalNodesBoundingVolumes(
      ExecutionSpace const &space,
      Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes,
      Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes,
      Kokkos::View<int *, ParentsViewProperties...> parents)
  {
    calculateInternalNodesBoundingVolumes(
        space,
        Kokkos::View<Node const *, LeafNodesViewProperties...>{leaf_nodes},
        internal_nodes,
        Kokkos::View<int const *, ParentsViewProperties...>{parents});
  }

  template <typename... MortonCodesViewProperties>
  KOKKOS_FUNCTION static int commonPrefix(
      Kokkos::View<unsigned int *, MortonCodesViewProperties...> morton_codes,
      int i, int j)
  {
    using KokkosExt::clz;

    int const n = morton_codes.extent(0);
    if (j < 0 || j > n - 1)
      return -1;

    // our construction algorithm relies on keys being unique so we handle
    // explicitly case of duplicate Morton codes by augmenting each key by
    // a bit representation of its index.
    if (morton_codes[i] == morton_codes[j])
    {
      // clz( k[i] ^ k[j] ) == 32
      return 32 + clz(i ^ j);
    }
    return clz(morton_codes[i] ^ morton_codes[j]);
  }

  template <typename... MortonCodesViewProperties>
  KOKKOS_FUNCTION static int
  findSplit(Kokkos::View<unsigned int *, MortonCodesViewProperties...>
                sorted_morton_codes,
            int first, int last)
  {
    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.

    int common_prefix = commonPrefix(sorted_morton_codes, first, last);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.

    int split = first; // initial guess
    int step = last - first;

    do
    {
      step = (step + 1) >> 1;       // exponential decrease
      int new_split = split + step; // proposed new position

      if (new_split < last)
      {
        if (commonPrefix(sorted_morton_codes, first, new_split) > common_prefix)
          split = new_split; // accept proposal
      }
    } while (step > 1);

    return split;
  }

  template <typename... MortonCodesViewProperties>
  KOKKOS_FUNCTION static Kokkos::pair<int, int>
  determineRange(Kokkos::View<unsigned int *, MortonCodesViewProperties...>
                     sorted_morton_codes,
                 int i)
  {
    using KokkosExt::max;
    using KokkosExt::min;
    using KokkosExt::sgn;

    // determine direction of the range (+1 or -1)
    int direction = sgn(commonPrefix(sorted_morton_codes, i, i + 1) -
                        commonPrefix(sorted_morton_codes, i, i - 1));
    assert(direction == +1 || direction == -1);

    // compute upper bound for the length of the range
    int max_step = 2;
    int common_prefix = commonPrefix(sorted_morton_codes, i, i - direction);
    while (commonPrefix(sorted_morton_codes, i, i + direction * max_step) >
           common_prefix)
    {
      max_step = max_step << 1;
    }

    // find the other end using binary search
    int split = 0;
    int step = max_step;
    do
    {
      step = step >> 1;
      if (commonPrefix(sorted_morton_codes, i, i + (split + step) * direction) >
          common_prefix)
        split += step;
    } while (step > 1);
    int j = i + split * direction;

    return {min(i, j), max(i, j)};
  }
};

template <typename Primitives>
class CalculateBoundingBoxOfTheSceneFunctor
{
public:
  using Access = typename Traits::Access<Primitives, Traits::PrimitivesTag>;

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

template <typename DeviceType>
template <typename ExecutionSpace, typename Primitives>
inline void TreeConstruction<DeviceType>::calculateBoundingBoxOfTheScene(
    ExecutionSpace const &space, Primitives const &primitives,
    Box &scene_bounding_box)
{
  using Access = typename Traits::Access<Primitives, Traits::PrimitivesTag>;
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
  using Access = typename Traits::Access<Primitives, Traits::PrimitivesTag>;
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
  using Access = typename Traits::Access<Primitives, Traits::PrimitivesTag>;
  auto const n = Access::size(primitives);
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("assign_morton_codes"),
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        Point xyz;
        translateAndScale(Access::get(primitives, i), xyz, scene_bounding_box);
        morton_codes(i) = morton3D(xyz[0], xyz[1], xyz[2]);
      });
}

template <typename DeviceType>
template <typename ExecutionSpace, typename Primitives,
          typename... MortonCodesViewProperties>
inline void TreeConstruction<DeviceType>::assignMortonCodes(
    ExecutionSpace const &space, Primitives const &primitives,
    Kokkos::View<unsigned int *, MortonCodesViewProperties...> morton_codes,
    Box const &scene_bounding_box)
{
  using Access = typename Traits::Access<Primitives, Traits::PrimitivesTag>;

  auto const n = Access::size(primitives);
  ARBORX_ASSERT(morton_codes.extent(0) == n);

  using Tag = typename Tag<decay_result_of_get_t<Access>>::type;
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
  using Access = typename Traits::Access<Primitives, Traits::PrimitivesTag>;
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
  using Access = typename Traits::Access<Primitives, Traits::PrimitivesTag>;
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

template <typename DeviceType>
template <typename ExecutionSpace, typename Primitives,
          typename... PermutationIndicesViewProperties,
          typename... LeafNodesViewProperties>
inline void TreeConstruction<DeviceType>::initializeLeafNodes(
    ExecutionSpace const &space, Primitives const &primitives,
    Kokkos::View<size_t const *, PermutationIndicesViewProperties...>
        permutation_indices,
    Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes)
{
  using Access = typename Traits::Access<Primitives, Traits::PrimitivesTag>;

  auto const n = Access::size(primitives);
  ARBORX_ASSERT(permutation_indices.extent(0) == n);
  ARBORX_ASSERT(leaf_nodes.extent(0) == n);

  using Tag = typename Tag<decay_result_of_get_t<Access>>::type;
  initializeLeafNodesDispatch(Tag{}, space, primitives, permutation_indices,
                              leaf_nodes);
}

template <typename DeviceType>
class GenerateHierarchyFunctor
{
public:
  GenerateHierarchyFunctor(
      Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes,
      Kokkos::View<Node *, DeviceType> leaf_nodes,
      Kokkos::View<Node *, DeviceType> internal_nodes,
      Kokkos::View<int *, DeviceType> parents)
      : _sorted_morton_codes(sorted_morton_codes)
      , _leaf_nodes(leaf_nodes)
      , _internal_nodes(internal_nodes)
      , _parents(parents)
      , _leaf_nodes_shift(internal_nodes.extent(0))
  {
  }

  // from "Thinking Parallel, Part III: Tree Construction on the GPU" by
  // Karras
  KOKKOS_INLINE_FUNCTION
  void operator()(int const i) const
  {
    // Construct internal nodes.
    // Find out which range of objects the node corresponds to.
    // (This is where the magic happens!)

    auto range =
        TreeConstruction<DeviceType>::determineRange(_sorted_morton_codes, i);
    int first = range.first;
    int last = range.second;

    // Determine where to split the range.

    int split = TreeConstruction<DeviceType>::findSplit(_sorted_morton_codes,
                                                        first, last);

    // Select first child and record parent-child relationship.

    if (split == first)
    {
      _internal_nodes(i).children.first = split + _leaf_nodes_shift;
      _parents(split + _leaf_nodes_shift) = i;
    }
    else
    {
      _internal_nodes(i).children.first = split;
      _parents(split) = i;
    }

    // Select second child and record parent-child relationship.

    if (split + 1 == last)
    {
      _internal_nodes(i).children.second = split + 1 + _leaf_nodes_shift;
      _parents(split + 1 + _leaf_nodes_shift) = i;
    }
    else
    {
      _internal_nodes(i).children.second = split + 1;
      _parents(split + 1) = i;
    }
  }

private:
  Kokkos::View<unsigned int *, DeviceType> _sorted_morton_codes;
  Kokkos::View<Node *, DeviceType> _leaf_nodes;
  Kokkos::View<Node *, DeviceType> _internal_nodes;
  Kokkos::View<int *, DeviceType> _parents;
  int _leaf_nodes_shift;
};

template <typename DeviceType>
template <typename ExecutionSpace, typename... MortonCodesViewProperties,
          typename... LeafNodesViewProperties,
          typename... InternalNodesViewProperties,
          typename... ParentsViewProperties>
Node *TreeConstruction<DeviceType>::generateHierarchy(
    ExecutionSpace const &space,
    Kokkos::View<unsigned int *, MortonCodesViewProperties...>
        sorted_morton_codes,
    Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes,
    Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes,
    Kokkos::View<int *, ParentsViewProperties...> parents)
{
  auto const n = sorted_morton_codes.extent(0);
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("generate_hierarchy"),
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n - 1),
      GenerateHierarchyFunctor<DeviceType>(sorted_morton_codes, leaf_nodes,
                                           internal_nodes, parents));
  // returns a pointer to the root node of the tree
  return internal_nodes.data();
}

template <typename DeviceType>
class CalculateInternalNodesBoundingVolumesFunctor
{
public:
  CalculateInternalNodesBoundingVolumesFunctor(
      Kokkos::View<Node *, DeviceType> internal_and_leaf_nodes,
      Kokkos::View<int const *, DeviceType> parents, size_t n_internal_nodes)
      : _flags(Kokkos::ViewAllocateWithoutInitializing("flags"),
               n_internal_nodes)
      , _parents(parents)
      , _internal_and_leaf_nodes(internal_and_leaf_nodes)
  {
    // Initialize flags to zero
    Kokkos::deep_copy(_flags, 0);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const
  {
    // Walk toward the root and do process it even though technically its
    // bounding box has already been computed (bounding box of the scene)
    while (true)
    {
      i = _parents(i);

      // Use an atomic flag per internal node to terminate the first
      // thread that enters it, while letting the second one through.
      // This ensures that every node gets processed only once, and not
      // before both of its children are processed.
      if (Kokkos::atomic_compare_exchange_strong(&_flags(i), 0, 1))
        break;

      // Internal node bounding boxes are unitialized hence the
      // assignment operator below.
      // FIXME: accessing Node::bounding_box is not ideal but I was
      // reluctant to pass the bounding volume hierarchy to
      // generateHierarchy()
      Node *node = &_internal_and_leaf_nodes(i);
      Node const *first_child = &_internal_and_leaf_nodes(node->children.first);
      Node const *second_child =
          &_internal_and_leaf_nodes(node->children.second);
      node->bounding_box = first_child->bounding_box;
      expand(node->bounding_box, second_child->bounding_box);
      if (i == 0) // root node
        break;
    }
    // NOTE: could check that bounding box of the root node is indeed the
    // union of the two children.
  }

private:
  // Use int instead of bool because CAS (Compare And Swap) on CUDA does not
  // support boolean
  Kokkos::View<int *, DeviceType> _flags;
  Kokkos::View<int const *, DeviceType> _parents;
  Kokkos::View<Node *, DeviceType> _internal_and_leaf_nodes;
};

template <typename DeviceType>
template <typename ExecutionSpace, typename... LeafNodesViewProperties,
          typename... InternalNodesViewProperties,
          typename... ParentsViewProperties>
void TreeConstruction<DeviceType>::calculateInternalNodesBoundingVolumes(
    ExecutionSpace const &space,
    Kokkos::View<Node const *, LeafNodesViewProperties...> leaf_nodes,
    Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes,
    Kokkos::View<int const *, ParentsViewProperties...> parents)
{
  auto const first = internal_nodes.extent(0);
  auto const last = first + leaf_nodes.extent(0);
  Kokkos::View<Node *, InternalNodesViewProperties...,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      internal_and_leaf_nodes(internal_nodes.data(), last);
  Kokkos::parallel_for(ARBORX_MARK_REGION("calculate_bounding_boxes"),
                       Kokkos::RangePolicy<ExecutionSpace>(space, first, last),
                       CalculateInternalNodesBoundingVolumesFunctor<DeviceType>(
                           internal_and_leaf_nodes, parents, first));
}

} // namespace Details
} // namespace ArborX

#endif
