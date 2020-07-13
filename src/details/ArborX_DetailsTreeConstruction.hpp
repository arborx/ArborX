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
#include <ArborX_DetailsKokkosExt.hpp>  // clz, min, max, sgn
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
template <typename... MortonCodesViewProperties>
KOKKOS_FUNCTION int
commonPrefix(Kokkos::View<unsigned int const *, MortonCodesViewProperties...>
                 morton_codes,
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
KOKKOS_INLINE_FUNCTION auto commonPrefix(
    Kokkos::View<unsigned int *, MortonCodesViewProperties...> morton_codes,
    int i, int j)
{
  return commonPrefix(
      Kokkos::View<unsigned int const *, MortonCodesViewProperties...>{
          morton_codes},
      i, j);
}

template <typename... MortonCodesViewProperties>
KOKKOS_FUNCTION int
findSplit(Kokkos::View<unsigned int const *, MortonCodesViewProperties...>
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
KOKKOS_INLINE_FUNCTION auto
findSplit(Kokkos::View<unsigned int *, MortonCodesViewProperties...>
              sorted_morton_codes,
          int first, int last)
{
  return findSplit(
      Kokkos::View<unsigned int const *, MortonCodesViewProperties...>{
          sorted_morton_codes},
      first, last);
}

template <typename... MortonCodesViewProperties>
KOKKOS_FUNCTION Kokkos::pair<int, int>
determineRange(Kokkos::View<unsigned int const *, MortonCodesViewProperties...>
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

template <typename... MortonCodesViewProperties>
KOKKOS_INLINE_FUNCTION auto
determineRange(Kokkos::View<unsigned int *, MortonCodesViewProperties...>
                   sorted_morton_codes,
               int i)
{
  return determineRange(
      Kokkos::View<unsigned int const *, MortonCodesViewProperties...>{
          sorted_morton_codes},
      i);
}

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
  using Access = AccessTraits<Primitives, Traits::PrimitivesTag>;
  using MemorySpace = typename Access::memory_space;
  auto const n = Access::size(primitives);
  Kokkos::View<Point *, MemorySpace> centroids(
      Kokkos::ViewAllocateWithoutInitializing("centroids"), n);
  Kokkos::parallel_for(ARBORX_MARK_REGION("compute_centroids"),
                       Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
                       KOKKOS_LAMBDA(int i) {
                         centroid(Access::get(primitives, i), centroids(i));
                       });
  auto centroids_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, centroids);
  auto morton_codes_host =
      Kokkos::create_mirror_view(Kokkos::HostSpace{}, morton_codes);
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("assign_morton_codes"),
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n),
      KOKKOS_LAMBDA(int i) {
        morton_codes_host(i) =
            flecsi_hilbert_proj(scene_bounding_box, centroids_host(i));
      });
  Kokkos::deep_copy(morton_codes, morton_codes_host);
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
  template <typename... MortonCodesViewProperties,
            typename... LeafNodesViewProperties,
            typename... InternalNodesViewProperties,
            typename... ParentsViewProperties>
  GenerateHierarchyFunctor(
      Kokkos::View<unsigned int const *, MortonCodesViewProperties...>
          sorted_morton_codes,
      Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes,
      Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes,
      Kokkos::View<int *, ParentsViewProperties...> parents)
      : _sorted_morton_codes(sorted_morton_codes)
      , _leaf_nodes(leaf_nodes)
      , _internal_nodes(internal_nodes)
      , _parents(parents)
      , _leaf_nodes_shift(internal_nodes.extent(0))
  {
  }

  // from "Thinking Parallel, Part III: Tree Construction on the GPU" by Karras
  KOKKOS_FUNCTION void operator()(int i) const
  {
    using TreeConstruction::determineRange;
    using TreeConstruction::findSplit;

    // Construct internal nodes.
    // Find out which range of objects the node corresponds to.
    // (This is where the magic happens!)

    auto range = determineRange(_sorted_morton_codes, i);
    int first = range.first;
    int last = range.second;

    // Determine where to split the range.

    int split = findSplit(_sorted_morton_codes, first, last);

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
  Kokkos::View<unsigned int const *, MemorySpace> _sorted_morton_codes;
  Kokkos::View<Node *, MemorySpace> _leaf_nodes;
  Kokkos::View<Node *, MemorySpace> _internal_nodes;
  Kokkos::View<int *, MemorySpace> _parents;
  int _leaf_nodes_shift;
};

template <typename ExecutionSpace, typename... MortonCodesViewProperties,
          typename... LeafNodesViewProperties,
          typename... InternalNodesViewProperties,
          typename... ParentsViewProperties>
Node *generateHierarchy(
    ExecutionSpace const &space,
    Kokkos::View<unsigned int const *, MortonCodesViewProperties...>
        sorted_morton_codes,
    Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes,
    Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes,
    Kokkos::View<int *, ParentsViewProperties...> parents)
{
  using MemorySpace = typename decltype(leaf_nodes)::memory_space;
  auto const n = sorted_morton_codes.extent(0);
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("generate_hierarchy"),
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n - 1),
      GenerateHierarchyFunctor<MemorySpace>(sorted_morton_codes, leaf_nodes,
                                            internal_nodes, parents));
  // returns a pointer to the root node of the tree
  return internal_nodes.data();
}

template <typename ExecutionSpace, typename... MortonCodesViewProperties,
          typename... LeafNodesViewProperties,
          typename... InternalNodesViewProperties,
          typename... ParentsViewProperties>
inline Node *generateHierarchy(
    ExecutionSpace const &space,
    Kokkos::View<unsigned int *, MortonCodesViewProperties...>
        sorted_morton_codes,
    Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes,
    Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes,
    Kokkos::View<int *, ParentsViewProperties...> parents)
{
  return generateHierarchy(
      space,
      Kokkos::View<unsigned int const *, MortonCodesViewProperties...>{
          sorted_morton_codes},
      leaf_nodes, internal_nodes, parents);
}

template <typename MemorySpace>
class CalculateInternalNodesBoundingVolumesFunctor
{
public:
  template <typename ExecutionSpace, typename... NodesViewProperties,
            typename... ParentsViewProperties>
  CalculateInternalNodesBoundingVolumesFunctor(
      ExecutionSpace const &space,
      Kokkos::View<Node *, NodesViewProperties...> internal_and_leaf_nodes,
      Kokkos::View<int const *, ParentsViewProperties...> parents,
      size_t n_internal_nodes)
      : _flags(Kokkos::ViewAllocateWithoutInitializing("flags"),
               n_internal_nodes)
      , _parents(parents)
      , _internal_and_leaf_nodes(internal_and_leaf_nodes)
  {
    // Initialize flags to zero
    Kokkos::deep_copy(space, _flags, 0);
  }

  KOKKOS_FUNCTION
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
  Kokkos::View<int *, MemorySpace> _flags;
  Kokkos::View<int const *, MemorySpace> _parents;
  Kokkos::View<Node *, MemorySpace> _internal_and_leaf_nodes;
};

template <typename ExecutionSpace, typename... LeafNodesViewProperties,
          typename... InternalNodesViewProperties,
          typename... ParentsViewProperties>
void calculateInternalNodesBoundingVolumes(
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
  using MemorySpace = typename decltype(leaf_nodes)::memory_space;
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("calculate_bounding_boxes"),
      Kokkos::RangePolicy<ExecutionSpace>(space, first, last),
      CalculateInternalNodesBoundingVolumesFunctor<MemorySpace>(
          space, internal_and_leaf_nodes, parents, first));
}

template <typename ExecutionSpace, typename... LeafNodesViewProperties,
          typename... InternalNodesViewProperties,
          typename... ParentsViewProperties>
inline void calculateInternalNodesBoundingVolumes(
    ExecutionSpace const &space,
    Kokkos::View<Node *, LeafNodesViewProperties...> leaf_nodes,
    Kokkos::View<Node *, InternalNodesViewProperties...> internal_nodes,
    Kokkos::View<int *, ParentsViewProperties...> parents)
{
  calculateInternalNodesBoundingVolumes(
      space, Kokkos::View<Node const *, LeafNodesViewProperties...>{leaf_nodes},
      internal_nodes,
      Kokkos::View<int const *, ParentsViewProperties...>{parents});
}
} // namespace TreeConstruction
} // namespace Details
} // namespace ArborX

#endif
