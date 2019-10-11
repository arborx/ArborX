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

// FIXME provides definition of Kokkos::Iterate for Kokkos_CopyViews.hpp
#include <KokkosExp_MDRangePolicy.hpp>
#include <Kokkos_Atomic.hpp>
#include <Kokkos_CopyViews.hpp> // deep_copy
#include <Kokkos_Macros.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

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
  using ExecutionSpace = typename DeviceType::execution_space;

  template <typename Primitives>
  static void calculateBoundingBoxOfTheScene(Primitives const &primitives,
                                             Box &scene_bounding_box);

  // to assign the Morton code for a given object, we use the centroid point
  // of its bounding box, and express it relative to the bounding box of the
  // scene.
  template <typename Primitives>
  static void
  assignMortonCodes(Primitives const &primitives,
                    Kokkos::View<unsigned int *, DeviceType> morton_codes,
                    Box const &scene_bounding_box);

  template <typename Primitives>
  static void initializeLeafNodes(
      Primitives const &primitives,
      Kokkos::View<size_t const *, DeviceType> permutation_indices,
      Kokkos::View<Node *, DeviceType> leaf_nodes);

  static Node *generateHierarchy(
      Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes,
      Kokkos::View<Node *, DeviceType> leaf_nodes,
      Kokkos::View<Node *, DeviceType> internal_nodes,
      Kokkos::View<int *, DeviceType> parents);

  static void calculateInternalNodesBoundingVolumes(
      Kokkos::View<Node const *, DeviceType> leaf_nodes,
      Kokkos::View<Node *, DeviceType> internal_nodes,
      Kokkos::View<int const *, DeviceType> parents);

  KOKKOS_FUNCTION
  static int commonPrefix(Kokkos::View<unsigned int *, DeviceType> morton_codes,
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

  KOKKOS_FUNCTION
  static int
  findSplit(Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes,
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

  KOKKOS_FUNCTION
  static Kokkos::pair<int, int>
  determineRange(Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes,
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
template <typename Primitives>
inline void TreeConstruction<DeviceType>::calculateBoundingBoxOfTheScene(
    Primitives const &primitives, Box &scene_bounding_box)
{
  using Access = typename Traits::Access<Primitives, Traits::PrimitivesTag>;
  auto const n = Access::size(primitives);
  Kokkos::parallel_reduce(
      ARBORX_MARK_REGION("calculate_bounding_box_of_the_scene"),
      Kokkos::RangePolicy<ExecutionSpace>(0, n),
      CalculateBoundingBoxOfTheSceneFunctor<Primitives>(primitives),
      scene_bounding_box);
  ExecutionSpace().fence();
}

template <typename Primitives, typename MortonCodes>
inline void assignMortonCodesDispatch(BoxTag, Primitives const &primitives,
                                      MortonCodes morton_codes,
                                      Box const &scene_bounding_box)
{
  using Access = typename Traits::Access<Primitives, Traits::PrimitivesTag>;
  using ExecutionSpace = typename Access::memory_space::execution_space;
  auto const n = Access::size(primitives);
  Kokkos::parallel_for(ARBORX_MARK_REGION("assign_morton_codes"),
                       Kokkos::RangePolicy<ExecutionSpace>(0, n),
                       KOKKOS_LAMBDA(int i) {
                         Point xyz;
                         centroid(Access::get(primitives, i), xyz);
                         translateAndScale(xyz, xyz, scene_bounding_box);
                         morton_codes(i) = morton3D(xyz[0], xyz[1], xyz[2]);
                       });
  ExecutionSpace().fence();
}

template <typename Primitives, typename MortonCodes>
inline void assignMortonCodesDispatch(PointTag, Primitives const &primitives,
                                      MortonCodes morton_codes,
                                      Box const &scene_bounding_box)
{
  using Access = typename Traits::Access<Primitives, Traits::PrimitivesTag>;
  using ExecutionSpace = typename Access::memory_space::execution_space;
  auto const n = Access::size(primitives);
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("assign_morton_codes"),
      Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
        Point xyz;
        translateAndScale(Access::get(primitives, i), xyz, scene_bounding_box);
        morton_codes(i) = morton3D(xyz[0], xyz[1], xyz[2]);
      });
  ExecutionSpace().fence();
}

template <typename DeviceType>
template <typename Primitives>
inline void TreeConstruction<DeviceType>::assignMortonCodes(
    Primitives const &primitives,
    Kokkos::View<unsigned int *, DeviceType> morton_codes,
    Box const &scene_bounding_box)
{
  using Access = typename Traits::Access<Primitives, Traits::PrimitivesTag>;

  auto const n = Access::size(primitives);
  ARBORX_ASSERT(morton_codes.extent(0) == n);

  using Tag = typename Tag<decay_result_of_get_t<Access>>::type;
  assignMortonCodesDispatch(Tag{}, primitives, morton_codes,
                            scene_bounding_box);
}

template <typename Primitives, typename Indices, typename Nodes>
inline void initializeLeafNodesDispatch(BoxTag, Primitives const &primitives,
                                        Indices permutation_indices,
                                        Nodes leaf_nodes)
{
  using Access = typename Traits::Access<Primitives, Traits::PrimitivesTag>;
  using ExecutionSpace = typename Access::memory_space::execution_space;
  auto const n = Access::size(primitives);
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("initialize_leaf_nodes"),
      Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
        leaf_nodes(i) = {
            {nullptr, reinterpret_cast<Node *>(permutation_indices(i))},
            Access::get(primitives, permutation_indices(i))};
      });
  ExecutionSpace().fence();
}

template <typename Primitives, typename Indices, typename Nodes>
inline void initializeLeafNodesDispatch(PointTag, Primitives const &primitives,
                                        Indices permutation_indices,
                                        Nodes leaf_nodes)
{
  using Access = typename Traits::Access<Primitives, Traits::PrimitivesTag>;
  using ExecutionSpace = typename Access::memory_space::execution_space;
  auto const n = Access::size(primitives);
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("initialize_leaf_nodes"),
      Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
        leaf_nodes(i) = {
            {nullptr, reinterpret_cast<Node *>(permutation_indices(i))},
            {Access::get(primitives, permutation_indices(i)),
             Access::get(primitives, permutation_indices(i))}};
      });
  ExecutionSpace().fence();
}

template <typename DeviceType>
template <typename Primitives>
inline void TreeConstruction<DeviceType>::initializeLeafNodes(
    Primitives const &primitives,
    Kokkos::View<size_t const *, DeviceType> permutation_indices,
    Kokkos::View<Node *, DeviceType> leaf_nodes)
{
  using Access = typename Traits::Access<Primitives, Traits::PrimitivesTag>;

  auto const n = Access::size(primitives);
  ARBORX_ASSERT(permutation_indices.extent(0) == n);
  ARBORX_ASSERT(leaf_nodes.extent(0) == n);

  static_assert(sizeof(typename decltype(permutation_indices)::value_type) ==
                    sizeof(Node *),
                "Encoding leaf index in pointer to child is not safe if the "
                "index and pointer types do not have the same size");

  using Tag = typename Tag<decay_result_of_get_t<Access>>::type;
  initializeLeafNodesDispatch(Tag{}, primitives, permutation_indices,
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
      , _shift(internal_nodes.extent(0))
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
      _internal_nodes(i).children.first = &_leaf_nodes(split);
      _parents(split + _shift) = i;
    }
    else
    {
      _internal_nodes(i).children.first = &_internal_nodes(split);
      _parents(split) = i;
    }

    // Select second child and record parent-child relationship.

    if (split + 1 == last)
    {
      _internal_nodes(i).children.second = &_leaf_nodes(split + 1);
      _parents(split + 1 + _shift) = i;
    }
    else
    {
      _internal_nodes(i).children.second = &_internal_nodes(split + 1);
      _parents(split + 1) = i;
    }
  }

private:
  Kokkos::View<unsigned int *, DeviceType> _sorted_morton_codes;
  Kokkos::View<Node *, DeviceType> _leaf_nodes;
  Kokkos::View<Node *, DeviceType> _internal_nodes;
  Kokkos::View<int *, DeviceType> _parents;
  int _shift;
};

template <typename DeviceType>
Node *TreeConstruction<DeviceType>::generateHierarchy(
    Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes,
    Kokkos::View<Node *, DeviceType> leaf_nodes,
    Kokkos::View<Node *, DeviceType> internal_nodes,
    Kokkos::View<int *, DeviceType> parents)
{
  auto const n = sorted_morton_codes.extent(0);
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("generate_hierarchy"),
      Kokkos::RangePolicy<ExecutionSpace>(0, n - 1),
      GenerateHierarchyFunctor<DeviceType>(sorted_morton_codes, leaf_nodes,
                                           internal_nodes, parents));
  ExecutionSpace().fence();
  // returns a pointer to the root node of the tree
  return internal_nodes.data();
}

template <typename DeviceType>
class CalculateInternalNodesBoundingVolumesFunctor
{
public:
  CalculateInternalNodesBoundingVolumesFunctor(
      Node *root, Kokkos::View<int const *, DeviceType> parents,
      size_t n_internal_nodes)
      : _root(root)
      , _flags(Kokkos::ViewAllocateWithoutInitializing("flags"),
               n_internal_nodes)
      , _parents(parents)
  {
    // Initialize flags to zero
    Kokkos::deep_copy(_flags, 0);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(int const i) const
  {
    Node *node = _root + _parents(i);
    // Walk toward the root and do process it even though technically its
    // bounding box has already been computed (bounding box of the scene)
    while (true)
    {
      // Use an atomic flag per internal node to terminate the first
      // thread that enters it, while letting the second one through.
      // This ensures that every node gets processed only once, and not
      // before both of its children are processed.
      if (Kokkos::atomic_compare_exchange_strong(&_flags(node - _root), 0, 1))
        break;

      // Internal node bounding boxes are unitialized hence the
      // assignment operator below.
      // FIXME: accessing Node::bounding_box is not ideal but I was
      // reluctant to pass the bounding volume hierarchy to
      // generateHierarchy()
      node->bounding_box = (node->children.first)->bounding_box;
      expand(node->bounding_box, (node->children.second)->bounding_box);
      if (node == _root)
        break;

      node = _root + _parents(node - _root);
    }
    // NOTE: could check that bounding box of the root node is indeed the
    // union of the two children.
  }

private:
  Node *_root;
  // Use int instead of bool because CAS (Compare And Swap) on CUDA does not
  // support boolean
  Kokkos::View<int *, DeviceType> _flags;
  Kokkos::View<int const *, DeviceType> _parents;
};

template <typename DeviceType>
void TreeConstruction<DeviceType>::calculateInternalNodesBoundingVolumes(
    Kokkos::View<Node const *, DeviceType> leaf_nodes,
    Kokkos::View<Node *, DeviceType> internal_nodes,
    Kokkos::View<int const *, DeviceType> parents)
{
  auto const first = internal_nodes.extent(0);
  auto const last = first + leaf_nodes.extent(0);
  Node *root = internal_nodes.data();
  Kokkos::parallel_for(ARBORX_MARK_REGION("calculate_bounding_boxes"),
                       Kokkos::RangePolicy<ExecutionSpace>(first, last),
                       CalculateInternalNodesBoundingVolumesFunctor<DeviceType>(
                           root, parents, first));
  ExecutionSpace().fence();
}

} // namespace Details
} // namespace ArborX

#endif
