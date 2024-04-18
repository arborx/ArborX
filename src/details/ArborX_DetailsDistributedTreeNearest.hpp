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
#ifndef ARBORX_DETAILS_DISTRIBUTED_TREE_NEAREST_HPP
#define ARBORX_DETAILS_DISTRIBUTED_TREE_NEAREST_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_Box.hpp>
#include <ArborX_DetailsDistributedTreeImpl.hpp>
#include <ArborX_DetailsDistributedTreeUtils.hpp>
#include <ArborX_DetailsHappyTreeFriends.hpp>
#include <ArborX_DetailsKokkosExtKernelStdAlgorithms.hpp>
#include <ArborX_DetailsKokkosExtMinMaxOperations.hpp>
#include <ArborX_DetailsKokkosExtStdAlgorithms.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Point.hpp>
#include <ArborX_Predicates.hpp>
#include <ArborX_Ray.hpp>
#include <ArborX_Sphere.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

// Don't really need it, but our self containment tests rely on its presence
#include <mpi.h>

namespace ArborX
{
namespace Details
{
template <class Predicates, class Distances>
struct WithinDistanceFromPredicates
{
  Predicates predicates;
  Distances distances;
};
} // namespace Details

template <class Predicates, class Distances>
struct AccessTraits<
    Details::WithinDistanceFromPredicates<Predicates, Distances>, PredicatesTag>
{
  using Predicate = typename Predicates::value_type;
  using Geometry =
      std::decay_t<decltype(getGeometry(std::declval<Predicate const &>()))>;
  using Self = Details::WithinDistanceFromPredicates<Predicates, Distances>;

  using memory_space = typename Predicates::memory_space;
  using size_type = decltype(std::declval<Predicates const &>().size());

  static KOKKOS_FUNCTION size_type size(Self const &x)
  {
    return x.predicates.size();
  }
  template <class Dummy = Geometry,
            std::enable_if_t<std::is_same_v<Dummy, Geometry> &&
                             std::is_same_v<Dummy, Point>> * = nullptr>
  static KOKKOS_FUNCTION auto get(Self const &x, size_type i)
  {
    auto const point = getGeometry(x.predicates(i));
    auto const distance = x.distances(i);
    return intersects(Sphere{point, distance});
  }
  template <class Dummy = Geometry,
            std::enable_if_t<std::is_same_v<Dummy, Geometry> &&
                             std::is_same_v<Dummy, Box>> * = nullptr>
  static KOKKOS_FUNCTION auto get(Self const &x, size_type i)
  {
    auto box = getGeometry(x.predicates(i));
    auto &min_corner = box.minCorner();
    auto &max_corner = box.maxCorner();
    auto const distance = x.distances(i);
    for (int d = 0; d < 3; ++d)
    {
      min_corner[d] -= distance;
      max_corner[d] += distance;
    }
    return intersects(box);
  }
  template <class Dummy = Geometry,
            std::enable_if_t<std::is_same_v<Dummy, Geometry> &&
                             std::is_same_v<Dummy, Sphere>> * = nullptr>
  static KOKKOS_FUNCTION auto get(Self const &x, size_type i)
  {
    auto const sphere = getGeometry(x.predicates(i));
    auto const distance = x.distances(i);
    return intersects(Sphere{sphere.centroid(), distance + sphere.radius()});
  }
  template <
      class Dummy = Geometry,
      std::enable_if_t<std::is_same_v<Dummy, Geometry> &&
                       std::is_same_v<Dummy, Experimental::Ray>> * = nullptr>
  static KOKKOS_FUNCTION auto get(Self const &x, size_type i)
  {
    auto const ray = getGeometry(x.predicates(i));
    return intersects(ray);
  }
};

namespace Details
{

template <typename Value>
struct PairValueDistance
{
  Value value;
  float distance;
};

template <typename Tree, bool UseValues>
struct CallbackWithDistance
{
  Tree _tree;

  template <typename ExecutionSpace>
  CallbackWithDistance(ExecutionSpace const &, Tree const &tree)
      : _tree(tree)
  {}

  template <typename Query, typename Value, typename Output>
  KOKKOS_FUNCTION void operator()(Query const &query, Value const &value,
                                  Output const &out) const
  {
    if constexpr (UseValues)
      out({value, distance(getGeometry(query), _tree.indexable_get()(value))});
    else
      out(distance(getGeometry(query), _tree.indexable_get()(value)));
  }
};

template <typename MemorySpace, bool UseValues>
struct CallbackWithDistance<
    BoundingVolumeHierarchy<MemorySpace, Details::LegacyDefaultTemplateValue,
                            Details::DefaultIndexableGetter,
                            ExperimentalHyperGeometry::Box<3, float>>,
    UseValues>
{
  using Tree =
      BoundingVolumeHierarchy<MemorySpace, Details::LegacyDefaultTemplateValue,
                              Details::DefaultIndexableGetter,
                              ExperimentalHyperGeometry::Box<3, float>>;

  Tree _tree;
  Kokkos::View<unsigned int *, typename Tree::memory_space> _rev_permute;

  template <typename ExecutionSpace>
  CallbackWithDistance(ExecutionSpace const &exec_space, Tree const &tree)
      : _tree(tree)
  {
    // NOTE cannot have extended __host__ __device__  lambda in constructor with
    // NVCC
    computeReversePermutation(exec_space);
  }

  template <typename ExecutionSpace>
  void computeReversePermutation(ExecutionSpace const &exec_space)
  {
    auto const n = _tree.size();

    _rev_permute = Kokkos::View<unsigned int *, typename Tree::memory_space>(
        Kokkos::view_alloc(
            Kokkos::WithoutInitializing,
            "ArborX::DistributedTree::query::nearest::reverse_permutation"),
        n);
    if (!_tree.empty())
    {
      Kokkos::parallel_for(
          "ArborX::DistributedTree::query::nearest::"
          "compute_reverse_permutation",
          Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
          KOKKOS_CLASS_LAMBDA(int const i) {
            _rev_permute(HappyTreeFriends::getValue(_tree, i).index) = i;
          });
    }
  }

  template <typename Query, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Query const &query, int index,
                                  OutputFunctor const &out) const
  {
    // TODO: This breaks the abstraction of the distributed Tree not knowing
    // the details of the local tree. Right now, this is the only way. Will
    // need to be fixed with a proper callback abstraction.
    int const leaf_node_index = _rev_permute(index);
    auto const &leaf_node_bounding_volume =
        HappyTreeFriends::getIndexable(_tree, leaf_node_index);
    if constexpr (UseValues)
      out({index, distance(getGeometry(query), leaf_node_bounding_volume)});
    else
      out(distance(getGeometry(query), leaf_node_bounding_volume));
  }
};

template <typename ExecutionSpace, typename Tree, typename Predicates,
          typename Distances>
void DistributedTreeImpl::phaseI(ExecutionSpace const &space, Tree const &tree,
                                 Predicates const &predicates,
                                 Distances &farthest_distances)
{
  std::string prefix = "ArborX::DistributedTree::query::nearest::phaseI";
  Kokkos::Profiling::ScopedRegion guard(prefix);

  using namespace DistributedTree;
  using MemorySpace = typename Tree::memory_space;

  auto comm = tree.getComm();

  // Find the k nearest local trees.
  Kokkos::View<int *, MemorySpace> offset(prefix + "::offset", 0);
  Kokkos::View<int *, MemorySpace> nearest_ranks(prefix + "::nearest_ranks", 0);
  tree._top_tree.query(space, predicates, nearest_ranks, offset);

  // Accumulate total leave count in the local trees until it reaches k which
  // is the number of neighbors queried for.  Stop if local trees get
  // empty because it means that they are no more leaves and there is no point
  // on forwarding predicates to leafless trees.
  auto const n_predicates = predicates.size();
  auto const &bottom_tree_sizes = tree._bottom_tree_sizes;
  Kokkos::View<int *, MemorySpace> new_offset(
      Kokkos::view_alloc(space, offset.label()), n_predicates + 1);
  Kokkos::parallel_for(
      "ArborX::DistributedTree::query::nearest::"
      "bottom_trees_with_required_cumulated_leaves_count",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_predicates),
      KOKKOS_LAMBDA(int i) {
        int leaves_count = 0;
        int const n_nearest_neighbors = getK(predicates(i));
        for (int j = offset(i); j < offset(i + 1); ++j)
        {
          int const bottom_tree_size = bottom_tree_sizes(nearest_ranks(j));
          if ((bottom_tree_size == 0) || (leaves_count >= n_nearest_neighbors))
            break;
          leaves_count += bottom_tree_size;
          ++new_offset(i);
        }
      });

  KokkosExt::exclusive_scan(space, new_offset, new_offset, 0);

  // Truncate results so that predicates will only be forwarded to as many local
  // trees as necessary to find k neighbors.
  Kokkos::View<int *, MemorySpace> new_nearest_ranks(
      Kokkos::view_alloc(space, nearest_ranks.label()),
      KokkosExt::lastElement(space, new_offset));
  Kokkos::parallel_for(
      prefix + "::truncate_before_forwarding",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_predicates),
      KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < new_offset(i + 1) - new_offset(i); ++j)
          new_nearest_ranks(new_offset(i) + j) = nearest_ranks(offset(i) + j);
      });

  offset = new_offset;
  nearest_ranks = new_nearest_ranks;

  auto const &bottom_tree = tree._bottom_tree;
  using BottomTree = std::decay_t<decltype(bottom_tree)>;

  // Gather distances from every identified rank
  Kokkos::View<float *, MemorySpace> distances(prefix + "::distances", 0);
  forwardQueriesAndCommunicateResults(
      comm, space, bottom_tree, predicates,
      CallbackWithDistance<BottomTree, false>(space, bottom_tree),
      nearest_ranks, offset, distances);

  // Postprocess distances to find the k-th farthest
  Kokkos::parallel_for(
      prefix + "::compute_farthest_distances",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, predicates.size()),
      KOKKOS_LAMBDA(int i) {
        auto const num_distances = offset(i + 1) - offset(i);
        if (num_distances == 0)
          return;

        auto const k = KokkosExt::min(getK(predicates(i)), num_distances) - 1;
        auto *begin = distances.data() + offset(i);
        KokkosExt::nth_element(begin, begin + k, begin + num_distances);
        farthest_distances(i) = *(begin + k);
      });
}

template <typename ExecutionSpace, typename Tree, typename Predicates,
          typename Distances, typename Offset, typename Values, typename Ranks>
void DistributedTreeImpl::phaseII(ExecutionSpace const &space, Tree const &tree,
                                  Predicates const &predicates,
                                  Distances &distances, Offset &offset,
                                  Values &values, Ranks &ranks)
{
  std::string prefix = "ArborX::DistributedTree::query::nearest::phaseII";
  Kokkos::Profiling::ScopedRegion guard(prefix);

  using MemorySpace = typename Tree::memory_space;

  Kokkos::View<int *, MemorySpace> nearest_ranks(prefix + "::nearest_ranks", 0);
  tree._top_tree.query(space,
                       WithinDistanceFromPredicates<Predicates, Distances>{
                           predicates, distances},
                       nearest_ranks, offset);

  auto const &bottom_tree = tree._bottom_tree;
  using BottomTree = std::decay_t<decltype(bottom_tree)>;

  // NOTE: in principle, we could perform radius searches on the bottom_tree
  // rather than nearest predicates.
  Kokkos::View<PairValueDistance<typename Values::value_type> *, MemorySpace>
      out(prefix + "::pairs_value_distance", 0);
  DistributedTree::forwardQueriesAndCommunicateResults(
      tree.getComm(), space, bottom_tree, predicates,
      CallbackWithDistance<BottomTree, true>(space, bottom_tree), nearest_ranks,
      offset, out, &ranks);

  // Unzip
  auto n = out.extent(0);
  KokkosExt::reallocWithoutInitializing(space, values, n);
  KokkosExt::reallocWithoutInitializing(space, distances, n);
  Kokkos::parallel_for(
      prefix + "::split_index_distance_pairs",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        values(i) = out(i).value;
        distances(i) = out(i).distance;
      });

  DistributedTree::filterResults(space, predicates, distances, values, offset,
                                 ranks);
}

template <typename Tree, typename ExecutionSpace, typename Predicates,
          typename Values, typename Offset, typename Ranks>
std::enable_if_t<Kokkos::is_view_v<Values> && Kokkos::is_view_v<Offset> &&
                 Kokkos::is_view_v<Ranks>>
DistributedTreeImpl::queryDispatchImpl(NearestPredicateTag, Tree const &tree,
                                       ExecutionSpace const &space,
                                       Predicates const &predicates,
                                       Values &values, Offset &offset,
                                       Ranks &ranks)
{
  std::string prefix = "ArborX::DistributedTree::query::nearest";

  Kokkos::Profiling::ScopedRegion guard(prefix);

  if (tree.empty())
  {
    KokkosExt::reallocWithoutInitializing(space, values, 0);
    KokkosExt::reallocWithoutInitializing(space, offset, predicates.size() + 1);
    Kokkos::deep_copy(space, offset, 0);
    return;
  }

  Kokkos::View<float *, typename Tree::memory_space> farthest_distances(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         prefix + "::farthest_distances"),
      predicates.size());

  // In the first phase, the predicates are sent to as many ranks as necessary
  // to guarantee that all k neighbors queried for are found. The farthest
  // distances are determined to reduce the communication in the second phase.
  phaseI(space, tree, predicates, farthest_distances);

  // In the second phase, predicates are sent again to all ranks that may have a
  // neighbor closer to the farthest neighbor identified in the first pass. It
  // is guaranteed that the nearest k neighbors are within that distance.
  //
  // The current implementation discards the results after the first phase.
  // Everything is recomputed from scratch instead of just searching for
  // potential better neighbors and updating the list.
  phaseII(space, tree, predicates, farthest_distances, offset, values, ranks);
}

template <typename Tree, typename ExecutionSpace, typename Predicates,
          typename Values, typename Offset>
std::enable_if_t<Kokkos::is_view_v<Values> && Kokkos::is_view_v<Offset>>
DistributedTreeImpl::queryDispatch(NearestPredicateTag tag, Tree const &tree,
                                   ExecutionSpace const &space,
                                   Predicates const &predicates, Values &values,
                                   Offset &offset)
{
  Kokkos::View<int *, ExecutionSpace> ranks(
      "ArborX::DistributedTree::query::nearest::ranks", 0);
  queryDispatchImpl(tag, tree, space, predicates, values, offset, ranks);
}

} // namespace Details
} // namespace ArborX

#endif
