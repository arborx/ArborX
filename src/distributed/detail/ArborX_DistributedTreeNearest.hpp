/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_DISTRIBUTED_TREE_NEAREST_HPP
#define ARBORX_DISTRIBUTED_TREE_NEAREST_HPP

#include <detail/ArborX_DistributedTreeImpl.hpp>
#include <detail/ArborX_DistributedTreeNearestHelpers.hpp>
#include <detail/ArborX_DistributedTreeUtils.hpp>
#include <detail/ArborX_HappyTreeFriends.hpp>
#include <detail/ArborX_Predicates.hpp>
#include <kokkos_ext/ArborX_KokkosExtKernelStdAlgorithms.hpp>
#include <kokkos_ext/ArborX_KokkosExtStdAlgorithms.hpp>
#include <kokkos_ext/ArborX_KokkosExtViewHelpers.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

// Don't really need it, but our self containment tests rely on its presence
#include <mpi.h>

namespace ArborX::Details
{

template <typename Value, typename Coordinate>
struct PairValueDistance
{
  Value value;
  Coordinate distance;
};

template <typename ExecutionSpace, typename Tree,
          Concepts::Predicates Predicates, typename Distances>
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
  tree._top_tree.query(space, predicates, DistributedTree::IndexOnlyCallback{},
                       nearest_ranks, offset);

  // Accumulate total leave count in the local trees until it reaches k which
  // is the number of neighbors queried for.  Stop if local trees get
  // empty because it means that there are no more leaves and there is no point
  // in forwarding predicates to leafless trees.
  auto const n_predicates = predicates.size();
  auto const &bottom_tree_sizes = tree._bottom_tree_sizes;
  Kokkos::View<int *, MemorySpace> new_offset(
      Kokkos::view_alloc(space, offset.label()), n_predicates + 1);
  Kokkos::parallel_for(
      "ArborX::DistributedTree::query::nearest::"
      "bottom_trees_with_required_cumulated_leaves_count",
      Kokkos::RangePolicy(space, 0, n_predicates), KOKKOS_LAMBDA(int i) {
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
      Kokkos::RangePolicy(space, 0, n_predicates), KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < new_offset(i + 1) - new_offset(i); ++j)
          new_nearest_ranks(new_offset(i) + j) = nearest_ranks(offset(i) + j);
      });

  offset = new_offset;
  nearest_ranks = new_nearest_ranks;

  auto const &bottom_tree = tree._bottom_tree;
  using BottomTree = std::decay_t<decltype(bottom_tree)>;
  using Coordinate = typename Distances::value_type;

  // Gather distances from every identified rank
  Kokkos::View<Coordinate *, MemorySpace> distances(prefix + "::distances", 0);
  forwardQueriesAndCommunicateResults(
      comm, space, bottom_tree, predicates,
      CallbackWithDistance<BottomTree, DefaultCallback, Coordinate, false>(
          space, bottom_tree, DefaultCallback{}),
      nearest_ranks, offset, distances);

  // Postprocess distances to find the k-th farthest
  Kokkos::parallel_for(
      prefix + "::compute_farthest_distances",
      Kokkos::RangePolicy(space, 0, predicates.size()), KOKKOS_LAMBDA(int i) {
        auto const num_distances = offset(i + 1) - offset(i);
        if (num_distances == 0)
          return;

        auto const k = Kokkos::min(getK(predicates(i)), num_distances) - 1;
        auto *begin = distances.data() + offset(i);
        KokkosExt::nth_element(begin, begin + k, begin + num_distances);
        farthest_distances(i) = *(begin + k);
      });
}

template <typename ExecutionSpace, typename Tree,
          Concepts::Predicates Predicates, typename Callback,
          typename Distances, typename Offset, typename Values>
void DistributedTreeImpl::phaseII(ExecutionSpace const &space, Tree const &tree,
                                  Predicates const &predicates,
                                  Callback const &callback,
                                  Distances &distances, Offset &offset,
                                  Values &values)
{
  std::string prefix = "ArborX::DistributedTree::query::nearest::phaseII";
  Kokkos::Profiling::ScopedRegion guard(prefix);

  using MemorySpace = typename Tree::memory_space;

  Kokkos::View<int *, MemorySpace> nearest_ranks(prefix + "::nearest_ranks", 0);
  tree._top_tree.query(space,
                       WithinDistanceFromPredicates<Predicates, Distances>{
                           predicates, distances},
                       DistributedTree::IndexOnlyCallback{}, nearest_ranks,
                       offset);

  auto const &bottom_tree = tree._bottom_tree;
  using BottomTree = std::decay_t<decltype(bottom_tree)>;
  using Coordinate = typename Distances::value_type;

  // NOTE: in principle, we could perform radius searches on the bottom_tree
  // rather than nearest predicates.
  Kokkos::View<PairValueDistance<typename Values::value_type, Coordinate> *,
               MemorySpace>
      out(prefix + "::pairs_value_distance", 0);
  DistributedTree::forwardQueriesAndCommunicateResults(
      tree.getComm(), space, bottom_tree, predicates,
      CallbackWithDistance<BottomTree, Callback, typename Values::value_type,
                           true>(space, bottom_tree, callback),
      nearest_ranks, offset, out);

  // Unzip
  auto n = out.extent(0);
  KokkosExt::reallocWithoutInitializing(space, values, n);
  KokkosExt::reallocWithoutInitializing(space, distances, n);
  Kokkos::parallel_for(
      prefix + "::split_index_distance_pairs", Kokkos::RangePolicy(space, 0, n),
      KOKKOS_LAMBDA(int i) {
        values(i) = out(i).value;
        distances(i) = out(i).distance;
      });

  DistributedTree::filterResults(space, predicates, distances, values, offset);
}

template <typename Tree, typename ExecutionSpace,
          Concepts::Predicates Predicates, typename Callback, typename Values,
          typename Offset>
void DistributedTreeImpl::queryDispatch2RoundImpl(
    NearestPredicateTag, Tree const &tree, ExecutionSpace const &space,
    Predicates const &predicates, Callback const &callback, Values &values,
    Offset &offset)
{
  std::string prefix = "ArborX::DistributedTree::query::nearest";

  Kokkos::Profiling::ScopedRegion guard(prefix);

  static_assert(is_constrained_callback_v<Callback>);

  if (tree.empty())
  {
    KokkosExt::reallocWithoutInitializing(space, values, 0);
    KokkosExt::reallocWithoutInitializing(space, offset, predicates.size() + 1);
    Kokkos::deep_copy(space, offset, 0);
    return;
  }

  // Set the type for the distances to be that of the distance to a leaf node.
  // It is possible that that is a higher precision compared to internal nodes,
  // but it safer.
  using Coordinate = decltype(predicates(0).distance(
      HappyTreeFriends::getIndexable(tree._bottom_tree, 0)));

  Kokkos::View<Coordinate *, typename Tree::memory_space> farthest_distances(
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
  phaseII(space, tree, predicates, callback, farthest_distances, offset,
          values);
}

template <typename Tree, typename ExecutionSpace,
          Concepts::Predicates Predicates, typename Values, typename Offset>
std::enable_if_t<Kokkos::is_view_v<Values> && Kokkos::is_view_v<Offset>>
DistributedTreeImpl::queryDispatch(NearestPredicateTag tag, Tree const &tree,
                                   ExecutionSpace const &space,
                                   Predicates const &predicates, Values &values,
                                   Offset &offset)
{
  queryDispatch2RoundImpl(tag, tree, space, predicates, DefaultCallback{},
                          values, offset);
}

template <typename Tree, typename ExecutionSpace,
          Concepts::Predicates Predicates, typename Callback, typename Values,
          typename Offset>
std::enable_if_t<Kokkos::is_view_v<Values> && Kokkos::is_view_v<Offset>>
DistributedTreeImpl::queryDispatch(NearestPredicateTag, Tree const &tree,
                                   ExecutionSpace const &space,
                                   Predicates const &predicates,
                                   Callback const &callback, Values &values,
                                   Offset &offset)
{
  if constexpr (is_constrained_callback_v<Callback>)
  {
    queryDispatch2RoundImpl(NearestPredicateTag{}, tree, space, predicates,
                            callback, values, offset);
  }
  else
  {
    Kokkos::abort("3-arg callback not implemented yet.");
  }
}

} // namespace ArborX::Details

#endif
