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
#include <kokkos_ext/ArborX_KokkosExtMinMaxReduce.hpp>
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
          Details::Concepts::Predicates Predicates, typename Distances>
void DistributedTreeImpl::phaseI(ExecutionSpace const &space, Tree const &tree,
                                 Predicates const &predicates,
                                 Distances &farthest_distances)
{
  std::string prefix = "ArborX::DistributedTree::query::nearest::phaseI";
  Kokkos::Profiling::ScopedRegion guard(prefix);

  using namespace DistributedTree;
  using MemorySpace = typename Tree::memory_space;

  auto comm = tree.getComm();

  // Find the k nearest local trees
  Kokkos::View<int *, MemorySpace> offset(prefix + "::offset", 0);
  Kokkos::View<int *, MemorySpace> nearest_ranks(prefix + "::nearest_ranks", 0);
  tree._top_tree.query(space, predicates,
                       Experimental::ExtractPairIndexCallback{}, nearest_ranks,
                       offset);

  // Accumulate total leave count in the local trees until it reaches k which
  // is the number of neighbors queried for
  {
    auto const n_predicates = predicates.size();
    auto const &bottom_tree_sizes = tree._bottom_tree_sizes;
    Kokkos::View<int *, MemorySpace> new_offset(
        Kokkos::view_alloc(space, offset.label()), n_predicates + 1);
    auto new_nearest_ranks =
        KokkosExt::cloneWithoutInitializingNorCopying(space, nearest_ranks);
    // Initialize num_new_nearest_ranks to work around the fact that
    // Kokkos::parallel_scan does not initialize it if n_predicates = 0.
    int num_new_nearest_ranks = 0;
    Kokkos::parallel_scan(
        prefix + "::bottom_trees_with_required_accumulated_leaves_count",
        Kokkos::RangePolicy(space, 0, n_predicates),
        KOKKOS_LAMBDA(int i, int &update, bool final_pass) {
          auto const k = getK(predicates(i));
          for (int j = offset(i), leaves_count = 0;
               j < offset(i + 1) && leaves_count < k; ++j, ++update)
          {
            auto const rank = nearest_ranks(j);
            leaves_count += bottom_tree_sizes(rank);
            if (final_pass)
              new_nearest_ranks(update) = rank;
          }
          if (final_pass)
            new_offset(i + 1) = update;
        },
        num_new_nearest_ranks);
    Kokkos::resize(Kokkos::view_alloc(space, Kokkos::WithoutInitializing),
                   new_nearest_ranks, num_new_nearest_ranks);
    nearest_ranks = new_nearest_ranks;
    offset = new_offset;
  }

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
          Details::Concepts::Predicates Predicates, typename Callback,
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
                       Experimental::ExtractPairIndexCallback{}, nearest_ranks,
                       offset);

  auto const &bottom_tree = tree._bottom_tree;
  using BottomTree = std::decay_t<decltype(bottom_tree)>;
  using Coordinate = typename Distances::value_type;

  // NOTE: in principle, we could perform radius searches on the bottom_tree
  // rather than nearest predicates.
  Kokkos::View<PairValueDistance<typename Values::value_type, Coordinate> *,
               MemorySpace>
      out(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             prefix + "::pairs_value_distance"),
          0);
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
          Details::Concepts::Predicates Predicates, typename Callback,
          typename Values, typename Offset>
void DistributedTreeImpl::queryDispatch2RoundImpl(
    NearestPredicateTag, Tree const &tree, ExecutionSpace const &space,
    Predicates const &predicates, Callback const &callback, Values &values,
    Offset &offset)
{
  std::string prefix = "ArborX::DistributedTree::query::nearest_2round";

  Kokkos::Profiling::ScopedRegion guard(prefix);

  static_assert(is_constrained_callback_v<Callback>);

  if (tree.empty())
  {
    KokkosExt::reallocWithoutInitializing(space, offset, predicates.size() + 1);
    Kokkos::deep_copy(space, offset, 0);
    KokkosExt::reallocWithoutInitializing(space, values, 0);
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

struct DefaultCallbackWithRank
{
  int _rank;

  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const &, Value const &value,
                                  OutputFunctor const &out) const
  {
    out({value, _rank});
  }
};

template <typename Query, typename Value>
struct TupleQueryIndexValue
{
  Query query;
  Value value;
  int query_id;
  int rank;
};

template <typename Tree, typename ExecutionSpace,
          Details::Concepts::Predicates Predicates, typename Callback,
          typename Values, typename Offset>
void DistributedTreeImpl::queryDispatch3RoundImpl(
    NearestPredicateTag, Tree const &tree, ExecutionSpace const &space,
    Predicates const &predicates, Callback const &callback, Values &values,
    Offset &offset)
{
  std::string prefix = "ArborX::DistributedTree::query::nearest_3round";
  Kokkos::Profiling::ScopedRegion guard(prefix);
  prefix += "::";

  KOKKOS_ASSERT(!tree.empty());

  using MemorySpace = typename Tree::memory_space;
  using Query = typename Predicates::value_type;
  using Value = typename Tree::value_type;

  auto comm = tree.getComm();
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  int const n_predicates = predicates.size();

  Kokkos::Profiling::pushRegion(prefix + "get_values_and_ranks");

  // Find the ranks and index values for each predicate
  Kokkos::View<int *, MemorySpace> ranks(prefix + "ranks", 0);
  Kokkos::View<Value *, MemorySpace> index_values(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, prefix + "index_values"),
      0);
  Kokkos::View<PairValueIndex<Value, int> *, MemorySpace> aux(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, prefix + "index_values"),
      0);
  queryDispatch2RoundImpl(NearestPredicateTag{}, tree, space, predicates,
                          Experimental::declare_callback_constrained(
                              DefaultCallbackWithRank{comm_rank}),
                          aux, offset);
  auto n_results = aux.extent(0);

  // unzip
  Kokkos::resize(Kokkos::view_alloc(space, Kokkos::WithoutInitializing), ranks,
                 n_results);
  Kokkos::resize(Kokkos::view_alloc(space, Kokkos::WithoutInitializing),
                 index_values, n_results);
  Kokkos::parallel_for(
      prefix + "unzip",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_results),
      KOKKOS_LAMBDA(int i) {
        index_values(i) = aux(i).value;
        ranks(i) = aux(i).index;
      });
  Kokkos::resize(Kokkos::view_alloc(space, Kokkos::WithoutInitializing), aux,
                 0); // free space

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion(prefix + "scatter");

  Distributor<MemorySpace> distributor(comm);
  auto const n_imports = distributor.createFromSends(space, ranks);
  Kokkos::resize(Kokkos::view_alloc(space), ranks, 0); // free space

  // Scatter (predicate, tree value) to the corresponding ranks
  using Tuple = TupleQueryIndexValue<Query, Value>;
  Kokkos::View<Tuple *, MemorySpace> exports(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         prefix + "exports"),
      n_results);
  Kokkos::parallel_for(
      prefix + "zip_predicates_and_values",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_predicates),
      KOKKOS_LAMBDA(int i) {
        for (auto j = offset(i); j < offset(i + 1); ++j)
          exports(j) = {.query = predicates(i),
                        .value = index_values(j),
                        .query_id = i,
                        .rank = comm_rank};
      });

  Kokkos::View<Tuple *, MemorySpace> imports(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         prefix + "imports"),
      n_imports);
  distributor.doPostsAndWaits(space, exports, imports);
  Kokkos::resize(Kokkos::view_alloc(space, Kokkos::WithoutInitializing),
                 exports, 0); // free space

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion(prefix + "execute_callback");

  // Execute the callback on the process owning the primitives.
  Values remote_values(Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                                          prefix + "remote_values"),
                       n_imports);
  Kokkos::View<int *, MemorySpace> remote_predicate_ids(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         prefix + "remote_predicate_ids"),
      n_imports);
  Kokkos::View<int *, MemorySpace> remote_ranks(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         prefix + "remote_ranks"),
      n_imports);
  if (n_imports > 0)
  {
    Kokkos::View<int *, MemorySpace> counts(
        Kokkos::view_alloc(space, prefix + "counts"), n_imports + 1);
    Kokkos::parallel_for(
        prefix + "execute_callbacks_1st_pass",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_imports),
        KOKKOS_LAMBDA(int i) {
          callback(imports(i).query, imports(i).value,
                   [&](typename Values::value_type const &value) {
                     auto count = Kokkos::atomic_fetch_inc(&counts(i));
                     if (count == 0)
                     {
                       remote_values(i) = value;
                       remote_predicate_ids(i) = imports(i).query_id;
                       remote_ranks(i) = imports(i).rank;
                     }
                   });
        });

    auto [min_count, max_count] = KokkosExt::minmax_reduce(
        space, Kokkos::subview(counts, std::make_pair(0, (int)n_imports)));
    if (min_count != 1 || max_count != 1)
    {
      KokkosExt::exclusive_scan(space, counts, counts, 0);

      auto const n_results = KokkosExt::lastElement(space, counts);
      KokkosExt::reallocWithoutInitializing(space, remote_values, n_results);
      KokkosExt::reallocWithoutInitializing(space, remote_predicate_ids,
                                            n_results);
      KokkosExt::reallocWithoutInitializing(space, remote_ranks, n_results);
      Kokkos::parallel_for(
          prefix + "execute_callback_2nd_pass",
          Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_imports),
          KOKKOS_LAMBDA(int i) {
            callback(imports(i).query, imports(i).value,
                     [&](typename Values::value_type const &value) {
                       auto j = Kokkos::atomic_fetch_inc(&counts(i));
                       remote_values(j) = value;
                       remote_predicate_ids(j) = imports(i).query_id;
                       remote_ranks(j) = imports(i).rank;
                     });
          });
    }
  }

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion(prefix + "gather");

  // Send the result back to the process owning the predicates.
  Distributor<MemorySpace> back_distributor(comm);
  auto const n_imports_back =
      back_distributor.createFromSends(space, remote_ranks);
  KokkosExt::reallocWithoutInitializing(space, values, n_imports_back);
  Kokkos::View<int *, MemorySpace> predicate_ids(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         prefix + "query_ids"),
      n_imports_back);

  back_distributor.doPostsAndWaits(space, remote_values, values);
  back_distributor.doPostsAndWaits(space, remote_predicate_ids, predicate_ids);

  Kokkos::resize(Kokkos::view_alloc(space, Kokkos::WithoutInitializing), offset,
                 n_predicates + 1);
  DistributedTree::countResults(space, n_predicates, predicate_ids, offset);
  KokkosExt::sortByKey(space, predicate_ids, values);

  Kokkos::Profiling::popRegion();
}

template <typename Tree, typename ExecutionSpace,
          Details::Concepts::Predicates Predicates, typename Values,
          typename Offset>
  requires(Kokkos::is_view_v<Values> && Kokkos::is_view_v<Offset>)
void DistributedTreeImpl::queryDispatch(NearestPredicateTag tag,
                                        Tree const &tree,
                                        ExecutionSpace const &space,
                                        Predicates const &predicates,
                                        Values &values, Offset &offset)
{
  queryDispatch2RoundImpl(tag, tree, space, predicates, DefaultCallback{},
                          values, offset);
}

template <typename Tree, typename ExecutionSpace,
          Details::Concepts::Predicates Predicates, typename Callback,
          typename Values, typename Offset>
  requires(Kokkos::is_view_v<Values> && Kokkos::is_view_v<Offset>)
void DistributedTreeImpl::queryDispatch(NearestPredicateTag, Tree const &tree,
                                        ExecutionSpace const &space,
                                        Predicates const &predicates,
                                        Callback const &callback,
                                        Values &values, Offset &offset)
{
  if (tree.empty())
  {
    KokkosExt::reallocWithoutInitializing(space, values, 0);
    KokkosExt::reallocWithoutInitializing(space, offset, predicates.size() + 1);
    Kokkos::deep_copy(space, offset, 0);
    return;
  }

  if constexpr (is_constrained_callback_v<Callback>)
    queryDispatch2RoundImpl(NearestPredicateTag{}, tree, space, predicates,
                            callback, values, offset);
  else
  {
    static_assert(!is_tagged_post_callback<Callback>::value);
    queryDispatch3RoundImpl(NearestPredicateTag{}, tree, space, predicates,
                            callback, values, offset);
  }
}

} // namespace ArborX::Details

#endif
