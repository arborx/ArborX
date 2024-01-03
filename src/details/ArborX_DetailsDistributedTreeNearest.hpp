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
#include <ArborX_DetailsKokkosExtMinMaxOperations.hpp>
#include <ArborX_DetailsKokkosExtStdAlgorithms.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_DetailsLegacy.hpp>
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

struct PairIndexDistance
{
  int index;
  float distance;
};

template <typename Tree>
struct CallbackWithDistance
{
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
    out({index, distance(getGeometry(query), leaf_node_bounding_volume)});
  }
};

template <typename ExecutionSpace, typename Tree, typename Predicates,
          typename Distances, typename Indices, typename Offset>
void DistributedTreeImpl::deviseStrategy(ExecutionSpace const &space,
                                         Tree const &tree,
                                         Predicates const &predicates,
                                         Distances const &,
                                         Indices &nearest_ranks, Offset &offset)
{
  Kokkos::Profiling::ScopedRegion guard(
      "ArborX::DistributedTree::query::nearest::deviseStrategy");

  using namespace DistributedTree;
  using MemorySpace = typename Tree::memory_space;

  auto const &top_tree = tree._top_tree;
  auto const &bottom_tree_sizes = tree._bottom_tree_sizes;

  // Find the k nearest local trees.
  query(top_tree, space, predicates, LegacyDefaultCallback{}, nearest_ranks,
        offset);

  // Accumulate total leave count in the local trees until it reaches k which
  // is the number of neighbors queried for.  Stop if local trees get
  // empty because it means that they are no more leaves and there is no point
  // on forwarding predicates to leafless trees.
  auto const n_predicates = predicates.size();
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
      "ArborX::DistributedTree::query::nearest::truncate_before_forwarding",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_predicates),
      KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < new_offset(i + 1) - new_offset(i); ++j)
          new_nearest_ranks(new_offset(i) + j) = nearest_ranks(offset(i) + j);
      });

  offset = new_offset;
  nearest_ranks = new_nearest_ranks;
}

template <typename ExecutionSpace, typename Tree, typename Predicates,
          typename Distances, typename Indices, typename Offset>
void DistributedTreeImpl::reassessStrategy(
    ExecutionSpace const &space, Tree const &tree, Predicates const &queries,
    Distances const &distances, Indices &nearest_ranks, Offset &offset)
{
  Kokkos::Profiling::ScopedRegion guard(
      "ArborX::DistributedTree::query::nearest::reassessStrategy");

  using namespace DistributedTree;
  using MemorySpace = typename Tree::memory_space;

  auto const &top_tree = tree._top_tree;
  auto const n_queries = queries.size();

  // Determine distance to the farthest neighbor found so far.
  Kokkos::View<float *, MemorySpace> farthest_distances(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::DistributedTree::query::nearest::"
                         "reassessStrategy::distances"),
      n_queries);
  // NOTE: in principle distances( j ) are arranged in ascending order for
  // offset( i ) <= j < offset( i + 1 ) so max() is not necessary.
  Kokkos::parallel_for(
      "ArborX::DistributedTree::query::nearest::most_distant_neighbor_so_far",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
      KOKKOS_LAMBDA(int i) {
        using KokkosExt::max;
        farthest_distances(i) = 0.;
        for (int j = offset(i); j < offset(i + 1); ++j)
          farthest_distances(i) = max(farthest_distances(i), distances(j));
      });

  check_valid_access_traits(
      PredicatesTag{},
      WithinDistanceFromPredicates<Predicates, decltype(farthest_distances)>{
          queries, farthest_distances});

  query(top_tree, space,
        WithinDistanceFromPredicates<Predicates, decltype(farthest_distances)>{
            queries, farthest_distances},
        LegacyDefaultCallback{}, nearest_ranks, offset);
  // NOTE: in principle, we could perform radius searches on the bottom_tree
  // rather than nearest queries.
}

template <typename Tree, typename ExecutionSpace, typename Predicates,
          typename Indices, typename Offset, typename Ranks, typename Distances>
std::enable_if_t<Kokkos::is_view<Indices>{} && Kokkos::is_view<Offset>{} &&
                 Kokkos::is_view<Ranks>{} && Kokkos::is_view<Distances>{}>
DistributedTreeImpl::queryDispatchImpl(NearestPredicateTag, Tree const &tree,
                                       ExecutionSpace const &space,
                                       Predicates const &queries,
                                       Indices &indices, Offset &offset,
                                       Ranks &ranks)
{
  Kokkos::Profiling::ScopedRegion guard(
      "ArborX::DistributedTree::query::nearest");

  using namespace DistributedTree;
  using MemorySpace = typename Tree::memory_space;

  auto const &bottom_tree = tree._bottom_tree;
  auto comm = tree.getComm();

  Distances distances("ArborX::DistributedTree::query::nearest::distances", 0);

  // "Strategy" is used to determine what ranks to forward queries to.  In
  // the 1st pass, the queries are sent to as many ranks as necessary to
  // guarantee that all k neighbors queried for are found.  In the 2nd pass,
  // queries are sent again to all ranks that may have a neighbor closer to
  // the farthest neighbor identified in the 1st pass.
  //
  // The current implementation discards the results after the 1st pass and
  // recompute everything instead of just searching for potential better
  // neighbors and updating the list.

  // Right now, distance calculations only work with BVH due to using functions
  // in DistributedTreeNearestUtils. So, there's no point in replacing this
  // with decltype.
  CallbackWithDistance<ArborX::BVH<MemorySpace>> callback_with_distance(
      space, bottom_tree);

  Kokkos::View<int *, MemorySpace> nearest_ranks(
      "ArborX::DistributedTree::query::nearest::nearest_ranks", 0);

  // NOTE: compiler would not deduce __range for the braced-init-list, but I
  // got it to work with the static_cast to function pointers.
  using Strategy =
      void (*)(ExecutionSpace const &, Tree const &, Predicates const &,
               Distances const &, decltype(nearest_ranks) &, Offset &);
  for (auto implementStrategy : {static_cast<Strategy>(deviseStrategy),
                                 static_cast<Strategy>(reassessStrategy)})
  {
    implementStrategy(space, tree, queries, distances, nearest_ranks, offset);

    {
      // NOTE_COMM_NEAREST: The communication pattern here for the nearest
      // search is identical to that of the spatial search (see
      // NOTE_COMM_SPATIAL). The code differences are:
      // - no callbacks
      // - explicit distances
      // - results filtering

      // Forward queries
      using Query = typename Predicates::value_type;
      Kokkos::View<int *, MemorySpace> ids(
          "ArborX::DistributedTree::query::nearest::query_ids", 0);
      Kokkos::View<Query *, MemorySpace> fwd_queries(
          "ArborX::DistributedTree::query::nearest::fwd_queries", 0);
      forwardQueries(comm, space, queries, nearest_ranks, offset, fwd_queries,
                     ids, ranks);

      // Perform queries that have been received
      Kokkos::View<PairIndexDistance *, MemorySpace> out(
          "ArborX::DistributedTree::query::pairs_index_distance", 0);
      query(bottom_tree, space, fwd_queries, callback_with_distance, out,
            offset);

      // Unzip
      auto const n = out.extent(0);
      KokkosExt::reallocWithoutInitializing(space, indices, n);
      KokkosExt::reallocWithoutInitializing(space, distances, n);
      Kokkos::parallel_for(
          "ArborX::DistributedTree::query::nearest::split_"
          "index_distance_pairs",
          Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
          KOKKOS_LAMBDA(int i) {
            indices(i) = out(i).index;
            distances(i) = out(i).distance;
          });

      // Communicate results back
      communicateResultsBack(comm, space, indices, offset, ranks, ids,
                             &distances);

      // Merge results
      Kokkos::Profiling::pushRegion(
          "ArborX::DistributedTree::query::nearest::postprocess_results");

      int const n_queries = queries.size();
      countResults(space, n_queries, ids, offset);
      sortResults(space, ids, indices, ranks, distances);
      filterResults(space, queries, distances, indices, offset, ranks);

      Kokkos::Profiling::popRegion();
    }
  }
}

template <typename Tree, typename ExecutionSpace, typename Predicates,
          typename IndicesAndRanks, typename Offset>
std::enable_if_t<Kokkos::is_view<IndicesAndRanks>{} &&
                 Kokkos::is_view<Offset>{}>
DistributedTreeImpl::queryDispatch(NearestPredicateTag tag, Tree const &tree,
                                   ExecutionSpace const &space,
                                   Predicates const &queries,
                                   IndicesAndRanks &values, Offset &offset)
{
  // FIXME avoid zipping when distributed nearest callbacks become available
  Kokkos::View<int *, ExecutionSpace> indices(
      "ArborX::DistributedTree::query::nearest::indices", 0);
  Kokkos::View<int *, ExecutionSpace> ranks(
      "ArborX::DistributedTree::query::nearest::ranks", 0);
  queryDispatchImpl(tag, tree, space, queries, indices, offset, ranks);
  auto const n = indices.extent(0);
  KokkosExt::reallocWithoutInitializing(space, values, n);
  Kokkos::parallel_for(
      "ArborX::DistributedTree::query::zip_indices_and_ranks",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        values(i) = {indices(i), ranks(i)};
      });
}

} // namespace Details
} // namespace ArborX

#endif
