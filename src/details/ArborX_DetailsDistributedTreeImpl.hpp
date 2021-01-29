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
#ifndef ARBORX_DETAILS_DISTRIBUTED_SEARCH_TREE_IMPL_HPP
#define ARBORX_DETAILS_DISTRIBUTED_SEARCH_TREE_IMPL_HPP

#include <ArborX_Config.hpp>

#include <ArborX_DetailsDistributor.hpp>
#include <ArborX_DetailsKokkosExtMinMaxOperations.hpp>
#include <ArborX_DetailsPriorityQueue.hpp>
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Predicates.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

namespace ArborX
{

namespace Details
{
// This is a struct only make it a friend to BVH
template <typename BVH>
struct DistributedTreeNearestUtils
{
  template <typename ReversePermutation>
  static KOKKOS_FUNCTION typename BVH::bounding_volume_type const &
  getPrimitiveBoundingVolume(BVH const &bvh,
                             ReversePermutation const &rev_permute,
                             int leaf_index)
  {
    int const leaf_nodes_shift = bvh.size() - 1;
    return bvh.getBoundingVolume(
        bvh.getNodePtr(rev_permute(leaf_index) + leaf_nodes_shift));
  }

  template <typename ExecutionSpace>
  static Kokkos::View<unsigned int *, typename BVH::memory_space>
  getReversePermutation(ExecutionSpace const &exec_space, BVH const &bvh)
  {
    auto const n = bvh.size();

    Kokkos::View<unsigned int *, typename BVH::memory_space> rev_permute(
        Kokkos::ViewAllocateWithoutInitializing(
            "ArborX::DistributedTree::query::nearest::reverse_permutation"),
        n);
    if (!bvh.empty())
    {
      auto const &leaf_nodes = bvh.getLeafNodes();
      Kokkos::parallel_for(
          "ArborX::DistributedTree::query::nearest::compute_reverse_"
          "permutation",
          Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
          KOKKOS_LAMBDA(int const i) {
            rev_permute(leaf_nodes(i).getLeafPermutationIndex()) = i;
          });
    }

    return rev_permute;
  }
};

struct DefaultCallbackWithRank
{
  int _rank;
  template <typename Predicate, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const &, int primitive_index,
                                  OutputFunctor const &out) const
  {
    out({primitive_index, _rank});
  }
};

template <typename DeviceType>
struct DistributedTreeImpl
{
  // spatial queries
  template <typename DistributedTree, typename ExecutionSpace,
            typename Predicates, typename IndicesAndRanks, typename Offset>
  static std::enable_if_t<Kokkos::is_view<IndicesAndRanks>{} &&
                          Kokkos::is_view<Offset>{}>
  queryDispatch(SpatialPredicateTag, DistributedTree const &tree,
                ExecutionSpace const &space, Predicates const &queries,
                IndicesAndRanks &values, Offset &offset)
  {
    int comm_rank;
    MPI_Comm_rank(tree.getComm(), &comm_rank);
    queryDispatch(SpatialPredicateTag{}, tree, space, queries,
                  DefaultCallbackWithRank{comm_rank}, values, offset);
  }

  // NOTE NVCC did not like having definition of that type within the
  // queryDispatch function below while using an extended __host__ __device__
  // lambda
  struct PairIndexRank
  {
    int index;
    int rank;
  };

  template <typename DistributedTree, typename ExecutionSpace,
            typename Predicates, typename Indices, typename Offset,
            typename Ranks>
  [[deprecated]] static std::enable_if_t<Kokkos::is_view<Indices>{} &&
                                         Kokkos::is_view<Offset>{} &&
                                         Kokkos::is_view<Ranks>{}>
  queryDispatch(SpatialPredicateTag, DistributedTree const &tree,
                ExecutionSpace const &space, Predicates const &queries,
                Indices &indices, Offset &offset, Ranks &ranks)
  {
    Kokkos::View<PairIndexRank *, ExecutionSpace> out(
        "ArborX::DistributedTree::query::spatial::pairs_index_rank", 0);
    queryDispatch(SpatialPredicateTag{}, tree, space, queries, out, offset);
    auto const n = out.extent(0);
    reallocWithoutInitializing(indices, n);
    reallocWithoutInitializing(ranks, n);
    Kokkos::parallel_for("ArborX::DistributedTree::query::split_pairs",
                         Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
                         KOKKOS_LAMBDA(int i) {
                           indices(i) = out(i).index;
                           ranks(i) = out(i).rank;
                         });
  }

  template <typename DistributedTree, typename ExecutionSpace,
            typename Predicates, typename OutputView, typename OffsetView,
            typename Callback>
  static std::enable_if_t<Kokkos::is_view<OutputView>{} &&
                          Kokkos::is_view<OffsetView>{}>
  queryDispatch(SpatialPredicateTag, DistributedTree const &tree,
                ExecutionSpace const &space, Predicates const &queries,
                Callback const &callback, OutputView &out, OffsetView &offset);

  // nearest neighbors queries
  template <typename DistributedTree, typename ExecutionSpace,
            typename Predicates, typename Indices, typename Offset,
            typename Ranks,
            typename Distances = Kokkos::View<float *, DeviceType>>
  static std::enable_if_t<
      Kokkos::is_view<Indices>{} && Kokkos::is_view<Offset>{} &&
      Kokkos::is_view<Ranks>{} && Kokkos::is_view<Distances>{}>
  queryDispatchImpl(NearestPredicateTag, DistributedTree const &tree,
                    ExecutionSpace const &space, Predicates const &queries,
                    Indices &indices, Offset &offset, Ranks &ranks,
                    Distances *distances_ptr = nullptr);

  template <typename DistributedTree, typename ExecutionSpace,
            typename Predicates, typename Indices, typename Offset,
            typename Ranks>
  [[deprecated]] static std::enable_if_t<Kokkos::is_view<Indices>{} &&
                                         Kokkos::is_view<Offset>{} &&
                                         Kokkos::is_view<Ranks>{}>
  queryDispatch(NearestPredicateTag tag, DistributedTree const &tree,
                ExecutionSpace const &space, Predicates const &queries,
                Indices &indices, Offset &offset, Ranks &ranks)
  {
    queryDispatchImpl(tag, tree, space, queries, indices, offset, ranks);
  }

  template <typename DistributedTree, typename ExecutionSpace,
            typename Predicates, typename Indices, typename Offset,
            typename Ranks, typename Distances>
  [[deprecated]] static std::enable_if_t<
      Kokkos::is_view<Indices>{} && Kokkos::is_view<Offset>{} &&
      Kokkos::is_view<Ranks>{} && Kokkos::is_view<Distances>{}>
  queryDispatch(NearestPredicateTag tag, DistributedTree const &tree,
                ExecutionSpace const &space, Predicates const &queries,
                Indices &indices, Offset &offset, Ranks &ranks,
                Distances &distances)
  {
    queryDispatchImpl(tag, tree, space, queries, indices, offset, ranks,
                      &distances);
  }

  template <typename DistributedTree, typename ExecutionSpace,
            typename Predicates, typename IndicesAndRanks, typename Offset>
  static std::enable_if_t<Kokkos::is_view<IndicesAndRanks>{} &&
                          Kokkos::is_view<Offset>{}>
  queryDispatch(NearestPredicateTag tag, DistributedTree const &tree,
                ExecutionSpace const &space, Predicates const &queries,
                IndicesAndRanks &values, Offset &offset)
  {
    // FIXME avoid zipping when distributed nearest callbacks become availale
    Kokkos::View<int *, ExecutionSpace> indices(
        "ArborX::DistributedTree::query::nearest::indices", 0);
    Kokkos::View<int *, ExecutionSpace> ranks(
        "ArborX::DistributedTree::query::nearest::ranks", 0);
    queryDispatchImpl(tag, tree, space, queries, indices, offset, ranks);
    auto const n = indices.extent(0);
    reallocWithoutInitializing(values, n);
    Kokkos::parallel_for(
        "ArborX::DistributedTree::query::zip_indices_and_ranks",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
          values(i) = {indices(i), ranks(i)};
        });
  }

  template <typename DistributedTree, typename ExecutionSpace,
            typename Predicates, typename Indices, typename Offset,
            typename Distances>
  static void deviseStrategy(ExecutionSpace const &space,
                             Predicates const &queries,
                             DistributedTree const &tree, Indices &indices,
                             Offset &offset, Distances &);

  template <typename DistributedTree, typename ExecutionSpace,
            typename Predicates, typename Indices, typename Offset,
            typename Distances>
  static void reassessStrategy(ExecutionSpace const &space,
                               Predicates const &queries,
                               DistributedTree const &tree, Indices &indices,
                               Offset &offset, Distances &distances);

  template <typename ExecutionSpace, typename Predicates, typename Ranks,
            typename Query>
  static void forwardQueries(MPI_Comm comm, ExecutionSpace const &space,
                             Predicates const &queries,
                             Kokkos::View<int *, DeviceType> indices,
                             Kokkos::View<int *, DeviceType> offset,
                             Kokkos::View<Query *, DeviceType> &fwd_queries,
                             Kokkos::View<int *, DeviceType> &fwd_ids,
                             Ranks &fwd_ranks);

  template <typename ExecutionSpace, typename OutputView, typename Ranks,
            typename Distances = Kokkos::View<float *, DeviceType>>
  static void communicateResultsBack(MPI_Comm comm, ExecutionSpace const &space,
                                     OutputView &out,
                                     Kokkos::View<int *, DeviceType> offset,
                                     Ranks &ranks,
                                     Kokkos::View<int *, DeviceType> &ids,
                                     Distances *distances_ptr = nullptr);

  template <typename ExecutionSpace, typename Predicates, typename Indices,
            typename Offset, typename Ranks>
  static void filterResults(ExecutionSpace const &space,
                            Predicates const &queries,
                            Kokkos::View<float *, DeviceType> distances,
                            Indices &indices, Offset &offset, Ranks &ranks);

  template <typename ExecutionSpace, typename View, typename... OtherViews>
  static void sortResults(ExecutionSpace const &space, View keys,
                          OtherViews... other_views);

  template <typename ExecutionSpace, typename OffsetView>
  static void countResults(ExecutionSpace const &space, int n_queries,
                           Kokkos::View<int *, DeviceType> query_ids,
                           OffsetView &offset);

  template <typename ExecutionSpace, typename View>
  static typename std::enable_if<Kokkos::is_view<View>::value>::type
  sendAcrossNetwork(ExecutionSpace const &space,
                    Distributor<DeviceType> const &distributor, View exports,
                    typename View::non_const_type imports);
};

template <typename DeviceType>
template <typename ExecutionSpace, typename View>
typename std::enable_if<Kokkos::is_view<View>::value>::type
DistributedTreeImpl<DeviceType>::sendAcrossNetwork(
    ExecutionSpace const &space, Distributor<DeviceType> const &distributor,
    View exports, typename View::non_const_type imports)
{
  Kokkos::Profiling::pushRegion("ArborX::DistributedTree::sendAcrossNetwork");

  ARBORX_ASSERT((exports.extent(0) == distributor.getTotalSendLength()) &&
                (imports.extent(0) == distributor.getTotalReceiveLength()) &&
                (exports.extent(1) == imports.extent(1)) &&
                (exports.extent(2) == imports.extent(2)) &&
                (exports.extent(3) == imports.extent(3)) &&
                (exports.extent(4) == imports.extent(4)) &&
                (exports.extent(5) == imports.extent(5)) &&
                (exports.extent(6) == imports.extent(6)) &&
                (exports.extent(7) == imports.extent(7)));

  auto const num_packets = exports.extent(1) * exports.extent(2) *
                           exports.extent(3) * exports.extent(4) *
                           exports.extent(5) * exports.extent(6) *
                           exports.extent(7);

  using NonConstValueType = typename View::non_const_value_type;

#ifndef ARBORX_USE_CUDA_AWARE_MPI
  using MirrorSpace = typename View::host_mirror_space;
  MirrorSpace const execution_space;
#else
  using MirrorSpace = typename View::device_type;
  auto const &execution_space = space;
#endif

  auto imports_layout_right =
      create_layout_right_mirror_view(execution_space, imports);

  Kokkos::View<NonConstValueType *, MirrorSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      import_buffer(imports_layout_right.data(), imports_layout_right.size());

  distributor.doPostsAndWaits(space, exports, num_packets, import_buffer);

  auto tmp_view =
      Kokkos::create_mirror_view_and_copy(space, imports_layout_right);
  Kokkos::deep_copy(space, imports, tmp_view);

  Kokkos::Profiling::popRegion();
}

template <typename DeviceType>
template <typename DistributedTree, typename ExecutionSpace,
          typename Predicates, typename Indices, typename Offset,
          typename Distances>
void DistributedTreeImpl<DeviceType>::deviseStrategy(
    ExecutionSpace const &space, Predicates const &queries,
    DistributedTree const &tree, Indices &indices, Offset &offset, Distances &)
{
  Kokkos::Profiling::pushRegion("ArborX::DistributedTree::deviseStrategy");

  auto const &top_tree = tree._top_tree;
  auto const &bottom_tree_sizes = tree._bottom_tree_sizes;

  // Find the k nearest local trees.
  query(top_tree, space, queries, indices, offset);

  // Accumulate total leave count in the local trees until it reaches k which
  // is the number of neighbors queried for.  Stop if local trees get
  // empty because it means that they are no more leaves and there is no point
  // on forwarding queries to leafless trees.
  using Access = AccessTraits<Predicates, PredicatesTag>;
  auto const n_queries = Access::size(queries);
  Kokkos::View<int *, DeviceType> new_offset(
      Kokkos::view_alloc(offset.label(), space), n_queries + 1);
  Kokkos::parallel_for(
      "ArborX::DistributedTree::query::"
      "bottom_trees_with_required_cumulated_leaves_count",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
      KOKKOS_LAMBDA(int i) {
        int leaves_count = 0;
        int const n_nearest_neighbors = getK(Access::get(queries, i));
        for (int j = offset(i); j < offset(i + 1); ++j)
        {
          int const bottom_tree_size = bottom_tree_sizes(indices(j));
          if ((bottom_tree_size == 0) || (leaves_count >= n_nearest_neighbors))
            break;
          leaves_count += bottom_tree_size;
          ++new_offset(i);
        }
      });

  exclusivePrefixSum(space, new_offset);

  // Truncate results so that queries will only be forwarded to as many local
  // trees as necessary to find k neighbors.
  Kokkos::View<int *, DeviceType> new_indices(
      Kokkos::view_alloc(indices.label(), space), lastElement(new_offset));
  Kokkos::parallel_for(
      "ArborX::DistributedTree::query::truncate_before_forwarding",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
      KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < new_offset(i + 1) - new_offset(i); ++j)
          new_indices(new_offset(i) + j) = indices(offset(i) + j);
      });

  offset = new_offset;
  indices = new_indices;

  Kokkos::Profiling::popRegion();
}

template <typename DeviceType>
template <typename DistributedTree, typename ExecutionSpace,
          typename Predicates, typename Indices, typename Offset,
          typename Distances>
void DistributedTreeImpl<DeviceType>::reassessStrategy(
    ExecutionSpace const &space, Predicates const &queries,
    DistributedTree const &tree, Indices &indices, Offset &offset,
    Distances &distances)
{
  Kokkos::Profiling::pushRegion("ArborX::DistributedTree::reassessStrategy");

  auto const &top_tree = tree._top_tree;
  using Access = AccessTraits<Predicates, PredicatesTag>;
  auto const n_queries = Access::size(queries);

  // Determine distance to the farthest neighbor found so far.
  Kokkos::View<float *, DeviceType> farthest_distances(
      Kokkos::ViewAllocateWithoutInitializing(
          "ArborX::DistributedTree::query::reassessStrategy::distances"),
      n_queries);
  // NOTE: in principle distances( j ) are arranged in ascending order for
  // offset( i ) <= j < offset( i + 1 ) so max() is not necessary.
  Kokkos::parallel_for(
      "ArborX::DistributedTree::query::most_distant_neighbor_so_far",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
      KOKKOS_LAMBDA(int i) {
        using KokkosExt::max;
        farthest_distances(i) = 0.;
        for (int j = offset(i); j < offset(i + 1); ++j)
          farthest_distances(i) = max(farthest_distances(i), distances(j));
      });

  // Identify what ranks may have leaves that are within that distance.
  Kokkos::View<decltype(intersects(Sphere{})) *, DeviceType> radius_searches(
      Kokkos::ViewAllocateWithoutInitializing(
          "ArborX::DistributedTree::query::reassessStrategy::queries"),
      n_queries);
  Kokkos::parallel_for(
      "ArborX::DistributedTree::query::bottom_trees_within_that_distance",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
      KOKKOS_LAMBDA(int i) {
        radius_searches(i) = intersects(Sphere{
            getGeometry(Access::get(queries, i)), farthest_distances(i)});
      });

  query(top_tree, space, radius_searches, indices, offset);
  // NOTE: in principle, we could perform radius searches on the bottom_tree
  // rather than nearest queries.

  Kokkos::Profiling::popRegion();
}

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
    _rev_permute = DistributedTreeNearestUtils<Tree>::getReversePermutation(
        exec_space, _tree);
  }

  template <typename Query, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Query const &query, int index,
                                  OutputFunctor const &output) const
  {
    // TODO: This breaks the abstraction of the distributed Tree not knowing
    // the details of the local tree. Right now, this is the only way. Will
    // need to be fixed with a proper callback abstraction.
    output(
        {index,
         distance(getGeometry(query),
                  DistributedTreeNearestUtils<Tree>::getPrimitiveBoundingVolume(
                      _tree, _rev_permute, index))});
  }
};

template <typename DeviceType>
template <typename DistributedTree, typename ExecutionSpace,
          typename Predicates, typename Indices, typename Offset,
          typename Ranks, typename Distances>
std::enable_if_t<Kokkos::is_view<Indices>{} && Kokkos::is_view<Offset>{} &&
                 Kokkos::is_view<Ranks>{} && Kokkos::is_view<Distances>{}>
DistributedTreeImpl<DeviceType>::queryDispatchImpl(
    NearestPredicateTag, DistributedTree const &tree,
    ExecutionSpace const &space, Predicates const &queries, Indices &indices,
    Offset &offset, Ranks &ranks, Distances *distances_ptr)
{
  Kokkos::Profiling::pushRegion("ArborX::DistributedTree::query::nearest");

  auto const &bottom_tree = tree._bottom_tree;
  auto comm = tree.getComm();

  Distances distances("ArborX::DistributedTree::query::nearest::distances", 0);
  if (distances_ptr)
    distances = *distances_ptr;

  // "Strategy" is used to determine what ranks to forward queries to.  In
  // the 1st pass, the queries are sent to as many ranks as necessary to
  // guarantee that all k neighbors queried for are found.  In the 2nd pass,
  // queries are sent again to all ranks that may have a neighbor closer to
  // the farthest neighbor identified in the 1st pass.
  //
  // The current implementation discards the results after the 1st pass and
  // recompute everything instead of just searching for potential better
  // neighbors and updating the list.

  // Right now, distance calcuations only work with BVH due to using functions
  // in DistributedTreeNearestUtils. So, there's no point in replacing this
  // with decltype.
  CallbackWithDistance<BVH<typename DeviceType::memory_space>>
      callback_with_distance(space, bottom_tree);

  // NOTE: compiler would not deduce __range for the braced-init-list but I
  // got it to work with the static_cast to function pointers.
  using Strategy =
      void (*)(ExecutionSpace const &, Predicates const &,
               DistributedTree const &, Indices &, Offset &, Distances &);
  for (auto implementStrategy :
       {static_cast<Strategy>(DistributedTreeImpl<DeviceType>::deviseStrategy),
        static_cast<Strategy>(
            DistributedTreeImpl<DeviceType>::reassessStrategy)})
  {
    implementStrategy(space, queries, tree, indices, offset, distances);

    {
      // NOTE_COMM_NEAREST: The communication pattern here for the nearest
      // search is identical to that of the spatial search (see
      // NOTE_COMM_SPATIAL). The code differences are:
      // - no callbacks
      // - explicit distances
      // - results filtering

      // Forward queries
      using Access = AccessTraits<Predicates, PredicatesTag>;
      using Query = typename AccessTraitsHelper<Access>::type;
      Kokkos::View<int *, DeviceType> ids(
          "ArborX::DistributedTree::query::nearest::query_ids", 0);
      Kokkos::View<Query *, DeviceType> fwd_queries(
          "ArborX::DistributedTree::query::nearest::fwd_queries", 0);
      forwardQueries(comm, space, queries, indices, offset, fwd_queries, ids,
                     ranks);

      // Perform queries that have been received
      Kokkos::View<PairIndexDistance *, DeviceType> out(
          "ArborX::DistributedTree::query::pairs_index_distance", 0);
      query(bottom_tree, space, fwd_queries, callback_with_distance, out,
            offset);

      // Unzip
      auto const n = out.extent(0);
      reallocWithoutInitializing(indices, n);
      reallocWithoutInitializing(distances, n);
      Kokkos::parallel_for("ArborX::DistributedTree::query::nearest::split_"
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
          "ArborX::DistributedTree::nearest::postprocess_results");

      int const n_queries = Access::size(queries);
      countResults(space, n_queries, ids, offset);
      sortResults(space, ids, indices, ranks, distances);
      filterResults(space, queries, distances, indices, offset, ranks);

      Kokkos::Profiling::popRegion();
    }
  }

  Kokkos::Profiling::popRegion();
}

template <typename DeviceType>
template <typename DistributedTree, typename ExecutionSpace,
          typename Predicates, typename OutputView, typename OffsetView,
          typename Callback>
std::enable_if_t<Kokkos::is_view<OutputView>{} && Kokkos::is_view<OffsetView>{}>
DistributedTreeImpl<DeviceType>::queryDispatch(
    SpatialPredicateTag, DistributedTree const &tree,
    ExecutionSpace const &space, Predicates const &queries,
    Callback const &callback, OutputView &out, OffsetView &offset)
{
  Kokkos::Profiling::pushRegion("ArborX::DistributedTree::query::spatial");

  auto const &top_tree = tree._top_tree;
  auto const &bottom_tree = tree._bottom_tree;
  auto comm = tree.getComm();

  Kokkos::View<int *, DeviceType> indices(
      "ArborX::DistributedTree::query::spatial::indices", 0);
  Kokkos::View<int *, DeviceType> ranks(
      "ArborX::DistributedTree::query::spatial::ranks", 0);
  query(top_tree, space, queries, indices, offset);

  {
    // NOTE_COMM_SPATIAL: The communication pattern here for the spatial search
    // is identical to that of the nearest search (see NOTE_COMM_NEAREST). The
    // code differences are:
    // - usage of callbacks
    // - no explicit distances
    // - no results filtering

    // Forward queries
    using Access = AccessTraits<Predicates, PredicatesTag>;
    using Query = typename AccessTraitsHelper<Access>::type;
    Kokkos::View<int *, DeviceType> ids(
        "ArborX::DistributedTree::query::spatial::query_ids", 0);
    Kokkos::View<Query *, DeviceType> fwd_queries(
        "ArborX::DistributedTree::query::spatial::fwd_queries", 0);
    forwardQueries(comm, space, queries, indices, offset, fwd_queries, ids,
                   ranks);

    // Perform queries that have been received
    query(bottom_tree, space, fwd_queries, callback, out, offset);

    // Communicate results back
    communicateResultsBack(comm, space, out, offset, ranks, ids);

    Kokkos::Profiling::pushRegion(
        "ArborX::DistributedTree::spatial::postprocess_results");

    // Merge results
    int const n_queries = Access::size(queries);
    countResults(space, n_queries, ids, offset);
    sortResults(space, ids, out);

    Kokkos::Profiling::popRegion();
  }

  Kokkos::Profiling::popRegion();
}

template <typename DeviceType>
template <typename ExecutionSpace, typename View, typename... OtherViews>
void DistributedTreeImpl<DeviceType>::sortResults(ExecutionSpace const &space,
                                                  View keys,
                                                  OtherViews... other_views)
{
  auto const n = keys.extent(0);
  // If they were no queries, min_val and max_val values won't change after
  // the parallel reduce (they are initialized to +infty and -infty
  // respectively) and the sort will hang.
  if (n == 0)
    return;

  // We only want to get the permutation here, but sortObjects also sorts the
  // elements given to it. Hence, we need to create a copy.
  // TODO try to avoid the copy
  View keys_clone(Kokkos::ViewAllocateWithoutInitializing(
                      "ArborX::DistributedTree::query::sortResults::keys"),
                  keys.size());
  Kokkos::deep_copy(space, keys_clone, keys);
  auto const permutation = ArborX::Details::sortObjects(space, keys_clone);

  // Call applyPermutation for every entry in the parameter pack.
  // We need to use the comma operator here since the function returns void.
  // The variable we assign to is actually not needed. We just need something
  // to store the initializer list (that contains only zeros).
  auto dummy = {
      (ArborX::Details::applyPermutation(space, permutation, other_views),
       0)...};
  std::ignore = dummy;
}

template <typename DeviceType>
template <typename ExecutionSpace, typename OffsetView>
void DistributedTreeImpl<DeviceType>::countResults(
    ExecutionSpace const &space, int n_queries,
    Kokkos::View<int *, DeviceType> query_ids, OffsetView &offset)
{
  int const nnz = query_ids.extent(0);

  Kokkos::realloc(offset, n_queries + 1);

  Kokkos::parallel_for(
      "ArborX::DistributedTree::query::count_results_per_query",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, nnz), KOKKOS_LAMBDA(int i) {
        Kokkos::atomic_increment(&offset(query_ids(i)));
      });

  exclusivePrefixSum(space, offset);
}

template <typename DeviceType>
template <typename ExecutionSpace, typename Predicates, typename Ranks,
          typename Query>
void DistributedTreeImpl<DeviceType>::forwardQueries(
    MPI_Comm comm, ExecutionSpace const &space, Predicates const &queries,
    Kokkos::View<int *, DeviceType> indices,
    Kokkos::View<int *, DeviceType> offset,
    Kokkos::View<Query *, DeviceType> &fwd_queries,
    Kokkos::View<int *, DeviceType> &fwd_ids, Ranks &fwd_ranks)
{
  Kokkos::Profiling::pushRegion("ArborX::DistributedTree::forwardQueries");

  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  Distributor<DeviceType> distributor(comm);

  using Access = AccessTraits<Predicates, PredicatesTag>;
  int const n_queries = Access::size(queries);
  int const n_exports = lastElement(offset);
  int const n_imports = distributor.createFromSends(space, indices);

  static_assert(
      std::is_same<Query, typename AccessTraitsHelper<Access>::type>{}, "");
  Kokkos::View<Query *, DeviceType> exports(
      Kokkos::ViewAllocateWithoutInitializing(
          "ArborX::DistributedTree::query::forwardQueries::queries"),
      n_exports);
  Kokkos::parallel_for(
      "ArborX::DistributedTree::query::forward_queries_fill_buffer",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
      KOKKOS_LAMBDA(int q) {
        for (int i = offset(q); i < offset(q + 1); ++i)
        {
          exports(i) = Access::get(queries, q);
        }
      });

  Kokkos::View<int *, DeviceType> export_ranks(
      Kokkos::ViewAllocateWithoutInitializing(
          "ArborX::DistributedTree::query::forwardQueries::export_ranks"),
      n_exports);
  Kokkos::deep_copy(space, export_ranks, comm_rank);

  Kokkos::View<int *, DeviceType> import_ranks(
      Kokkos::ViewAllocateWithoutInitializing(
          "ArborX::DistributedTree::query::forwardQueries::import_ranks"),
      n_imports);
  sendAcrossNetwork(space, distributor, export_ranks, import_ranks);

  Kokkos::View<int *, DeviceType> export_ids(
      Kokkos::ViewAllocateWithoutInitializing(
          "ArborX::DistributedTree::query::forwardQueries::export_ids"),
      n_exports);
  Kokkos::parallel_for(
      "ArborX::DistributedTree::query::forward_queries_fill_ids",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
      KOKKOS_LAMBDA(int q) {
        for (int i = offset(q); i < offset(q + 1); ++i)
        {
          export_ids(i) = q;
        }
      });
  Kokkos::View<int *, DeviceType> import_ids(
      Kokkos::ViewAllocateWithoutInitializing(
          "ArborX::DistributedTree::query::forwardQueries::import_ids"),
      n_imports);
  sendAcrossNetwork(space, distributor, export_ids, import_ids);

  // Send queries across the network
  Kokkos::View<Query *, DeviceType> imports(
      Kokkos::ViewAllocateWithoutInitializing(
          "ArborX::DistributedTree::query::forwardQueries::queries"),
      n_imports);
  sendAcrossNetwork(space, distributor, exports, imports);

  fwd_queries = imports;
  fwd_ids = import_ids;
  fwd_ranks = import_ranks;

  Kokkos::Profiling::popRegion();
}

template <typename DeviceType>
template <typename ExecutionSpace, typename OutputView, typename Ranks,
          typename Distances>
void DistributedTreeImpl<DeviceType>::communicateResultsBack(
    MPI_Comm comm, ExecutionSpace const &space, OutputView &out,
    Kokkos::View<int *, DeviceType> offset, Ranks &ranks,
    Kokkos::View<int *, DeviceType> &ids, Distances *distances_ptr)
{
  Kokkos::Profiling::pushRegion(
      "ArborX::DistributedTree::communicateResultsBack");

  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  int const n_fwd_queries = offset.extent_int(0) - 1;
  int const n_exports = lastElement(offset);

  // We are assuming here that if the same rank is related to multiple batches
  // these batches appear consecutively. Hence, no reordering is necessary.
  Distributor<DeviceType> distributor(comm);
  // FIXME Distributor::createFromSends takes two views of the same type by
  // a const reference.  There were two easy ways out, either take the views by
  // value or cast at the callsite.  I went with the latter.  Proper fix
  // involves more code cleanup in ArborX_DetailsDistributor.hpp than I am
  // willing to do just now.
  int const n_imports =
      distributor.createFromSends(space, ranks, static_cast<Ranks>(offset));

  Kokkos::View<int *, DeviceType> export_ranks(
      Kokkos::ViewAllocateWithoutInitializing(ranks.label()), n_exports);
  Kokkos::deep_copy(space, export_ranks, comm_rank);
  Kokkos::View<int *, DeviceType> export_ids(
      Kokkos::ViewAllocateWithoutInitializing(ids.label()), n_exports);
  Kokkos::parallel_for(
      "ArborX::DistributedTree::query::fill_buffer",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_fwd_queries),
      KOKKOS_LAMBDA(int q) {
        for (int i = offset(q); i < offset(q + 1); ++i)
        {
          export_ids(i) = ids(q);
        }
      });
  OutputView export_out = out;

  OutputView import_out(Kokkos::ViewAllocateWithoutInitializing(out.label()),
                        n_imports);
  Kokkos::View<int *, DeviceType> import_ranks(
      Kokkos::ViewAllocateWithoutInitializing(ranks.label()), n_imports);
  Kokkos::View<int *, DeviceType> import_ids(
      Kokkos::ViewAllocateWithoutInitializing(ids.label()), n_imports);

  sendAcrossNetwork(space, distributor, export_out, import_out);
  sendAcrossNetwork(space, distributor, export_ranks, import_ranks);
  sendAcrossNetwork(space, distributor, export_ids, import_ids);

  ids = import_ids;
  ranks = import_ranks;
  out = import_out;

  if (distances_ptr)
  {
    auto &distances = *distances_ptr;
    Kokkos::View<float *, DeviceType> export_distances = distances;
    Kokkos::View<float *, DeviceType> import_distances(
        Kokkos::ViewAllocateWithoutInitializing(distances.label()), n_imports);
    sendAcrossNetwork(space, distributor, export_distances, import_distances);
    distances = import_distances;
  }

  Kokkos::Profiling::popRegion();
}

template <typename DeviceType>
template <typename ExecutionSpace, typename Predicates, typename Indices,
          typename Offset, typename Ranks>
void DistributedTreeImpl<DeviceType>::filterResults(
    ExecutionSpace const &space, Predicates const &queries,
    Kokkos::View<float *, DeviceType> distances, Indices &indices,
    Offset &offset, Ranks &ranks)
{
  Kokkos::Profiling::pushRegion("ArborX::DistributedTree::filterResults");

  using Access = AccessTraits<Predicates, PredicatesTag>;
  int const n_queries = Access::size(queries);
  // truncated views are prefixed with an underscore
  Kokkos::View<int *, DeviceType> new_offset(offset.label(), n_queries + 1);

  Kokkos::parallel_for("ArborX::DistributedTree::query::discard_results",
                       Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
                       KOKKOS_LAMBDA(int q) {
                         using KokkosExt::min;
                         new_offset(q) = min(offset(q + 1) - offset(q),
                                             getK(Access::get(queries, q)));
                       });

  exclusivePrefixSum(space, new_offset);

  int const n_truncated_results = lastElement(new_offset);
  Kokkos::View<int *, DeviceType> new_indices(indices.label(),
                                              n_truncated_results);
  Kokkos::View<int *, DeviceType> new_ranks(ranks.label(), n_truncated_results);

  using PairIndexDistance = Kokkos::pair<Kokkos::Array<int, 2>, float>;
  struct CompareDistance
  {
    KOKKOS_INLINE_FUNCTION bool operator()(PairIndexDistance const &lhs,
                                           PairIndexDistance const &rhs)
    {
      // reverse order (larger distance means lower priority)
      return lhs.second > rhs.second;
    }
  };

  int const n_results = lastElement(offset);
  Kokkos::View<PairIndexDistance *, DeviceType> buffer(
      Kokkos::ViewAllocateWithoutInitializing(
          "ArborX::DistributedTree::query::filterResults::buffer"),
      n_results);
  using PriorityQueue =
      Details::PriorityQueue<PairIndexDistance, CompareDistance,
                             UnmanagedStaticVector<PairIndexDistance>>;

  Kokkos::parallel_for(
      "ArborX::DistributedTree::query::truncate_results",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
      KOKKOS_LAMBDA(int q) {
        if (offset(q + 1) > offset(q))
        {
          auto local_buffer = Kokkos::subview(
              buffer, Kokkos::make_pair(offset(q), offset(q + 1)));
          PriorityQueue queue(UnmanagedStaticVector<PairIndexDistance>(
              local_buffer.data(), local_buffer.size()));
          for (int i = offset(q); i < offset(q + 1); ++i)
          {
            queue.emplace(Kokkos::Array<int, 2>{{indices(i), ranks(i)}},
                          distances(i));
          }

          int count = 0;
          while (!queue.empty() && count < getK(Access::get(queries, q)))
          {
            new_indices(new_offset(q) + count) = queue.top().first[0];
            new_ranks(new_offset(q) + count) = queue.top().first[1];
            queue.pop();
            ++count;
          }
        }
      });
  indices = new_indices;
  ranks = new_ranks;
  offset = new_offset;

  Kokkos::Profiling::popRegion();
}

} // namespace Details
} // namespace ArborX

#endif
