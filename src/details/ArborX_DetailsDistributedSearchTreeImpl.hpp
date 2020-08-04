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
#ifndef ARBORX_DETAILS_DISTRIBUTED_SEARCH_TREE_IMPL_HPP
#define ARBORX_DETAILS_DISTRIBUTED_SEARCH_TREE_IMPL_HPP

#include <ArborX_Config.hpp>

#include <ArborX_DetailsDistributor.hpp>
#include <ArborX_DetailsKokkosExt.hpp> // min, max
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

struct CallbackDefaultSpatialPredicateWithRank
{
  int _rank;
  template <typename Query, typename Insert>
  KOKKOS_FUNCTION void operator()(Query const &, int index,
                                  Insert const &insert) const
  {
    insert(Kokkos::pair<int, int>{index, _rank});
  }
};

template <typename DeviceType>
struct DistributedSearchTreeImpl
{
  // spatial queries
  template <typename DistributedTree, typename ExecutionSpace,
            typename Predicates, typename Indices, typename Offset,
            typename Ranks>
  static std::enable_if_t<Kokkos::is_view<Indices>{} &&
                          Kokkos::is_view<Offset>{} && Kokkos::is_view<Ranks>{}>
  queryDispatch(SpatialPredicateTag, DistributedTree const &tree,
                ExecutionSpace const &space, Predicates const &queries,
                Indices &indices, Offset &offset, Ranks &ranks)
  {
    Kokkos::View<Kokkos::pair<int, int> *, DeviceType> out("pairs_index_rank",
                                                           0);
    int comm_rank;
    MPI_Comm_rank(tree._comm, &comm_rank);
    queryDispatch(SpatialPredicateTag{}, tree, space, queries,
                  CallbackDefaultSpatialPredicateWithRank{comm_rank}, out,
                  offset);
    auto const n = out.extent(0);
    reallocWithoutInitializing(indices, n);
    reallocWithoutInitializing(ranks, n);
    Kokkos::parallel_for(ARBORX_MARK_REGION("split_pairs"),
                         Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
                         KOKKOS_LAMBDA(int i) {
                           indices(i) = out(i).first;
                           ranks(i) = out(i).second;
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
  queryDispatch(NearestPredicateTag, DistributedTree const &tree,
                ExecutionSpace const &space, Predicates const &queries,
                Indices &indices, Offset &offset, Ranks &ranks,
                Distances *distances_ptr = nullptr);

  template <typename DistributedTree, typename ExecutionSpace,
            typename Predicates, typename Indices, typename Offset,
            typename Ranks, typename Distances>
  static std::enable_if_t<
      Kokkos::is_view<Indices>{} && Kokkos::is_view<Offset>{} &&
      Kokkos::is_view<Ranks>{} && Kokkos::is_view<Distances>{}>
  queryDispatch(NearestPredicateTag tag, DistributedTree const &tree,
                ExecutionSpace const &space, Predicates const &queries,
                Indices &indices, Offset &offset, Ranks &ranks,
                Distances &distances)
  {
    queryDispatch(tag, tree, space, queries, indices, offset, ranks,
                  &distances);
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
                                     OutputView &view,
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

template <typename View>
inline Kokkos::View<typename View::traits::data_type, Kokkos::LayoutRight,
                    typename View::traits::host_mirror_space>
create_layout_right_mirror_view(
    View const &src,
    typename std::enable_if<!(
        (std::is_same<typename View::traits::array_layout,
                      Kokkos::LayoutRight>::value ||
         (View::rank == 1 && !std::is_same<typename View::traits::array_layout,
                                           Kokkos::LayoutStride>::value)) &&
        std::is_same<typename View::traits::memory_space,
                     typename View::traits::host_mirror_space::memory_space>::
            value)>::type * = 0)
{
  constexpr int pointer_depth =
      internal::PointerDepth<typename View::traits::data_type>::value;
  return Kokkos::View<typename View::traits::data_type, Kokkos::LayoutRight,
                      typename View::traits::host_mirror_space>(
      std::string(src.label()).append("_layout_right_mirror"), src.extent(0),
      pointer_depth > 1 ? src.extent(1) : KOKKOS_INVALID_INDEX,
      pointer_depth > 2 ? src.extent(2) : KOKKOS_INVALID_INDEX,
      pointer_depth > 3 ? src.extent(3) : KOKKOS_INVALID_INDEX,
      pointer_depth > 4 ? src.extent(4) : KOKKOS_INVALID_INDEX,
      pointer_depth > 5 ? src.extent(5) : KOKKOS_INVALID_INDEX,
      pointer_depth > 6 ? src.extent(6) : KOKKOS_INVALID_INDEX,
      pointer_depth > 7 ? src.extent(7) : KOKKOS_INVALID_INDEX);
}

template <typename View>
inline Kokkos::View<typename View::traits::data_type, Kokkos::LayoutRight,
                    typename View::traits::host_mirror_space>
create_layout_right_mirror_view(
    View const &src,
    typename std::enable_if<
        ((std::is_same<typename View::traits::array_layout,
                       Kokkos::LayoutRight>::value ||
          (View::rank == 1 && !std::is_same<typename View::traits::array_layout,
                                            Kokkos::LayoutStride>::value)) &&
         std::is_same<typename View::traits::memory_space,
                      typename View::traits::host_mirror_space::memory_space>::
             value)>::type * = 0)
{
  return src;
}

template <typename DeviceType>
template <typename ExecutionSpace, typename View>
typename std::enable_if<Kokkos::is_view<View>::value>::type
DistributedSearchTreeImpl<DeviceType>::sendAcrossNetwork(
    ExecutionSpace const &space, Distributor<DeviceType> const &distributor,
    View exports, typename View::non_const_type imports)
{
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
  using ConstValueType = typename View::const_value_type;

  Kokkos::View<ConstValueType *, typename View::device_type,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      export_buffer(exports.data(), exports.size());

#ifndef ARBORX_USE_CUDA_AWARE_MPI
  auto imports_host = create_layout_right_mirror_view(imports);

  Kokkos::View<NonConstValueType *, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      import_buffer(imports_host.data(), imports_host.size());

  distributor.doPostsAndWaits(space, export_buffer, num_packets, import_buffer);
  Kokkos::deep_copy(space, imports, imports_host);
#else
  Kokkos::View<NonConstValueType *, typename View::device_type,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      import_buffer(imports.data(), imports.size());

  distributor.doPostsAndWaits(space, export_buffer, num_packets, import_buffer);
#endif
}

template <typename DeviceType>
template <typename DistributedTree, typename ExecutionSpace,
          typename Predicates, typename Indices, typename Offset,
          typename Distances>
void DistributedSearchTreeImpl<DeviceType>::deviseStrategy(
    ExecutionSpace const &space, Predicates const &queries,
    DistributedTree const &tree, Indices &indices, Offset &offset, Distances &)
{
  auto const &top_tree = tree._top_tree;
  auto const &bottom_tree_sizes = tree._bottom_tree_sizes;

  // Find the k nearest local trees.
  top_tree.query(space, queries, indices, offset);

  // Accumulate total leave count in the local trees until it reaches k which
  // is the number of neighbors queried for.  Stop if local trees get
  // empty because it means that they are no more leaves and there is no point
  // on forwarding queries to leafless trees.
  using Access = AccessTraits<Predicates, PredicatesTag>;
  auto const n_queries = Access::size(queries);
  Kokkos::View<int *, DeviceType> new_offset(
      Kokkos::view_alloc(offset.label(), space), n_queries + 1);
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("bottom_trees_with_required_cumulated_leaves_count"),
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
      ARBORX_MARK_REGION("truncate_before_forwarding"),
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
      KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < new_offset(i + 1) - new_offset(i); ++j)
          new_indices(new_offset(i) + j) = indices(offset(i) + j);
      });

  offset = new_offset;
  indices = new_indices;
}

template <typename DeviceType>
template <typename DistributedTree, typename ExecutionSpace,
          typename Predicates, typename Indices, typename Offset,
          typename Distances>
void DistributedSearchTreeImpl<DeviceType>::reassessStrategy(
    ExecutionSpace const &space, Predicates const &queries,
    DistributedTree const &tree, Indices &indices, Offset &offset,
    Distances &distances)
{
  auto const &top_tree = tree._top_tree;
  using Access = AccessTraits<Predicates, PredicatesTag>;
  auto const n_queries = Access::size(queries);

  // Determine distance to the farthest neighbor found so far.
  Kokkos::View<float *, DeviceType> farthest_distances(
      Kokkos::ViewAllocateWithoutInitializing("distances"), n_queries);
  // NOTE: in principle distances( j ) are arranged in ascending order for
  // offset( i ) <= j < offset( i + 1 ) so max() is not necessary.
  Kokkos::parallel_for(ARBORX_MARK_REGION("most_distant_neighbor_so_far"),
                       Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
                       KOKKOS_LAMBDA(int i) {
                         using KokkosExt::max;
                         farthest_distances(i) = 0.;
                         for (int j = offset(i); j < offset(i + 1); ++j)
                           farthest_distances(i) =
                               max(farthest_distances(i), distances(j));
                       });

  // Identify what ranks may have leaves that are within that distance.
  Kokkos::View<decltype(intersects(Sphere{})) *, DeviceType> radius_searches(
      Kokkos::ViewAllocateWithoutInitializing("queries"), n_queries);
  Kokkos::parallel_for(ARBORX_MARK_REGION("bottom_trees_within_that_distance"),
                       Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
                       KOKKOS_LAMBDA(int i) {
                         radius_searches(i) = intersects(
                             Sphere{getGeometry(Access::get(queries, i)),
                                    farthest_distances(i)});
                       });

  top_tree.query(space, radius_searches, indices, offset);
  // NOTE: in principle, we could perform radius searches on the bottom_tree
  // rather than nearest queries.
}

template <typename DeviceType>
template <typename DistributedTree, typename ExecutionSpace,
          typename Predicates, typename Indices, typename Offset,
          typename Ranks, typename Distances>
std::enable_if_t<Kokkos::is_view<Indices>{} && Kokkos::is_view<Offset>{} &&
                 Kokkos::is_view<Ranks>{} && Kokkos::is_view<Distances>{}>
DistributedSearchTreeImpl<DeviceType>::queryDispatch(
    NearestPredicateTag, DistributedTree const &tree,
    ExecutionSpace const &space, Predicates const &queries, Indices &indices,
    Offset &offset, Ranks &ranks, Distances *distances_ptr)
{
  auto const &bottom_tree = tree._bottom_tree;
  auto comm = tree._comm;

  Distances distances("distances", 0);
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

  // NOTE: compiler would not deduce __range for the braced-init-list but I
  // got it to work with the static_cast to function pointers.
  using Strategy =
      void (*)(ExecutionSpace const &, Predicates const &,
               DistributedTree const &, Indices &, Offset &, Distances &);
  for (auto implementStrategy :
       {static_cast<Strategy>(
            DistributedSearchTreeImpl<DeviceType>::deviseStrategy),
        static_cast<Strategy>(
            DistributedSearchTreeImpl<DeviceType>::reassessStrategy)})
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
      Kokkos::View<int *, DeviceType> ids("query_ids", 0);
      Kokkos::View<Query *, DeviceType> fwd_queries("fwd_queries", 0);
      forwardQueries(comm, space, queries, indices, offset, fwd_queries, ids,
                     ranks);

      // Perform queries that have been received
      bottom_tree.query(space, fwd_queries, indices, offset, distances);

      // Communicate results back
      communicateResultsBack(comm, space, indices, offset, ranks, ids,
                             &distances);

      // Merge results
      int const n_queries = Access::size(queries);
      countResults(space, n_queries, ids, offset);
      sortResults(space, ids, indices, ranks, distances);
      filterResults(space, queries, distances, indices, offset, ranks);
    }
  }
}

template <typename DeviceType>
template <typename DistributedTree, typename ExecutionSpace,
          typename Predicates, typename OutputView, typename OffsetView,
          typename Callback>
std::enable_if_t<Kokkos::is_view<OutputView>{} && Kokkos::is_view<OffsetView>{}>
DistributedSearchTreeImpl<DeviceType>::queryDispatch(
    SpatialPredicateTag, DistributedTree const &tree,
    ExecutionSpace const &space, Predicates const &queries,
    Callback const &callback, OutputView &out, OffsetView &offset)
{
  auto const &top_tree = tree._top_tree;
  auto const &bottom_tree = tree._bottom_tree;
  auto comm = tree._comm;

  Kokkos::View<int *, DeviceType> indices("indices", 0);
  Kokkos::View<int *, DeviceType> ranks("ranks", 0);
  top_tree.query(space, queries, indices, offset);

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
    Kokkos::View<int *, DeviceType> ids("query_ids", 0);
    Kokkos::View<Query *, DeviceType> fwd_queries("fwd_queries", 0);
    forwardQueries(comm, space, queries, indices, offset, fwd_queries, ids,
                   ranks);

    // Perform queries that have been received
    bottom_tree.query(space, fwd_queries, callback, out, offset);

    // Communicate results back
    communicateResultsBack(comm, space, out, offset, ranks, ids);

    // Merge results
    int const n_queries = Access::size(queries);
    countResults(space, n_queries, ids, offset);
    sortResults(space, ids, out);
  }
}

template <typename DeviceType>
template <typename ExecutionSpace, typename View, typename... OtherViews>
void DistributedSearchTreeImpl<DeviceType>::sortResults(
    ExecutionSpace const &space, View keys, OtherViews... other_views)
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
  View keys_clone(Kokkos::ViewAllocateWithoutInitializing("keys"), keys.size());
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
void DistributedSearchTreeImpl<DeviceType>::countResults(
    ExecutionSpace const &space, int n_queries,
    Kokkos::View<int *, DeviceType> query_ids, OffsetView &offset)
{
  int const nnz = query_ids.extent(0);

  Kokkos::realloc(offset, n_queries + 1);

  Kokkos::parallel_for(ARBORX_MARK_REGION("count_results_per_query"),
                       Kokkos::RangePolicy<ExecutionSpace>(space, 0, nnz),
                       KOKKOS_LAMBDA(int i) {
                         Kokkos::atomic_increment(&offset(query_ids(i)));
                       });

  exclusivePrefixSum(space, offset);
}

template <typename DeviceType>
template <typename ExecutionSpace, typename Predicates, typename Ranks,
          typename Query>
void DistributedSearchTreeImpl<DeviceType>::forwardQueries(
    MPI_Comm comm, ExecutionSpace const &space, Predicates const &queries,
    Kokkos::View<int *, DeviceType> indices,
    Kokkos::View<int *, DeviceType> offset,
    Kokkos::View<Query *, DeviceType> &fwd_queries,
    Kokkos::View<int *, DeviceType> &fwd_ids, Ranks &fwd_ranks)
{
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
      Kokkos::ViewAllocateWithoutInitializing("queries"), n_exports);
  Kokkos::parallel_for(ARBORX_MARK_REGION("forward_queries_fill_buffer"),
                       Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
                       KOKKOS_LAMBDA(int q) {
                         for (int i = offset(q); i < offset(q + 1); ++i)
                         {
                           exports(i) = Access::get(queries, q);
                         }
                       });

  Kokkos::View<int *, DeviceType> export_ranks(
      Kokkos::ViewAllocateWithoutInitializing("export_ranks"), n_exports);
  Kokkos::deep_copy(space, export_ranks, comm_rank);

  Kokkos::View<int *, DeviceType> import_ranks(
      Kokkos::ViewAllocateWithoutInitializing("import_ranks"), n_imports);
  sendAcrossNetwork(space, distributor, export_ranks, import_ranks);

  Kokkos::View<int *, DeviceType> export_ids(
      Kokkos::ViewAllocateWithoutInitializing("export_ids"), n_exports);
  Kokkos::parallel_for(ARBORX_MARK_REGION("forward_queries_fill_ids"),
                       Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
                       KOKKOS_LAMBDA(int q) {
                         for (int i = offset(q); i < offset(q + 1); ++i)
                         {
                           export_ids(i) = q;
                         }
                       });
  Kokkos::View<int *, DeviceType> import_ids(
      Kokkos::ViewAllocateWithoutInitializing("import_ids"), n_imports);
  sendAcrossNetwork(space, distributor, export_ids, import_ids);

  // Send queries across the network
  Kokkos::View<Query *, DeviceType> imports(
      Kokkos::ViewAllocateWithoutInitializing("queries"), n_imports);
  sendAcrossNetwork(space, distributor, exports, imports);

  fwd_queries = imports;
  fwd_ids = import_ids;
  fwd_ranks = import_ranks;
}

template <typename DeviceType>
template <typename ExecutionSpace, typename OutputView, typename Ranks,
          typename Distances>
void DistributedSearchTreeImpl<DeviceType>::communicateResultsBack(
    MPI_Comm comm, ExecutionSpace const &space, OutputView &out,
    Kokkos::View<int *, DeviceType> offset, Ranks &ranks,
    Kokkos::View<int *, DeviceType> &ids, Distances *distances_ptr)
{
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
      ARBORX_MARK_REGION("fill_buffer"),
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
}

template <typename DeviceType>
template <typename ExecutionSpace, typename Predicates, typename Indices,
          typename Offset, typename Ranks>
void DistributedSearchTreeImpl<DeviceType>::filterResults(
    ExecutionSpace const &space, Predicates const &queries,
    Kokkos::View<float *, DeviceType> distances, Indices &indices,
    Offset &offset, Ranks &ranks)
{
  using Access = AccessTraits<Predicates, PredicatesTag>;
  int const n_queries = Access::size(queries);
  // truncated views are prefixed with an underscore
  Kokkos::View<int *, DeviceType> new_offset(offset.label(), n_queries + 1);

  Kokkos::parallel_for(ARBORX_MARK_REGION("discard_results"),
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
      Kokkos::ViewAllocateWithoutInitializing("buffer"), n_results);
  using PriorityQueue =
      Details::PriorityQueue<PairIndexDistance, CompareDistance,
                             UnmanagedStaticVector<PairIndexDistance>>;

  Kokkos::parallel_for(
      ARBORX_MARK_REGION("truncate_results"),
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
}

} // namespace Details
} // namespace ArborX

#endif
