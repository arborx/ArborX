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
#ifndef ARBORX_DETAILS_DISTRIBUTED_SEARCH_TREE_IMPL_HPP
#define ARBORX_DETAILS_DISTRIBUTED_SEARCH_TREE_IMPL_HPP

#include <ArborX_DetailsDistributor.hpp>
#include <ArborX_DetailsKokkosExt.hpp> // min, max
#include <ArborX_DetailsPriorityQueue.hpp>
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Predicates.hpp>

#include <Kokkos_Atomic.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Sort.hpp>
#include <Kokkos_View.hpp>

#include <numeric> // accumulate

#include <mpi.h>

namespace ArborX
{

template <typename DeviceType>
class DistributedSearchTree;

namespace Details
{

template <typename DeviceType>
struct DistributedSearchTreeImpl
{
  using ExecutionSpace = typename DeviceType::execution_space;

  // spatial queries
  template <typename Query>
  static void queryDispatch(Details::SpatialPredicateTag,
                            DistributedSearchTree<DeviceType> const &tree,
                            Kokkos::View<Query *, DeviceType> queries,
                            Kokkos::View<int *, DeviceType> &indices,
                            Kokkos::View<int *, DeviceType> &offset,
                            Kokkos::View<int *, DeviceType> &ranks);

  // nearest neighbors queries
  template <typename Query>
  static void
  queryDispatch(Details::NearestPredicateTag,
                DistributedSearchTree<DeviceType> const &tree,
                Kokkos::View<Query *, DeviceType> queries,
                Kokkos::View<int *, DeviceType> &indices,
                Kokkos::View<int *, DeviceType> &offset,
                Kokkos::View<int *, DeviceType> &ranks,
                Kokkos::View<double *, DeviceType> *distances_ptr = nullptr);

  template <typename Query>
  static void queryDispatch(Details::NearestPredicateTag tag,
                            DistributedSearchTree<DeviceType> const &tree,
                            Kokkos::View<Query *, DeviceType> queries,
                            Kokkos::View<int *, DeviceType> &indices,
                            Kokkos::View<int *, DeviceType> &offset,
                            Kokkos::View<int *, DeviceType> &ranks,
                            Kokkos::View<double *, DeviceType> &distances)
  {
    queryDispatch(tag, tree, queries, indices, offset, ranks, &distances);
  }

  template <typename Query>
  static void deviseStrategy(Kokkos::View<Query *, DeviceType> queries,
                             DistributedSearchTree<DeviceType> const &tree,
                             Kokkos::View<int *, DeviceType> &indices,
                             Kokkos::View<int *, DeviceType> &offset,
                             Kokkos::View<double *, DeviceType> &);

  template <typename Query>
  static void reassessStrategy(Kokkos::View<Query *, DeviceType> queries,
                               DistributedSearchTree<DeviceType> const &tree,
                               Kokkos::View<int *, DeviceType> &indices,
                               Kokkos::View<int *, DeviceType> &offset,
                               Kokkos::View<double *, DeviceType> &distances);

  template <typename Query>
  static void forwardQueries(MPI_Comm comm,
                             Kokkos::View<Query *, DeviceType> queries,
                             Kokkos::View<int *, DeviceType> indices,
                             Kokkos::View<int *, DeviceType> offset,
                             Kokkos::View<Query *, DeviceType> &fwd_queries,
                             Kokkos::View<int *, DeviceType> &fwd_ids,
                             Kokkos::View<int *, DeviceType> &fwd_ranks);

  static void communicateResultsBack(
      MPI_Comm comm, Kokkos::View<int *, DeviceType> &indices,
      Kokkos::View<int *, DeviceType> offset,
      Kokkos::View<int *, DeviceType> &ranks,
      Kokkos::View<int *, DeviceType> &ids,
      Kokkos::View<double *, DeviceType> *distances_ptr = nullptr);

  template <typename Query>
  static void filterResults(Kokkos::View<Query *, DeviceType> queries,
                            Kokkos::View<double *, DeviceType> distances,
                            Kokkos::View<int *, DeviceType> &indices,
                            Kokkos::View<int *, DeviceType> &offset,
                            Kokkos::View<int *, DeviceType> &ranks);

  template <typename View, typename... OtherViews>
  static void sortResults(View keys, OtherViews... other_views);

  static void countResults(int n_queries,
                           Kokkos::View<int *, DeviceType> query_ids,
                           Kokkos::View<int *, DeviceType> &offset);

  template <typename View>
  static typename std::enable_if<Kokkos::is_view<View>::value>::type
  sendAcrossNetwork(Distributor const &distributor, View exports,
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
  return Kokkos::View<typename View::traits::data_type, Kokkos::LayoutRight,
                      typename View::traits::host_mirror_space>(
      std::string(src.label()).append("_layout_right_mirror"),
      src.dimension_0(), src.dimension_1(), src.dimension_2(),
      src.dimension_3(), src.dimension_4(), src.dimension_5(),
      src.dimension_6(), src.dimension_7());
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
template <typename View>
typename std::enable_if<Kokkos::is_view<View>::value>::type
DistributedSearchTreeImpl<DeviceType>::sendAcrossNetwork(
    Distributor const &distributor, View exports,
    typename View::non_const_type imports)
{
  ARBORX_ASSERT(
      (exports.dimension_0() == distributor.getTotalSendLength()) &&
      (imports.dimension_0() == distributor.getTotalReceiveLength()) &&
      (exports.dimension_1() == imports.dimension_1()) &&
      (exports.dimension_2() == imports.dimension_2()) &&
      (exports.dimension_3() == imports.dimension_3()) &&
      (exports.dimension_4() == imports.dimension_4()) &&
      (exports.dimension_5() == imports.dimension_5()) &&
      (exports.dimension_6() == imports.dimension_6()) &&
      (exports.dimension_7() == imports.dimension_7()));

  auto const num_packets = exports.dimension_1() * exports.dimension_2() *
                           exports.dimension_3() * exports.dimension_4() *
                           exports.dimension_5() * exports.dimension_6() *
                           exports.dimension_7();

  auto exports_host = create_layout_right_mirror_view(exports);
  Kokkos::deep_copy(exports_host, exports);

  auto imports_host = create_layout_right_mirror_view(imports);

  using NonConstValueType = typename View::non_const_value_type;
  using ConstValueType = typename View::const_value_type;

  Kokkos::View<ConstValueType *, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      export_buffer(exports_host.data(), exports_host.size());

  Kokkos::View<NonConstValueType *, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      import_buffer(imports_host.data(), imports_host.size());

  distributor.doPostsAndWaits(export_buffer, num_packets, import_buffer);

  Kokkos::deep_copy(imports, imports_host);
}

template <typename DeviceType>
template <typename Query>
void DistributedSearchTreeImpl<DeviceType>::deviseStrategy(
    Kokkos::View<Query *, DeviceType> queries,
    DistributedSearchTree<DeviceType> const &tree,
    Kokkos::View<int *, DeviceType> &indices,
    Kokkos::View<int *, DeviceType> &offset,
    Kokkos::View<double *, DeviceType> &)
{
  auto const &top_tree = tree._top_tree;
  auto const &bottom_tree_sizes = tree._bottom_tree_sizes;

  // Find the k nearest local trees.
  top_tree.query(queries, indices, offset);

  // Accumulate total leave count in the local trees until it reaches k which
  // is the number of neighbors queried for.  Stop if local trees get
  // empty because it means that they are no more leaves and there is no point
  // on forwarding queries to leafless trees.
  auto const n_queries = queries.extent(0);
  Kokkos::View<int *, DeviceType> new_offset(offset.label(), n_queries + 1);
  Kokkos::deep_copy(new_offset, 0);
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("bottom_trees_with_required_cumulated_leaves_count"),
      Kokkos::RangePolicy<ExecutionSpace>(0, n_queries), KOKKOS_LAMBDA(int i) {
        int leaves_count = 0;
        int const n_nearest_neighbors = queries(i)._k;
        for (int j = offset(i); j < offset(i + 1); ++j)
        {
          int const bottom_tree_size = bottom_tree_sizes(indices(j));
          if ((bottom_tree_size == 0) || (leaves_count >= n_nearest_neighbors))
            break;
          leaves_count += bottom_tree_size;
          ++new_offset(i);
        }
      });
  Kokkos::fence();

  exclusivePrefixSum(new_offset);

  // Truncate results so that queries will only be forwarded to as many local
  // trees as necessary to find k neighbors.
  Kokkos::View<int *, DeviceType> new_indices(indices.label(),
                                              lastElement(new_offset));
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("truncate_before_forwarding"),
      Kokkos::RangePolicy<ExecutionSpace>(0, n_queries), KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < new_offset(i + 1) - new_offset(i); ++j)
          new_indices(new_offset(i) + j) = indices(offset(i) + j);
      });
  Kokkos::fence();

  offset = new_offset;
  indices = new_indices;
}

template <typename DeviceType>
template <typename Query>
void DistributedSearchTreeImpl<DeviceType>::reassessStrategy(
    Kokkos::View<Query *, DeviceType> queries,
    DistributedSearchTree<DeviceType> const &tree,
    Kokkos::View<int *, DeviceType> &indices,
    Kokkos::View<int *, DeviceType> &offset,
    Kokkos::View<double *, DeviceType> &distances)
{
  auto const &top_tree = tree._top_tree;
  auto const n_queries = queries.extent(0);

  // Determine distance to the farthest neighbor found so far.
  Kokkos::View<double *, DeviceType> farthest_distances("distances", n_queries);
  Kokkos::deep_copy(farthest_distances, 0.);
  // NOTE: in principle distances( j ) are arranged in ascending order for
  // offset( i ) <= j < offset( i + 1 ) so max() is not necessary.
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("most_distant_neighbor_so_far"),
      Kokkos::RangePolicy<ExecutionSpace>(0, n_queries), KOKKOS_LAMBDA(int i) {
        using KokkosExt::max;
        for (int j = offset(i); j < offset(i + 1); ++j)
          farthest_distances(i) = max(farthest_distances(i), distances(j));
      });
  Kokkos::fence();

  // Identify what ranks may have leaves that are within that distance.
  Kokkos::View<Within *, DeviceType> within_queries("queries", n_queries);
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("bottom_trees_within_that_distance"),
      Kokkos::RangePolicy<ExecutionSpace>(0, n_queries), KOKKOS_LAMBDA(int i) {
        within_queries(i) = within(queries(i)._geometry, farthest_distances(i));
      });
  Kokkos::fence();

  top_tree.query(within_queries, indices, offset);
  // NOTE: in principle, we could perform within queries on the bottom_tree
  // rather than nearest queries.
}

template <typename DeviceType>
template <typename Query>
void DistributedSearchTreeImpl<DeviceType>::queryDispatch(
    Details::NearestPredicateTag, DistributedSearchTree<DeviceType> const &tree,
    Kokkos::View<Query *, DeviceType> queries,
    Kokkos::View<int *, DeviceType> &indices,
    Kokkos::View<int *, DeviceType> &offset,
    Kokkos::View<int *, DeviceType> &ranks,
    Kokkos::View<double *, DeviceType> *distances_ptr)
{
  auto const &bottom_tree = tree._bottom_tree;
  auto comm = tree._comm;

  Kokkos::View<double *, DeviceType> distances("distances");
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
  using Strategy = void (*)(Kokkos::View<Query *, DeviceType>,
                            DistributedSearchTree<DeviceType> const &,
                            Kokkos::View<int *, DeviceType> &,
                            Kokkos::View<int *, DeviceType> &,
                            Kokkos::View<double *, DeviceType> &);
  for (auto implementStrategy :
       {static_cast<Strategy>(
            DistributedSearchTreeImpl<DeviceType>::deviseStrategy),
        static_cast<Strategy>(
            DistributedSearchTreeImpl<DeviceType>::reassessStrategy)})
  {
    implementStrategy(queries, tree, indices, offset, distances);

    ////////////////////////////////////////////////////////////////////////////
    // Forward queries
    ////////////////////////////////////////////////////////////////////////////
    Kokkos::View<int *, DeviceType> ids("query_ids");
    Kokkos::View<Query *, DeviceType> fwd_queries("fwd_queries");
    forwardQueries(comm, queries, indices, offset, fwd_queries, ids, ranks);
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // Perform queries that have been received
    ////////////////////////////////////////////////////////////////////////////
    bottom_tree.query(fwd_queries, indices, offset, distances);
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // Communicate results back
    ////////////////////////////////////////////////////////////////////////////
    communicateResultsBack(comm, indices, offset, ranks, ids, &distances);
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // Merge results
    ////////////////////////////////////////////////////////////////////////////
    int const n_queries = queries.extent_int(0);
    countResults(n_queries, ids, offset);
    sortResults(ids, indices, ranks, distances);
    filterResults(queries, distances, indices, offset, ranks);
    ////////////////////////////////////////////////////////////////////////////
  }
}

template <typename DeviceType>
template <typename Query>
void DistributedSearchTreeImpl<DeviceType>::queryDispatch(
    Details::SpatialPredicateTag, DistributedSearchTree<DeviceType> const &tree,
    Kokkos::View<Query *, DeviceType> queries,
    Kokkos::View<int *, DeviceType> &indices,
    Kokkos::View<int *, DeviceType> &offset,
    Kokkos::View<int *, DeviceType> &ranks)
{
  auto const &top_tree = tree._top_tree;
  auto const &bottom_tree = tree._bottom_tree;
  auto comm = tree._comm;

  ////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////
  top_tree.query(queries, indices, offset);
  ////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////
  // Forward queries
  ////////////////////////////////////////////////////////////////////////////
  Kokkos::View<int *, DeviceType> ids("query_ids");
  Kokkos::View<Query *, DeviceType> fwd_queries("fwd_queries");
  forwardQueries(comm, queries, indices, offset, fwd_queries, ids, ranks);
  ////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////
  // Perform queries that have been received
  ////////////////////////////////////////////////////////////////////////////
  bottom_tree.query(fwd_queries, indices, offset);
  ////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////
  // Communicate results back
  ////////////////////////////////////////////////////////////////////////////
  communicateResultsBack(comm, indices, offset, ranks, ids);
  ////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////////////////////////////////////////////////
  // Merge results
  ////////////////////////////////////////////////////////////////////////////
  int const n_queries = queries.extent_int(0);
  countResults(n_queries, ids, offset);
  sortResults(ids, indices, ranks);
  ////////////////////////////////////////////////////////////////////////////
}

// FIXME: for some reason Kokkos::BinSort::sort() was not const.
// If https://github.com/kokkos/kokkos/pull/1310 makes it into master in
// Trilinos, we might want to pass bin_sort by const reference.
template <typename BinSort>
void applyPermutations(BinSort &)
{
  // do nothing
}

template <typename BinSort, typename View, typename... OtherViews>
void applyPermutations(BinSort &bin_sort, View view, OtherViews... other_views)
{
  ARBORX_ASSERT(bin_sort.get_permute_vector().extent(0) == view.extent(0));
  bin_sort.sort(view);
  applyPermutations(bin_sort, other_views...);
}

template <typename DeviceType>
template <typename View, typename... OtherViews>
void DistributedSearchTreeImpl<DeviceType>::sortResults(
    View keys, OtherViews... other_views)
{
  auto const n = keys.extent(0);
  // If they were no queries, min_val and max_val values won't change after
  // the parallel reduce (they are initialized to +infty and -infty
  // respectively) and the sort will hang.
  if (n == 0)
    return;

  using Comp = Kokkos::BinOp1D<View>;
  using Value = typename View::non_const_value_type;

  Kokkos::Experimental::MinMaxScalar<Value> result;
  Kokkos::Experimental::MinMax<Value> reducer(result);
  parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(0, n),
                  Kokkos::Impl::min_max_functor<View>(keys), reducer);
  if (result.min_val == result.max_val)
    return;
  Kokkos::BinSort<View, Comp> bin_sort(
      keys, Comp(n / 2, result.min_val, result.max_val), true);
  bin_sort.create_permute_vector();
  applyPermutations(bin_sort, other_views...);
  Kokkos::fence();
}

template <typename DeviceType>
void DistributedSearchTreeImpl<DeviceType>::countResults(
    int n_queries, Kokkos::View<int *, DeviceType> query_ids,
    Kokkos::View<int *, DeviceType> &offset)
{
  int const nnz = query_ids.extent(0);

  Kokkos::realloc(offset, n_queries + 1);
  Kokkos::deep_copy(offset, 0);

  Kokkos::parallel_for(ARBORX_MARK_REGION("count_results_per_query"),
                       Kokkos::RangePolicy<ExecutionSpace>(0, nnz),
                       KOKKOS_LAMBDA(int i) {
                         Kokkos::atomic_increment(&offset(query_ids(i)));
                       });
  Kokkos::fence();

  exclusivePrefixSum(offset);
}

template <typename DeviceType>
template <typename Query>
void DistributedSearchTreeImpl<DeviceType>::forwardQueries(
    MPI_Comm comm, Kokkos::View<Query *, DeviceType> queries,
    Kokkos::View<int *, DeviceType> indices,
    Kokkos::View<int *, DeviceType> offset,
    Kokkos::View<Query *, DeviceType> &fwd_queries,
    Kokkos::View<int *, DeviceType> &fwd_ids,
    Kokkos::View<int *, DeviceType> &fwd_ranks)
{
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  Distributor distributor(comm);

  int const n_queries = queries.extent(0);
  int const n_exports = lastElement(offset);
  int const n_imports = distributor.createFromSends(indices);

  Kokkos::View<Query *, DeviceType> exports(queries.label(), n_exports);
  Kokkos::parallel_for(ARBORX_MARK_REGION("forward_queries_fill_buffer"),
                       Kokkos::RangePolicy<ExecutionSpace>(0, n_queries),
                       KOKKOS_LAMBDA(int q) {
                         for (int i = offset(q); i < offset(q + 1); ++i)
                         {
                           exports(i) = queries(q);
                         }
                       });
  Kokkos::fence();

  Kokkos::View<int *, DeviceType> export_ranks("export_ranks", n_exports);
  Kokkos::deep_copy(export_ranks, comm_rank);

  Kokkos::View<int *, DeviceType> import_ranks("import_ranks", n_imports);
  sendAcrossNetwork(distributor, export_ranks, import_ranks);

  Kokkos::View<int *, DeviceType> export_ids("export_ids", n_exports);
  Kokkos::parallel_for(ARBORX_MARK_REGION("forward_queries_fill_ids"),
                       Kokkos::RangePolicy<ExecutionSpace>(0, n_queries),
                       KOKKOS_LAMBDA(int q) {
                         for (int i = offset(q); i < offset(q + 1); ++i)
                         {
                           export_ids(i) = q;
                         }
                       });
  Kokkos::fence();
  Kokkos::View<int *, DeviceType> import_ids("import_ids", n_imports);
  sendAcrossNetwork(distributor, export_ids, import_ids);

  // Send queries across the network
  Kokkos::View<Query *, DeviceType> imports(queries.label(), n_imports);
  sendAcrossNetwork(distributor, exports, imports);

  fwd_queries = imports;
  fwd_ids = import_ids;
  fwd_ranks = import_ranks;
}

template <typename DeviceType>
void DistributedSearchTreeImpl<DeviceType>::communicateResultsBack(
    MPI_Comm comm, Kokkos::View<int *, DeviceType> &indices,
    Kokkos::View<int *, DeviceType> offset,
    Kokkos::View<int *, DeviceType> &ranks,
    Kokkos::View<int *, DeviceType> &ids,
    Kokkos::View<double *, DeviceType> *distances_ptr)
{
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  int const n_fwd_queries = offset.extent_int(0) - 1;
  int const n_exports = lastElement(offset);
  Kokkos::View<int *, DeviceType> export_ranks(ranks.label(), n_exports);
  Kokkos::parallel_for(ARBORX_MARK_REGION("setup_communication_plan"),
                       Kokkos::RangePolicy<ExecutionSpace>(0, n_fwd_queries),
                       KOKKOS_LAMBDA(int q) {
                         for (int i = offset(q); i < offset(q + 1); ++i)
                         {
                           export_ranks(i) = ranks(q);
                         }
                       });
  Kokkos::fence();

  Distributor distributor(comm);
  int const n_imports = distributor.createFromSends(export_ranks);

  // export_ranks already has adequate size since it was used as a buffer to
  // make the new communication plan.
  Kokkos::deep_copy(export_ranks, comm_rank);

  Kokkos::View<int *, DeviceType> export_ids(ids.label(), n_exports);
  Kokkos::parallel_for(ARBORX_MARK_REGION("fill_buffer"),
                       Kokkos::RangePolicy<ExecutionSpace>(0, n_fwd_queries),
                       KOKKOS_LAMBDA(int q) {
                         for (int i = offset(q); i < offset(q + 1); ++i)
                         {
                           export_ids(i) = ids(q);
                         }
                       });
  Kokkos::fence();
  Kokkos::View<int *, DeviceType> export_indices = indices;

  Kokkos::View<int *, DeviceType> import_indices(indices.label(), n_imports);
  Kokkos::View<int *, DeviceType> import_ranks(ranks.label(), n_imports);
  Kokkos::View<int *, DeviceType> import_ids(ids.label(), n_imports);
  sendAcrossNetwork(distributor, export_indices, import_indices);
  sendAcrossNetwork(distributor, export_ranks, import_ranks);
  sendAcrossNetwork(distributor, export_ids, import_ids);

  ids = import_ids;
  ranks = import_ranks;
  indices = import_indices;

  if (distances_ptr)
  {
    Kokkos::View<double *, DeviceType> &distances = *distances_ptr;
    Kokkos::View<double *, DeviceType> export_distances = distances;
    Kokkos::View<double *, DeviceType> import_distances(distances.label(),
                                                        n_imports);
    sendAcrossNetwork(distributor, export_distances, import_distances);
    distances = import_distances;
  }
}

template <typename DeviceType>
template <typename Query>
void DistributedSearchTreeImpl<DeviceType>::filterResults(
    Kokkos::View<Query *, DeviceType> queries,
    Kokkos::View<double *, DeviceType> distances,
    Kokkos::View<int *, DeviceType> &indices,
    Kokkos::View<int *, DeviceType> &offset,
    Kokkos::View<int *, DeviceType> &ranks)
{
  int const n_queries = queries.extent_int(0);
  // truncated views are prefixed with an underscore
  Kokkos::View<int *, DeviceType> new_offset(offset.label(), n_queries + 1);
  Kokkos::deep_copy(new_offset, 0);

  Kokkos::parallel_for(
      ARBORX_MARK_REGION("discard_results"),
      Kokkos::RangePolicy<ExecutionSpace>(0, n_queries), KOKKOS_LAMBDA(int q) {
        using KokkosExt::min;
        new_offset(q) = min(offset(q + 1) - offset(q), queries(q)._k);
      });
  Kokkos::fence();

  exclusivePrefixSum(new_offset);

  int const n_truncated_results = lastElement(new_offset);
  Kokkos::View<int *, DeviceType> new_indices(indices.label(),
                                              n_truncated_results);
  Kokkos::View<int *, DeviceType> new_ranks(ranks.label(), n_truncated_results);

  using PairIndexDistance = Kokkos::pair<Kokkos::Array<int, 2>, double>;
  struct CompareDistance
  {
    KOKKOS_INLINE_FUNCTION bool operator()(PairIndexDistance const &lhs,
                                           PairIndexDistance const &rhs)
    {
      // reverse order (larger distance means lower priority)
      return lhs.second > rhs.second;
    }
  };
  using PriorityQueue =
      Details::PriorityQueue<PairIndexDistance, CompareDistance>;

  Kokkos::parallel_for(
      ARBORX_MARK_REGION("truncate_results"),
      Kokkos::RangePolicy<ExecutionSpace>(0, n_queries), KOKKOS_LAMBDA(int q) {
        PriorityQueue queue;
        for (int i = offset(q); i < offset(q + 1); ++i)
          queue.emplace(Kokkos::Array<int, 2>{{indices(i), ranks(i)}},
                        distances(i));

        int count = 0;
        while (!queue.empty() && count < queries(q)._k)
        {
          new_indices(new_offset(q) + count) = queue.top().first[0];
          new_ranks(new_offset(q) + count) = queue.top().first[1];
          queue.pop();
          ++count;
        }
      });
  Kokkos::fence();
  indices = new_indices;
  ranks = new_ranks;
  offset = new_offset;
}

} // namespace Details
} // namespace ArborX

#endif
