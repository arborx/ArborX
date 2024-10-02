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
#ifndef ARBORX_DETAILS_DISTRIBUTED_TREE_UTILS_HPP
#define ARBORX_DETAILS_DISTRIBUTED_TREE_UTILS_HPP

#include <ArborX_Config.hpp>

#include <ArborX_Containers.hpp>
#include <ArborX_PriorityQueue.hpp>
#include <detail/ArborX_Distributor.hpp>
#include <kokkos_ext/ArborX_KokkosExtSort.hpp>
#include <kokkos_ext/ArborX_KokkosExtStdAlgorithms.hpp>
#include <kokkos_ext/ArborX_KokkosExtViewHelpers.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <mpi.h>

namespace ArborX::Details::DistributedTree
{

template <typename ExecutionSpace, typename QueryIdsView, typename OffsetView>
void countResults(ExecutionSpace const &space, int n_queries,
                  QueryIdsView const &query_ids, OffsetView &offset)
{
  int const nnz = query_ids.extent(0);

  Kokkos::realloc(Kokkos::view_alloc(space), offset, n_queries + 1);

  Kokkos::parallel_for(
      "ArborX::DistributedTree::query::count_results_per_query",
      Kokkos::RangePolicy(space, 0, nnz), KOKKOS_LAMBDA(int i) {
        Kokkos::atomic_increment(&offset(query_ids(i)));
      });

  KokkosExt::exclusive_scan(space, offset, offset, 0);
}

template <typename ExecutionSpace, typename Predicates, typename Indices,
          typename Offset, typename FwdQueries, typename FwdIds, typename Ranks>
void forwardQueries(MPI_Comm comm, ExecutionSpace const &space,
                    Predicates const &queries, Indices const &indices,
                    Offset const &offset, FwdQueries &fwd_queries,
                    FwdIds &fwd_ids, Ranks &fwd_ranks)
{
  std::string prefix = "ArborX::DistributedTree::query::forwardQueries";

  Kokkos::Profiling::ScopedRegion guard(prefix);

  using MemorySpace = typename Predicates::memory_space;
  using Query = typename Predicates::value_type;

  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  Distributor<MemorySpace> distributor(comm);

  int const n_queries = queries.size();
  int const n_exports = KokkosExt::lastElement(space, offset);
  int const n_imports = distributor.createFromSends(space, indices);

  {
    Kokkos::View<Query *, MemorySpace> export_queries(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           prefix + "::export_queries"),
        n_exports);
    Kokkos::parallel_for(
        prefix + "::forward_queries_fill_buffer",
        Kokkos::RangePolicy(space, 0, n_queries), KOKKOS_LAMBDA(int q) {
          for (int i = offset(q); i < offset(q + 1); ++i)
            export_queries(i) = queries(q);
        });

    KokkosExt::reallocWithoutInitializing(space, fwd_queries, n_imports);
    distributor.doPostsAndWaits(space, export_queries, fwd_queries);
  }

  {
    Kokkos::View<int *, MemorySpace> export_ranks(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           prefix + "::export_ranks"),
        n_exports);
    Kokkos::deep_copy(space, export_ranks, comm_rank);

    KokkosExt::reallocWithoutInitializing(space, fwd_ranks, n_imports);
    distributor.doPostsAndWaits(space, export_ranks, fwd_ranks);
  }

  {
    Kokkos::View<int *, MemorySpace> export_ids(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           prefix + "::export_ids"),
        n_exports);
    Kokkos::parallel_for(
        prefix + "::forward_queries_fill_ids",
        Kokkos::RangePolicy(space, 0, n_queries), KOKKOS_LAMBDA(int q) {
          for (int i = offset(q); i < offset(q + 1); ++i)
            export_ids(i) = q;
        });

    KokkosExt::reallocWithoutInitializing(space, fwd_ids, n_imports);
    distributor.doPostsAndWaits(space, export_ids, fwd_ids);
  }
}

template <typename ExecutionSpace, typename Predicates, typename Indices,
          typename Offset, typename FwdQueries>
void forwardQueries(MPI_Comm comm, ExecutionSpace const &space,
                    Predicates const &queries, Indices const &indices,
                    Offset const &offset, FwdQueries &fwd_queries)
{
  std::string prefix =
      "ArborX::DistributedTree::query::forwardQueries(partial)";

  Kokkos::Profiling::ScopedRegion guard(prefix);

  using MemorySpace = typename Predicates::memory_space;
  using Query = typename Predicates::value_type;

  Distributor<MemorySpace> distributor(comm);

  int const n_exports = KokkosExt::lastElement(space, offset);
  int const n_imports = distributor.createFromSends(space, indices);

  Kokkos::View<Query *, MemorySpace> export_queries(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         prefix + "::export_queries"),
      n_exports);
  Kokkos::parallel_for(
      prefix + "::forward_queries_fill_buffer",
      Kokkos::RangePolicy(space, 0, queries.size()), KOKKOS_LAMBDA(int q) {
        for (int i = offset(q); i < offset(q + 1); ++i)
          export_queries(i) = queries(q);
      });

  KokkosExt::reallocWithoutInitializing(space, fwd_queries, n_imports);
  distributor.doPostsAndWaits(space, export_queries, fwd_queries);
}

template <typename ExecutionSpace, typename OutputView, typename Offset,
          typename Ranks, typename Ids>
void communicateResultsBack(MPI_Comm comm, ExecutionSpace const &space,
                            OutputView &out, Offset const &offset, Ranks &ranks,
                            Ids &ids)
{
  Kokkos::Profiling::ScopedRegion guard(
      "ArborX::DistributedTree::communicateResultsBack");

  using MemorySpace = typename OutputView::memory_space;

  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  int const n_fwd_queries = offset.extent_int(0) - 1;
  int const n_exports = KokkosExt::lastElement(space, offset);

  // We are assuming here that if the same rank is related to multiple batches
  // these batches appear consecutively. Hence, no reordering is necessary.
  Distributor<MemorySpace> distributor(comm);
  // FIXME Distributor::createFromSends takes two views of the same type by
  // a const reference.  There were two easy ways out, either take the views by
  // value or cast at the call site.  I went with the latter.  Proper fix
  // involves more code cleanup in ArborX_Distributor.hpp than I am
  // willing to do just now.
  int const n_imports =
      distributor.createFromSends(space, ranks, static_cast<Ranks>(offset));

  {
    Kokkos::View<int *, MemorySpace> export_ranks(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing, ranks.label()),
        n_exports);
    Kokkos::deep_copy(space, export_ranks, comm_rank);

    Kokkos::View<int *, MemorySpace> import_ranks(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing, ranks.label()),
        n_imports);

    distributor.doPostsAndWaits(space, export_ranks, import_ranks);
    ranks = import_ranks;
  }

  {
    Kokkos::View<int *, MemorySpace> export_ids(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing, ids.label()),
        n_exports);
    Kokkos::parallel_for(
        "ArborX::DistributedTree::query::fill_buffer",
        Kokkos::RangePolicy(space, 0, n_fwd_queries), KOKKOS_LAMBDA(int q) {
          for (int i = offset(q); i < offset(q + 1); ++i)
          {
            export_ids(i) = ids(q);
          }
        });

    Kokkos::View<int *, MemorySpace> import_ids(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing, ids.label()),
        n_imports);

    distributor.doPostsAndWaits(space, export_ids, import_ids);
    ids = import_ids;
  }

  {
    OutputView export_out = out;

    OutputView import_out(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing, out.label()),
        n_imports);

    distributor.doPostsAndWaits(space, export_out, import_out);
    out = import_out;
  }
}

template <typename ExecutionSpace, typename BottomTree, typename Predicates,
          typename Callback, typename RanksTo, typename Offset, typename Values>
void forwardQueriesAndCommunicateResults(
    MPI_Comm comm, ExecutionSpace const &space, BottomTree const &bottom_tree,
    Predicates const &predicates, Callback const &callback,
    RanksTo const &ranks_to, Offset &offset, Values &values)
{
  std::string prefix =
      "ArborX::DistributedTree::query::forwardQueriesAndCommunicateResults";
  Kokkos::Profiling::ScopedRegion guard(prefix);

  using Query = typename Predicates::value_type;
  using MemorySpace = typename BottomTree::memory_space;

  // Forward predicates
  Kokkos::View<int *, MemorySpace> ids(prefix + "::query_ids", 0);
  Kokkos::View<Query *, MemorySpace> fwd_predicates(prefix + "::fwd_predicates",
                                                    0);
  Kokkos::View<int *, MemorySpace> ranks(prefix + "::ranks", 0);
  forwardQueries(comm, space, predicates, ranks_to, offset, fwd_predicates, ids,
                 ranks);

  // Perform predicates that have been received
  bottom_tree.query(space, fwd_predicates, callback, values, offset);

  // Communicate results back
  communicateResultsBack(comm, space, values, offset, ranks, ids);

  Kokkos::Profiling::pushRegion(prefix + "::postprocess_results");

  // Merge results
  int const n_predicates = predicates.size();
  countResults(space, n_predicates, ids, offset);
  KokkosExt::sortByKey(space, ids, values);

  Kokkos::Profiling::popRegion();
}

template <typename ExecutionSpace, typename MemorySpace, typename Predicates,
          typename Values, typename Offset>
void filterResults(ExecutionSpace const &space, Predicates const &queries,
                   Kokkos::View<float *, MemorySpace> const &distances,
                   Values &values, Offset &offset)
{
  Kokkos::Profiling::ScopedRegion guard(
      "ArborX::DistributedTree::filterResults");

  using Value = typename Values::value_type;

  int const n_queries = queries.size();
  // truncated views are prefixed with an underscore
  Kokkos::View<int *, MemorySpace> new_offset(
      Kokkos::view_alloc(space, offset.label()), n_queries + 1);

  Kokkos::parallel_for(
      "ArborX::DistributedTree::query::discard_results",
      Kokkos::RangePolicy(space, 0, n_queries), KOKKOS_LAMBDA(int q) {
        using Kokkos::min;
        new_offset(q) = min(offset(q + 1) - offset(q), getK(queries(q)));
      });

  KokkosExt::exclusive_scan(space, new_offset, new_offset, 0);

  int const n_truncated_results = KokkosExt::lastElement(space, new_offset);
  Kokkos::View<Value *, MemorySpace> new_values(
      Kokkos::view_alloc(space, values.label()), n_truncated_results);

  using PairValueDistance = Kokkos::pair<Value, float>;
  struct CompareDistance
  {
    KOKKOS_INLINE_FUNCTION bool operator()(PairValueDistance const &lhs,
                                           PairValueDistance const &rhs)
    {
      // reverse order (larger distance means lower priority)
      return lhs.second > rhs.second;
    }
  };

  int const n_results = KokkosExt::lastElement(space, offset);
  Kokkos::View<PairValueDistance *, MemorySpace> buffer(
      Kokkos::view_alloc(
          space, Kokkos::WithoutInitializing,
          "ArborX::DistributedTree::query::filterResults::buffer"),
      n_results);
  using PriorityQueue =
      Details::PriorityQueue<PairValueDistance, CompareDistance,
                             UnmanagedStaticVector<PairValueDistance>>;

  Kokkos::parallel_for(
      "ArborX::DistributedTree::query::truncate_results",
      Kokkos::RangePolicy(space, 0, n_queries), KOKKOS_LAMBDA(int q) {
        if (offset(q) == offset(q + 1))
          return;

        auto local_buffer = Kokkos::subview(
            buffer, Kokkos::make_pair(offset(q), offset(q + 1)));

        PriorityQueue queue(UnmanagedStaticVector<PairValueDistance>(
            local_buffer.data(), local_buffer.size()));

        for (int i = offset(q); i < offset(q + 1); ++i)
          queue.emplace(values(i), distances(i));

        int count = 0;
        while (!queue.empty() && count < getK(queries(q)))
        {
          new_values(new_offset(q) + count) = queue.top().first;
          queue.pop();
          ++count;
        }
      });
  values = new_values;
  offset = new_offset;
}

} // namespace ArborX::Details::DistributedTree

#endif
