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

#include <ArborX_DetailsContainers.hpp>
#include <ArborX_DetailsDistributor.hpp>
#include <ArborX_DetailsKokkosExtMinMaxOperations.hpp>
#include <ArborX_DetailsKokkosExtSort.hpp>
#include <ArborX_DetailsKokkosExtStdAlgorithms.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_DetailsPriorityQueue.hpp>
#include <ArborX_DetailsUtils.hpp> // create_layout*

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <mpi.h>

namespace ArborX::Details::DistributedTree
{

template <typename ExecutionSpace, typename Distributor, typename View>
std::enable_if_t<Kokkos::is_view<View>::value>
sendAcrossNetwork(ExecutionSpace const &space, Distributor const &distributor,
                  View exports, typename View::non_const_type imports)
{
  Kokkos::Profiling::ScopedRegion guard(
      "ArborX::DistributedTree::sendAcrossNetwork (" + exports.label() + ")");

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

#ifndef ARBORX_ENABLE_GPU_AWARE_MPI
  using MirrorSpace = typename View::host_mirror_space;
  typename MirrorSpace::execution_space const execution_space;
#else
  using MirrorSpace = typename View::device_type::memory_space;
  auto const &execution_space = space;
#endif

  auto imports_layout_right = create_layout_right_mirror_view_no_init(
      execution_space, MirrorSpace{}, imports);

#ifndef ARBORX_ENABLE_GPU_AWARE_MPI
  execution_space.fence();
#endif

  Kokkos::View<NonConstValueType *, MirrorSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      import_buffer(imports_layout_right.data(), imports_layout_right.size());

  distributor.doPostsAndWaits(space, exports, num_packets, import_buffer);

  constexpr bool can_skip_copy =
      (View::rank == 1 &&
       (std::is_same_v<typename View::array_layout, Kokkos::LayoutLeft> ||
        std::is_same_v<typename View::array_layout, Kokkos::LayoutRight>));
  if constexpr (can_skip_copy)
  {
    // For 1D non-strided views, we can directly copy to the original location,
    // as layout is the same
    Kokkos::deep_copy(space, imports, imports_layout_right);
  }
  else
  {
    // For multi-dimensional views, we need to first copy into a separate
    // storage because of a different layout
    auto tmp_view = Kokkos::create_mirror_view_and_copy(
        Kokkos::view_alloc(space, typename ExecutionSpace::memory_space{}),
        imports_layout_right);
    Kokkos::deep_copy(space, imports, tmp_view);
  }
}

template <typename ExecutionSpace, typename View, typename... OtherViews>
void sortResultsByKey(ExecutionSpace const &space, View keys,
                      OtherViews... other_views)
{
  auto const n = keys.extent(0);
  // If they were no queries, min_val and max_val values won't change after
  // the parallel reduce (they are initialized to +infty and -infty
  // respectively) and the sort will hang.
  if (n == 0)
    return;

  using ViewType = std::tuple_element_t<0, std::tuple<OtherViews...>>;
  if constexpr (sizeof...(OtherViews) == 1 && ViewType::rank == 1 &&
                std::is_arithmetic_v<typename ViewType::value_type>)
  {
    // If there's only one 1D view to process, we can avoid computing the
    // permutation. We also avoid 1D views with non-arithmetic types as we
    // can't guarantee they provide comparison operator.
    KokkosExt::sortByKey(space, keys, other_views...);
  }
  else
  {
    auto const permutation = ArborX::Details::sortObjects(space, keys);

    // Call applyPermutation for every entry in the parameter pack.
    // We need to use the comma operator here since the function returns void.
    // The variable we assign to is actually not needed. We just need something
    // to store the initializer list (that contains only zeros).
    auto dummy = {
        (ArborX::Details::applyPermutation(space, permutation, other_views),
         0)...};
    std::ignore = dummy;
  }
}

template <typename ExecutionSpace, typename QueryIdsView, typename OffsetView>
void countResults(ExecutionSpace const &space, int n_queries,
                  QueryIdsView const &query_ids, OffsetView &offset)
{
  int const nnz = query_ids.extent(0);

  Kokkos::realloc(Kokkos::view_alloc(space), offset, n_queries + 1);

  Kokkos::parallel_for(
      "ArborX::DistributedTree::query::count_results_per_query",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, nnz), KOKKOS_LAMBDA(int i) {
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
  Kokkos::Profiling::ScopedRegion guard(
      "ArborX::DistributedTree::forwardQueries");

  using MemorySpace = typename Predicates::memory_space;
  using Query = typename Predicates::value_type;

  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  Distributor<MemorySpace> distributor(comm);

  int const n_queries = queries.size();
  int const n_exports = KokkosExt::lastElement(space, offset);
  int const n_imports = distributor.createFromSends(space, indices);

  static_assert(std::is_same_v<Query, typename Predicates::value_type>);

  {
    Kokkos::View<int *, MemorySpace> export_ranks(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::DistributedTree::query::forwardQueries::export_ranks"),
        n_exports);
    Kokkos::deep_copy(space, export_ranks, comm_rank);

    Kokkos::View<int *, MemorySpace> import_ranks(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::DistributedTree::query::forwardQueries::import_ranks"),
        n_imports);

    sendAcrossNetwork(space, distributor, export_ranks, import_ranks);
    fwd_ranks = import_ranks;
  }

  {
    Kokkos::View<Query *, MemorySpace> exports(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::DistributedTree::query::forwardQueries::exports"),
        n_exports);
    Kokkos::parallel_for(
        "ArborX::DistributedTree::query::forward_queries_fill_buffer",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
        KOKKOS_LAMBDA(int q) {
          for (int i = offset(q); i < offset(q + 1); ++i)
          {
            exports(i) = queries(q);
          }
        });
    Kokkos::View<Query *, MemorySpace> imports(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::DistributedTree::query::forwardQueries::imports"),
        n_imports);

    sendAcrossNetwork(space, distributor, exports, imports);
    fwd_queries = imports;
  }

  {
    Kokkos::View<int *, MemorySpace> export_ids(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
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
    Kokkos::View<int *, MemorySpace> import_ids(
        Kokkos::view_alloc(
            space, Kokkos::WithoutInitializing,
            "ArborX::DistributedTree::query::forwardQueries::import_ids"),
        n_imports);

    sendAcrossNetwork(space, distributor, export_ids, import_ids);
    fwd_ids = import_ids;
  }
}

template <typename ExecutionSpace, typename OutputView, typename Offset,
          typename Ranks, typename Ids,
          typename Distances =
              Kokkos::View<float *, typename OutputView::memory_space>>
void communicateResultsBack(MPI_Comm comm, ExecutionSpace const &space,
                            OutputView &out, Offset const &offset, Ranks &ranks,
                            Ids &ids, Distances *distances_ptr = nullptr)
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
  // involves more code cleanup in ArborX_DetailsDistributor.hpp than I am
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

    sendAcrossNetwork(space, distributor, export_ranks, import_ranks);
    ranks = import_ranks;
  }

  {
    Kokkos::View<int *, MemorySpace> export_ids(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing, ids.label()),
        n_exports);
    Kokkos::parallel_for(
        "ArborX::DistributedTree::query::fill_buffer",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_fwd_queries),
        KOKKOS_LAMBDA(int q) {
          for (int i = offset(q); i < offset(q + 1); ++i)
          {
            export_ids(i) = ids(q);
          }
        });

    Kokkos::View<int *, MemorySpace> import_ids(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing, ids.label()),
        n_imports);

    sendAcrossNetwork(space, distributor, export_ids, import_ids);
    ids = import_ids;
  }

  {
    OutputView export_out = out;

    OutputView import_out(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing, out.label()),
        n_imports);

    sendAcrossNetwork(space, distributor, export_out, import_out);
    out = import_out;
  }

  if (distances_ptr)
  {
    auto &distances = *distances_ptr;
    Kokkos::View<float *, MemorySpace> export_distances = distances;
    Kokkos::View<float *, MemorySpace> import_distances(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           distances.label()),
        n_imports);
    sendAcrossNetwork(space, distributor, export_distances, import_distances);
    distances = import_distances;
  }
}

template <
    typename ExecutionSpace, typename BottomTree, typename Predicates,
    typename Callback, typename RanksTo, typename Offset, typename Values,
    typename Ranks = Kokkos::View<int *, typename BottomTree::memory_space>>
void forwardQueriesAndCommunicateResults(
    MPI_Comm comm, ExecutionSpace const &space, BottomTree const &bottom_tree,
    Predicates const &predicates, Callback const &callback,
    RanksTo const &ranks_to, Offset &offset, Values &values,
    Ranks *ranks_ptr = nullptr)
{
  std::string prefix =
      "ArborX::DistributedTree::query::forwardQueriesAndCommunicateResults";
  Kokkos::Profiling::ScopedRegion guard(prefix);

  using Query = typename Predicates::value_type;
  using MemorySpace = typename BottomTree::memory_space;

  Ranks ranks_aux(prefix + "::ranks", 0);
  auto &ranks = (ranks_ptr != nullptr ? *ranks_ptr : ranks_aux);

  // Forward predicates
  Kokkos::View<int *, MemorySpace> ids(prefix + "::query_ids", 0);
  Kokkos::View<Query *, MemorySpace> fwd_predicates(prefix + "::fwd_predicates",
                                                    0);
  forwardQueries(comm, space, predicates, ranks_to, offset, fwd_predicates, ids,
                 ranks);

  // Perform predicates that have been received
  bottom_tree.query(space, fwd_predicates, callback, values, offset);

  // Communicate results back
  communicateResultsBack(comm, space, values, offset, ranks, ids);

  Kokkos::Profiling::pushRegion(prefix + "postprocess_results");

  // Merge results
  int const n_predicates = predicates.size();
  countResults(space, n_predicates, ids, offset);
  if (ranks_ptr != nullptr)
    sortResultsByKey(space, ids, values, ranks);
  else
    sortResultsByKey(space, ids, values);

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
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
      KOKKOS_LAMBDA(int q) {
        using KokkosExt::min;
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
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
      KOKKOS_LAMBDA(int q) {
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
