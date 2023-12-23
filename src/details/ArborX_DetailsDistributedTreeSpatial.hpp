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
#ifndef ARBORX_DETAILS_DISTRIBUTED_TREE_SPATIAL_HPP
#define ARBORX_DETAILS_DISTRIBUTED_TREE_SPATIAL_HPP

#include <ArborX_DetailsDistributedTreeImpl.hpp>
#include <ArborX_DetailsDistributedTreeUtils.hpp>
#include <ArborX_Predicates.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <mpi.h>

namespace ArborX::Details
{
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

template <typename Tree, typename ExecutionSpace, typename Predicates,
          typename OutputView, typename OffsetView, typename Callback>
std::enable_if_t<Kokkos::is_view<OutputView>{} && Kokkos::is_view<OffsetView>{}>
DistributedTreeImpl::queryDispatch(SpatialPredicateTag, Tree const &tree,
                                   ExecutionSpace const &space,
                                   Predicates const &queries,
                                   Callback const &callback, OutputView &out,
                                   OffsetView &offset)
{
  Kokkos::Profiling::ScopedRegion guard(
      "ArborX::DistributedTree::query::spatial");

  using namespace DistributedTree;
  using MemorySpace = typename Tree::memory_space;

  auto const &top_tree = tree._top_tree;
  auto const &bottom_tree = tree._bottom_tree;
  auto comm = tree.getComm();

  Kokkos::View<int *, MemorySpace> indices(
      "ArborX::DistributedTree::query::spatial::indices", 0);
  Kokkos::View<int *, MemorySpace> ranks(
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
    using Query = typename Predicates::value_type;
    Kokkos::View<int *, MemorySpace> ids(
        "ArborX::DistributedTree::query::spatial::query_ids", 0);
    Kokkos::View<Query *, MemorySpace> fwd_queries(
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
    int const n_queries = queries.size();
    countResults(space, n_queries, ids, offset);
    sortResults(space, ids, out);

    Kokkos::Profiling::popRegion();
  }
}

template <typename Tree, typename ExecutionSpace, typename Predicates,
          typename IndicesAndRanks, typename Offset>
std::enable_if_t<Kokkos::is_view<IndicesAndRanks>{} &&
                 Kokkos::is_view<Offset>{}>
DistributedTreeImpl::queryDispatch(SpatialPredicateTag, Tree const &tree,
                                   ExecutionSpace const &space,
                                   Predicates const &queries,
                                   IndicesAndRanks &values, Offset &offset)
{
  int comm_rank;
  MPI_Comm_rank(tree.getComm(), &comm_rank);
  queryDispatch(SpatialPredicateTag{}, tree, space, queries,
                DefaultCallbackWithRank{comm_rank}, values, offset);
}

} // namespace ArborX::Details

#endif
