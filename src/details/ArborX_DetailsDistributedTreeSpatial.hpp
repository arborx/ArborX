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

#include <ArborX_Callbacks.hpp>
#include <ArborX_DetailsDistributedTreeImpl.hpp>
#include <ArborX_DetailsDistributedTreeUtils.hpp>
#include <ArborX_DetailsLegacy.hpp>
#include <ArborX_Predicates.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <mpi.h>

namespace ArborX::Details
{

template <typename Tree, typename ExecutionSpace, typename Predicates,
          typename Values, typename Offset, typename Callback>
std::enable_if_t<Kokkos::is_view<Values>{} && Kokkos::is_view<Offset>{}>
DistributedTreeImpl::queryDispatch(SpatialPredicateTag, Tree const &tree,
                                   ExecutionSpace const &space,
                                   Predicates const &predicates,
                                   Callback const &callback, Values &values,
                                   Offset &offset)
{
  Kokkos::Profiling::ScopedRegion guard(
      "ArborX::DistributedTree::query::spatial");

  using namespace DistributedTree;
  using MemorySpace = typename Tree::memory_space;

  auto const &top_tree = tree._top_tree;
  auto const &bottom_tree = tree._bottom_tree;
  auto comm = tree.getComm();

  Kokkos::View<int *, MemorySpace> intersected_ranks(
      "ArborX::DistributedTree::query::spatial::intersected_ranks", 0);
  top_tree.query(space, predicates, LegacyDefaultCallback{}, intersected_ranks,
                 offset);

  Kokkos::View<int *, MemorySpace> ranks(
      "ArborX::DistributedTree::query::spatial::ranks", 0);
  {
    // NOTE_COMM_SPATIAL: The communication pattern here for the spatial search
    // is identical to that of the nearest search (see NOTE_COMM_NEAREST). The
    // code differences are:
    // - usage of callbacks
    // - no explicit distances
    // - no results filtering

    // Forward predicates
    using Query = typename Predicates::value_type;
    Kokkos::View<int *, MemorySpace> ids(
        "ArborX::DistributedTree::query::spatial::query_ids", 0);
    Kokkos::View<Query *, MemorySpace> fwd_predicates(
        "ArborX::DistributedTree::query::spatial::fwd_predicates", 0);
    forwardQueries(comm, space, predicates, intersected_ranks, offset,
                   fwd_predicates, ids, ranks);

    // Perform predicates that have been received
    bottom_tree.query(space, fwd_predicates, callback, values, offset);

    // Communicate results back
    communicateResultsBack(comm, space, values, offset, ranks, ids);

    Kokkos::Profiling::pushRegion(
        "ArborX::DistributedTree::spatial::postprocess_results");

    // Merge results
    int const n_predicates = predicates.size();
    countResults(space, n_predicates, ids, offset);
    sortResults(space, ids, values);

    Kokkos::Profiling::popRegion();
  }
}

template <typename Tree, typename ExecutionSpace, typename Predicates,
          typename Values, typename Offset>
std::enable_if_t<Kokkos::is_view<Values>{} && Kokkos::is_view<Offset>{}>
DistributedTreeImpl::queryDispatch(SpatialPredicateTag, Tree const &tree,
                                   ExecutionSpace const &space,
                                   Predicates const &predicates, Values &values,
                                   Offset &offset)
{
  queryDispatch(SpatialPredicateTag{}, tree, space, predicates,
                DefaultCallback{}, values, offset);
}

} // namespace ArborX::Details

#endif
