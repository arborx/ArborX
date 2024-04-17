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
std::enable_if_t<Kokkos::is_view_v<Values> && Kokkos::is_view_v<Offset>>
DistributedTreeImpl::queryDispatch(SpatialPredicateTag, Tree const &tree,
                                   ExecutionSpace const &space,
                                   Predicates const &predicates,
                                   Callback const &callback, Values &values,
                                   Offset &offset)
{
  Kokkos::Profiling::ScopedRegion guard(
      "ArborX::DistributedTree::query::spatial");

  if (tree.empty())
  {
    KokkosExt::reallocWithoutInitializing(space, values, 0);
    KokkosExt::reallocWithoutInitializing(space, offset, predicates.size() + 1);
    Kokkos::deep_copy(space, offset, 0);
    return;
  }

  using MemorySpace = typename Tree::memory_space;

  Kokkos::View<int *, MemorySpace> intersected_ranks(
      "ArborX::DistributedTree::query::spatial::intersected_ranks", 0);
  tree._top_tree.query(space, predicates, LegacyDefaultCallback{},
                       intersected_ranks, offset);

  Kokkos::View<int *, MemorySpace> ranks(
      "ArborX::DistributedTree::query::spatial::ranks", 0);
  DistributedTree::forwardQueriesAndCommunicateResults(
      tree.getComm(), space, tree._bottom_tree, predicates, callback,
      intersected_ranks, offset, values, ranks);
}

template <typename Tree, typename ExecutionSpace, typename Predicates,
          typename Values, typename Offset>
std::enable_if_t<Kokkos::is_view_v<Values> && Kokkos::is_view_v<Offset>>
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
