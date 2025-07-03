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
#ifndef ARBORX_DISTRIBUTED_TREE_SPATIAL_HPP
#define ARBORX_DISTRIBUTED_TREE_SPATIAL_HPP

#include <detail/ArborX_Callbacks.hpp>
#include <detail/ArborX_DistributedTreeImpl.hpp>
#include <detail/ArborX_DistributedTreeUtils.hpp>
#include <detail/ArborX_Predicates.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <mpi.h>

namespace ArborX::Details
{

template <typename Tree, typename ExecutionSpace,
          Concepts::Predicates Predicates, typename Values, typename Offset,
          typename Callback>
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

  auto const &top_tree = tree._top_tree;

  Kokkos::View<int *, MemorySpace> intersected_ranks(
      "ArborX::DistributedTree::query::spatial::intersected_ranks", 0);
  top_tree.query(space, predicates, DistributedTree::IndexOnlyCallback{},
                 intersected_ranks, offset);

  DistributedTree::forwardQueriesAndCommunicateResults(
      tree.getComm(), space, tree._bottom_tree, predicates, callback,
      intersected_ranks, offset, values);
}

template <typename Tree, typename ExecutionSpace,
          Concepts::Predicates Predicates, typename Callback>
void DistributedTreeImpl::queryDispatch(SpatialPredicateTag, Tree const &tree,
                                        ExecutionSpace const &space,
                                        Predicates const &predicates,
                                        Callback const &callback)
{
  std::string prefix = "ArborX::DistributedTree::query::spatial(pure)";

  Kokkos::Profiling::ScopedRegion guard(prefix);

  if (tree.empty())
    return;

  using MemorySpace = typename Tree::memory_space;
  using namespace DistributedTree;

  auto const &top_tree = tree._top_tree;
  auto const &bottom_tree = tree._bottom_tree;
  auto comm = tree.getComm();

  Kokkos::View<int *, MemorySpace> intersected_ranks(
      prefix + "::intersected_ranks", 0);
  Kokkos::View<int *, MemorySpace> offset(prefix + "::offset", 0);
  top_tree.query(space, predicates, DistributedTree::IndexOnlyCallback{},
                 intersected_ranks, offset);

  using Query = typename Predicates::value_type;
  Kokkos::View<Query *, MemorySpace> fwd_predicates(prefix + "::fwd_predicates",
                                                    0);
  forwardQueries(comm, space, predicates, intersected_ranks, offset,
                 fwd_predicates);

  bottom_tree.query(space, fwd_predicates, callback);
}

template <typename Tree, typename ExecutionSpace,
          Concepts::Predicates Predicates, typename Values, typename Offset>
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
