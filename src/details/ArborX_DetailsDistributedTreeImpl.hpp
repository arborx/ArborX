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
#ifndef ARBORX_DETAILS_DISTRIBUTED_TREE_IMPL_HPP
#define ARBORX_DETAILS_DISTRIBUTED_TREE_IMPL_HPP

#include <ArborX_AccessTraits.hpp>

#include <Kokkos_Core.hpp>

// Don't really need it, but our self containment tests rely on its presence
#include <mpi.h>

namespace ArborX::Details
{
struct DistributedTreeImpl
{
  // spatial queries
  template <typename DistributedTree, typename ExecutionSpace,
            typename Predicates, typename IndicesAndRanks, typename Offset>
  static std::enable_if_t<Kokkos::is_view_v<IndicesAndRanks> &&
                          Kokkos::is_view_v<Offset>>
  queryDispatch(SpatialPredicateTag, DistributedTree const &tree,
                ExecutionSpace const &space, Predicates const &queries,
                IndicesAndRanks &values, Offset &offset);

  template <typename DistributedTree, typename ExecutionSpace,
            typename Predicates, typename OutputView, typename OffsetView,
            typename Callback>
  static std::enable_if_t<Kokkos::is_view_v<OutputView> &&
                          Kokkos::is_view_v<OffsetView>>
  queryDispatch(SpatialPredicateTag, DistributedTree const &tree,
                ExecutionSpace const &space, Predicates const &queries,
                Callback const &callback, OutputView &out, OffsetView &offset);

  // nearest neighbors queries
  template <typename DistributedTree, typename ExecutionSpace,
            typename Predicates, typename Indices, typename Offset,
            typename Ranks>
  static std::enable_if_t<Kokkos::is_view_v<Indices> &&
                          Kokkos::is_view_v<Offset> && Kokkos::is_view_v<Ranks>>
  queryDispatchImpl(NearestPredicateTag, DistributedTree const &tree,
                    ExecutionSpace const &space, Predicates const &queries,
                    Indices &indices, Offset &offset, Ranks &ranks);

  template <typename DistributedTree, typename ExecutionSpace,
            typename Predicates, typename IndicesAndRanks, typename Offset>
  static std::enable_if_t<Kokkos::is_view_v<IndicesAndRanks> &&
                          Kokkos::is_view_v<Offset>>
  queryDispatch(NearestPredicateTag tag, DistributedTree const &tree,
                ExecutionSpace const &space, Predicates const &queries,
                IndicesAndRanks &values, Offset &offset);

  // nearest neighbors helpers
  template <typename ExecutionSpace, typename Tree, typename Predicates,
            typename Distances>
  static void phaseI(ExecutionSpace const &space, Tree const &tree,
                     Predicates const &predicates,
                     Distances &farthest_distances);

  template <typename ExecutionSpace, typename Tree, typename Predicates,
            typename Distances, typename Offset, typename Values,
            typename Ranks>
  static void phaseII(ExecutionSpace const &space, Tree const &tree,
                      Predicates const &queries, Distances &distances,
                      Offset &offset, Values &values, Ranks &ranks);
};

} // namespace ArborX::Details

#endif
