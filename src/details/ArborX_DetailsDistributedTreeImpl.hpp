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

namespace ArborX::Details
{
struct DistributedTreeImpl
{
  // spatial queries
  template <typename DistributedTree, typename ExecutionSpace,
            typename Predicates, typename IndicesAndRanks, typename Offset>
  static std::enable_if_t<Kokkos::is_view<IndicesAndRanks>{} &&
                          Kokkos::is_view<Offset>{}>
  queryDispatch(SpatialPredicateTag, DistributedTree const &tree,
                ExecutionSpace const &space, Predicates const &queries,
                IndicesAndRanks &values, Offset &offset);

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
            typename Distances =
                Kokkos::View<float *, typename DistributedTree::memory_space>>
  static std::enable_if_t<
      Kokkos::is_view<Indices>{} && Kokkos::is_view<Offset>{} &&
      Kokkos::is_view<Ranks>{} && Kokkos::is_view<Distances>{}>
  queryDispatchImpl(NearestPredicateTag, DistributedTree const &tree,
                    ExecutionSpace const &space, Predicates const &queries,
                    Indices &indices, Offset &offset, Ranks &ranks,
                    Distances *distances_ptr = nullptr);

  template <typename DistributedTree, typename ExecutionSpace,
            typename Predicates, typename IndicesAndRanks, typename Offset>
  static std::enable_if_t<Kokkos::is_view<IndicesAndRanks>{} &&
                          Kokkos::is_view<Offset>{}>
  queryDispatch(NearestPredicateTag tag, DistributedTree const &tree,
                ExecutionSpace const &space, Predicates const &queries,
                IndicesAndRanks &values, Offset &offset);

  // nearest neighbors helpers
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
};

} // namespace ArborX::Details

#endif
