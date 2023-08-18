/****************************************************************************
 * Copyright (c) 2023 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#pragma once

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

#include "DetailsDistributedTreePostQueryComms.hpp"
#include "DetailsMovingLeastSquaresComputation.hpp"

namespace Details
{

// This is done to avoid clashing with another predicate access trait
template <typename Points>
struct TargetPointsPredicateWrapper
{
  Points target_points;
  std::size_t num_neighbors;
};

} // namespace Details

template <typename Points>
struct ArborX::AccessTraits<Details::TargetPointsPredicateWrapper<Points>,
                            ArborX::PredicatesTag>
{
  static KOKKOS_FUNCTION std::size_t
  size(::Details::TargetPointsPredicateWrapper<Points> const &tp)
  {
    return ArborX::AccessTraits<Points, ArborX::PrimitivesTag>::size(
        tp.target_points);
  }

  static KOKKOS_FUNCTION auto
  get(::Details::TargetPointsPredicateWrapper<Points> const &tp, std::size_t i)
  {
    return ArborX::nearest(
        ArborX::AccessTraits<Points, ArborX::PrimitivesTag>::get(
            tp.target_points, i),
        tp.num_neighbors);
  }

  using memory_space =
      typename ArborX::AccessTraits<Points,
                                    ArborX::PrimitivesTag>::memory_space;
};

// Public interface to compute the moving least squares approximation between a
// souce and target point cloud
template <typename MemorySpace, typename FloatingCalculationType = float>
class MovingLeastSquares
{
public:
  template <typename ExecustionSpace, typename PolynomialBasis,
            typename RadialBasisFunction, typename SourcePoints,
            typename TargetPoints>
  MovingLeastSquares(MPI_Comm comm, ExecustionSpace const &space,
                     SourcePoints const &source_points,
                     TargetPoints const &target_points,
                     PolynomialBasis const &pb, RadialBasisFunction const &rbf,
                     std::size_t num_neighbors = PolynomialBasis::size)
  {
    // Organize the source points as a tree and create the predicates
    ArborX::DistributedTree<MemorySpace> source_tree(comm, space,
                                                     source_points);
    Details::TargetPointsPredicateWrapper<TargetPoints> predicates{
        target_points, num_neighbors};

    // Makes the NN query
    Kokkos::View<ArborX::PairIndexRank *, MemorySpace> indices_and_ranks(
        "Example::MLS::indices_and_ranks", 0);
    Kokkos::View<int *, MemorySpace> offsets("Example::MLS::offsets", 0);
    source_tree.query(space, predicates, indices_and_ranks, offsets);

    // Set up comms and collect the points for a local MLS
    _comms = Details::DistributedTreePostQueryComms<MemorySpace>(
        comm, space, indices_and_ranks);
    auto local_source_points = _comms.distribute(space, source_points);

    // Finally, compute the local MLS for the local target points
    _mlsc = Details::MovingLeastSquaresComputation<MemorySpace,
                                                   FloatingCalculationType>(
        space, local_source_points, target_points, pb, rbf);
  }

  template <typename ExecutionSpace, typename SourceValues>
  auto apply(ExecutionSpace const &space, SourceValues const &source_values)
  {
    // Distribute and compute the result
    return _mlsc.apply(space, _comms.distribute(space, source_values));
  }

private:
  Details::MovingLeastSquaresComputation<MemorySpace, FloatingCalculationType>
      _mlsc;
  Details::DistributedTreePostQueryComms<MemorySpace> _comms;
};