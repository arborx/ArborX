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
#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>

#include <Kokkos_Core.hpp>

#include <cassert>

#include "DetailsMovingLeastSquaresComputation.hpp"
#include "mpi_comms.hpp"

template <typename Points>
struct TargetPoints
{
  Points target_points;
  std::size_t num_neighbors;
};

template <typename Points>
struct ArborX::AccessTraits<TargetPoints<Points>, ArborX::PredicatesTag>
{
  static KOKKOS_FUNCTION std::size_t size(TargetPoints<Points> const &tp)
  {
    return ArborX::AccessTraits<Points, ArborX::PrimitivesTag>::size(
        tp.target_points);
  }

  static KOKKOS_FUNCTION auto get(TargetPoints<Points> const &tp, std::size_t i)
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

template <typename ValueType, typename PolynomialBasis, typename RBF,
          typename MemorySpace>
class MLS
{
public:
  template <typename ExecutionSpace, typename Points>
  MLS(MPI_Comm comm, ExecutionSpace const &space, Points const &source_points,
      Points const &target_points,
      std::size_t num_neighbors = PolynomialBasis::size)
      : _num_neighbors(num_neighbors)
      , _src_size(ArborX::AccessTraits<Points, ArborX::PrimitivesTag>::size(
            source_points))
      , _tgt_size(ArborX::AccessTraits<Points, ArborX::PrimitivesTag>::size(
            target_points))
  {
    static_assert(
        KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value);
    static_assert(
        KokkosExt::is_accessible_from<
            typename ArborX::AccessTraits<Points,
                                          ArborX::PrimitivesTag>::memory_space,
            ExecutionSpace>::value);
    ArborX::Details::check_valid_access_traits(ArborX::PrimitivesTag{},
                                               source_points);

    // A minimum nuber of source points are needed
    assert(_src_size >= _num_neighbors);

    // Organize source points as tree
    ArborX::DistributedTree<MemorySpace> source_tree(comm, space,
                                                     source_points);

    // Perform the query
    Kokkos::View<ArborX::PairIndexRank *, MemorySpace> index_ranks(
        "Example::MLS::index_ranks", 0);
    Kokkos::View<int *, MemorySpace> offsets("Example::MLS::offsets", 0);
    source_tree.query(space,
                      TargetPoints<Points>{target_points, _num_neighbors},
                      index_ranks, offsets);

    // Split indices/ranks
    Kokkos::View<int *, MemorySpace> local_indices(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::MLS::local_indices"),
        _tgt_size * _num_neighbors);
    Kokkos::View<int *, MemorySpace> local_ranks(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::MLS::local_ranks"),
        _tgt_size * _num_neighbors);
    Kokkos::parallel_for(
        "Example::MLS::index_ranks_split",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0,
                                            _tgt_size * _num_neighbors),
        KOKKOS_LAMBDA(int const i) {
          local_indices(i) = index_ranks(i).index;
          local_ranks(i) = index_ranks(i).rank;
        });

    // Set up comms and local source points
    _comms = MPIComms<MemorySpace>(comm, space, local_indices, local_ranks);
    auto local_source_points = _comms.distributeArborX(space, source_points);

    // Compute the internal MLS
    _mlsc = Details::MovingLeastSquaresComputation<ValueType, MemorySpace>(
        space, local_source_points, target_points, PolynomialBasis{}, RBF{});
  }

  template <typename ExecutionSpace>
  Kokkos::View<ValueType *, MemorySpace>
  apply(ExecutionSpace const &space,
        Kokkos::View<ValueType *, MemorySpace> const &source_values)
  {
    static_assert(
        KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value);
    assert(source_values.extent(0) == _src_size);
    return _mlsc.apply(space, _comms.distributeView(space, source_values));
  }

private:
  Details::MovingLeastSquaresComputation<ValueType, MemorySpace> _mlsc;
  MPIComms<MemorySpace> _comms;
  std::size_t _num_neighbors;
  std::size_t _src_size;
  std::size_t _tgt_size;
};