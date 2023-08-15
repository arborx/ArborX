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

#include <cassert>

#include "mls_computation.hpp"
#include "mpi_comms.hpp"

template <typename MemorySpace>
struct TargetPoints
{
  Kokkos::View<ArborX::Point *, MemorySpace> target_points;
  std::size_t num_neighbors;
};

template <typename ValueType, typename PolynomialBasis, typename RBF,
          typename MemorySpace>
class MLS
{
public:
  template <typename ExecutionSpace>
  MLS(ExecutionSpace const &space, MPI_Comm comm,
      Kokkos::View<ArborX::Point *, MemorySpace> const &source_points,
      Kokkos::View<ArborX::Point *, MemorySpace> const &target_points,
      std::size_t num_neighbors = PolynomialBasis::size)
      : _num_neighbors(num_neighbors)
      , _src_size(source_points.extent(0))
      , _tgt_size(target_points.extent(0))
  {
    // There must be enough source points
    assert(_src_size >= _num_neighbors);

    // Organize source points as tree
    ArborX::DistributedTree<MemorySpace> source_tree(comm, space,
                                                     source_points);

    // Perform the query
    Kokkos::View<Kokkos::pair<int, int> *, MemorySpace> index_ranks(
        "Example::MLS::index_ranks", 0);
    Kokkos::View<int *, MemorySpace> offsets("Example::MLS::offsets", 0);
    source_tree.query(space,
                      TargetPoints<MemorySpace>{target_points, _num_neighbors},
                      index_ranks, offsets);

    // Split indices/ranks
    Kokkos::View<int *, MemorySpace> local_indices(
        "Example::MLS::local_indices", _tgt_size * _num_neighbors);
    Kokkos::View<int *, MemorySpace> local_ranks("Example::MLS::local_ranks",
                                                 _tgt_size * _num_neighbors);
    Kokkos::parallel_for(
        "Example::MLS::index_ranks_split",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0,
                                            _tgt_size * _num_neighbors),
        KOKKOS_LAMBDA(int const i) {
          local_indices(i) = index_ranks(i).first;
          local_ranks(i) = index_ranks(i).second;
        });

    // Set up comms and local source points
    _comms = MPIComms<MemorySpace>(space, comm, local_indices, local_ranks);
    auto local_source_points = _comms.distribute(space, source_points);

    // Compute the internal MLS
    _mlsc =
        MLSComputation<ValueType, PolynomialBasis, RBF,
                       MemorySpace>(space, local_source_points, target_points);
  }

  template <typename ExecutionSpace>
  Kokkos::View<ValueType *, MemorySpace>
  apply(ExecutionSpace const &space,
        Kokkos::View<ValueType *, MemorySpace> const &source_values)
  {
    assert(source_values.extent(0) == _src_size);
    return _mlsc.apply(space, _comms.distribute(space, source_values));
  }

private:
  MLSComputation<ValueType, PolynomialBasis, RBF, MemorySpace>
      _mlsc;
  MPIComms<MemorySpace> _comms;
  std::size_t _num_neighbors;
  std::size_t _src_size;
  std::size_t _tgt_size;
};

template <typename MemorySpace>
struct ArborX::AccessTraits<TargetPoints<MemorySpace>, ArborX::PredicatesTag>
{
  static KOKKOS_FUNCTION std::size_t size(TargetPoints<MemorySpace> const &tp)
  {
    return tp.target_points.extent(0);
  }

  static KOKKOS_FUNCTION auto get(TargetPoints<MemorySpace> const &tp,
                                  std::size_t i)
  {
    return ArborX::nearest(tp.target_points(i), tp.num_neighbors);
  }

  using memory_space = MemorySpace;
};