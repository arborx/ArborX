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
#ifndef ARBORX_NEAREST_BUFFER_PROVIDER_HPP
#define ARBORX_NEAREST_BUFFER_PROVIDER_HPP

#include <kokkos_ext/ArborX_KokkosExtStdAlgorithms.hpp>
#include <kokkos_ext/ArborX_KokkosExtViewHelpers.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX::Details
{

template <typename MemorySpace, typename Coordinate>
struct NearestBufferProvider
{
  static_assert(Kokkos::is_memory_space_v<MemorySpace>);

  using PairIndexDistance = Kokkos::pair<int, Coordinate>;

  Kokkos::View<PairIndexDistance *, MemorySpace> _buffer;
  Kokkos::View<int *, MemorySpace> _offset;

  NearestBufferProvider()
      : _buffer("ArborX::NearestBufferProvider::buffer", 0)
      , _offset("ArborX::NearestBufferProvider::offset", 0)
  {}

  template <typename ExecutionSpace, typename Predicates>
  NearestBufferProvider(ExecutionSpace const &space,
                        Predicates const &predicates)
      : _buffer("ArborX::NearestBufferProvider::buffer", 0)
      , _offset("ArborX::NearestBufferProvider::offset", 0)
  {
    allocateBuffer(space, predicates);
  }

  KOKKOS_FUNCTION auto operator()(int i) const
  {
    return Kokkos::subview(_buffer,
                           Kokkos::make_pair(_offset(i), _offset(i + 1)));
  }

  template <typename ExecutionSpace, typename Predicates>
  void allocateBuffer(ExecutionSpace const &space, Predicates const &predicates)
  {
    auto const n_queries = predicates.size();

    KokkosExt::reallocWithoutInitializing(space, _offset, n_queries + 1);

    Kokkos::parallel_for(
        "ArborX::NearestBufferProvider::scan_queries_for_numbers_of_neighbors",
        Kokkos::RangePolicy(space, 0, n_queries),
        KOKKOS_CLASS_LAMBDA(int i) { _offset(i) = getK(predicates(i)); });
    KokkosExt::exclusive_scan(space, _offset, _offset, 0);
    int const buffer_size = KokkosExt::lastElement(space, _offset);
    // Allocate buffer over which to perform heap operations in the nearest
    // query to store nearest nodes found so far.
    // It is not possible to anticipate how much memory to allocate since the
    // number of nearest neighbors k is only known at runtime.

    KokkosExt::reallocWithoutInitializing(space, _buffer, buffer_size);
  }
};

} // namespace ArborX::Details

#endif
