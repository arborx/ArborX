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
#ifndef ARBORX_HDBSCAN_HPP
#define ARBORX_HDBSCAN_HPP

#include <ArborX_Dendrogram.hpp>
#include <ArborX_DetailsKokkosExtScopedProfileRegion.hpp>
#include <ArborX_MinimumSpanningTree.hpp>

namespace ArborX::Experimental
{

template <typename ExecutionSpace, typename Primitives>
auto hdbscan(ExecutionSpace const &exec_space, Primitives const &primitives,
             int core_min_size,
             DendrogramImplementation dendrogram_impl =
                 DendrogramImplementation::BORUVKA)
{
  KokkosExt::ScopedProfileRegion guard("ArborX::HDBSCAN");

  using namespace ArborX::Details;

  using MemorySpace = typename Primitives::memory_space;

  if (dendrogram_impl == DendrogramImplementation::BORUVKA)
  {
    // Hybrid Boruvka+dendrogram
    MinimumSpanningTree<MemorySpace, BoruvkaMode::HDBSCAN> mst(
        exec_space, primitives, core_min_size);
    return Dendrogram<MemorySpace>{mst.dendrogram_parents,
                                   mst.dendrogram_parent_heights};
  }

  Kokkos::Profiling::pushRegion("ArborX::HDBSCAN::mst");
  MinimumSpanningTree<MemorySpace> mst(exec_space, primitives, core_min_size);
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::pushRegion("ArborX::HDBSCAN::dendrogram");
  Dendrogram<MemorySpace> dendrogram(exec_space, mst.edges);
  Kokkos::Profiling::popRegion();

  return dendrogram;
}

} // namespace ArborX::Experimental

#endif
