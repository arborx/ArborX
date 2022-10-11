/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
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
#include <ArborX_MinimumSpanningTree.hpp>

#include <Kokkos_Profiling_ProfileSection.hpp>

#include <iomanip>
#include <limits>

namespace ArborX::Experimental
{

template <typename ExecutionSpace, typename Primitives>
auto hdbscan(ExecutionSpace const &exec_space, Primitives const &primitives,
             int core_min_size)
{
  Kokkos::Profiling::pushRegion("ArborX::HDBSCAN");

  using MemorySpace = typename Primitives::memory_space;

  Kokkos::Profiling::ProfilingSection profile_mst("ArborX::HDBSCAN::mst");
  profile_mst.start();
  Details::MinimumSpanningTree<MemorySpace> mst(exec_space, primitives,
                                                core_min_size);
  profile_mst.stop();

  Kokkos::Profiling::ProfilingSection profile_dendrogram(
      "ArborX::HDBSCAN::dendrogram");
  profile_dendrogram.start();
  Kokkos::Profiling::pushRegion("ArborX::HDBSCAN::dendrogram");
  Dendrogram<MemorySpace> dendrogram(exec_space, mst.edges);
  Kokkos::Profiling::popRegion();
  profile_dendrogram.stop();

  Kokkos::Profiling::popRegion();

  return std::make_pair(dendrogram._parents, dendrogram._parent_heights);
}

} // namespace ArborX::Experimental

#endif
