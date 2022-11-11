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
#ifndef ARBORX_DENDROGRAM_HPP
#define ARBORX_DENDROGRAM_HPP

#include <ArborX_DetailsDendrogram.hpp>
#include <ArborX_DetailsWeightedEdge.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX::Experimental
{

template <typename MemorySpace>
struct Dendrogram
{
  Kokkos::View<int *, MemorySpace> _parents;
  Kokkos::View<float *, MemorySpace> _parent_heights;

  template <typename ExecutionSpace>
  Dendrogram(ExecutionSpace const &exec_space,
             Kokkos::View<Details::WeightedEdge *, MemorySpace> edges)
      : _parents("ArborX::Dendrogram::parents", 0)
      , _parent_heights("ArborX::Dendrogram::parent_heights", 0)
  {
    Kokkos::Profiling::pushRegion("ArborX::Dendrogram::Dendrogram");

    using ConstEdges = Kokkos::View<Details::WeightedEdge const *, MemorySpace>;
    Details::dendrogramUnionFind(exec_space, ConstEdges(edges), _parents,
                                 _parent_heights);

    Kokkos::Profiling::popRegion();
  }
};

} // namespace ArborX::Experimental

#endif
