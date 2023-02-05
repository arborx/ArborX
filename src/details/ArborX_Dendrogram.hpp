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
#ifndef ARBORX_DENDROGRAM_HPP
#define ARBORX_DENDROGRAM_HPP

#include <ArborX_DetailsDendrogram.hpp>
#include <ArborX_DetailsKokkosExtSort.hpp>
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

    auto const num_edges = edges.size();
    auto const num_vertices = num_edges + 1;

    KokkosExt::reallocWithoutInitializing(exec_space, _parents,
                                          num_edges + num_vertices);
    KokkosExt::reallocWithoutInitializing(exec_space, _parent_heights,
                                          num_edges);

    Kokkos::View<Details::UnweightedEdge *, MemorySpace> unweighted_edges(
        Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                           "ArborX::Dendrogram::unweighted_edges"),
        num_edges);
    splitEdges(exec_space, edges, unweighted_edges, _parent_heights);

    Kokkos::Profiling::pushRegion("ArborX::Dendrogram::sort_edges");
    KokkosExt::sortByKey(exec_space, _parent_heights, unweighted_edges);
    Kokkos::Profiling::popRegion();

    using ConstEdges =
        Kokkos::View<Details::UnweightedEdge const *, MemorySpace>;
    Details::dendrogramUnionFind(exec_space, ConstEdges(unweighted_edges),
                                 _parents);

    Kokkos::Profiling::popRegion();
  }

  template <typename ExecutionSpace>
  void splitEdges(
      ExecutionSpace const &exec_space,
      Kokkos::View<Details::WeightedEdge *, MemorySpace> edges,
      Kokkos::View<Details::UnweightedEdge *, MemorySpace> unweighted_edges,
      Kokkos::View<float *, MemorySpace> weights)
  {
    Kokkos::parallel_for(
        "ArborX::Dendrogram::copy_weights_and_edges",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, edges.size()),
        KOKKOS_LAMBDA(int const e) {
          weights(e) = edges(e).weight;
          unweighted_edges(e) = {edges(e).source, edges(e).target};
        });
  }
};

} // namespace ArborX::Experimental

#endif
