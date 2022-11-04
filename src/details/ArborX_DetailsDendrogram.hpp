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

#ifndef ARBORX_DETAILS_DENDROGRAM_HPP
#define ARBORX_DETAILS_DENDROGRAM_HPP

#include <ArborX_DetailsKokkosExtSwap.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_DetailsUnionFind.hpp>
#include <ArborX_MinimumSpanningTree.hpp> // WeightedEdge

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ProfileSection.hpp>

namespace ArborX::Details
{

// Compute permutation that orders edges in increasing order
template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<unsigned int *, MemorySpace>
computeEdgesPermutation(ExecutionSpace const &exec_space,
                        Kokkos::View<WeightedEdge *, MemorySpace> &edges)
{
  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::sort_edges");

  int const num_edges = edges.size();

  Kokkos::View<float *, MemorySpace> weights(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::weights"),
      num_edges);
  Kokkos::parallel_for(
      "ArborX::Dendrogram::copy_weights",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
      KOKKOS_LAMBDA(int const e) { weights(e) = edges(e).weight; });

  auto permute = Details::sortObjects(exec_space, weights);

  Kokkos::Profiling::popRegion();

  return permute;
}

template <typename Edges, typename Permute, typename Parents>
void dendrogramUnionFindHost(Edges edges_host, Permute permute,
                             Parents &parents_host)
{
  using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
  ExecutionSpace host_space;

  int const num_edges = edges_host.size();
  int const num_vertices = num_edges + 1;
  auto const vertices_offset = num_edges;

  Kokkos::View<int *, Kokkos::HostSpace> labels(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::dendrogram_union_find::labels"),
      num_vertices);
  iota(host_space, labels);
  UnionFind<Kokkos::HostSpace, true> union_find(labels);

  Kokkos::View<int *, Kokkos::HostSpace> set_edges_host(
      Kokkos::view_alloc(host_space, Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::labels"),
      num_vertices);
  iota(host_space, set_edges_host, vertices_offset);

  Kokkos::parallel_for(
      "ArborX::Dendrogram::dendrogram_union_find::union_find",
      Kokkos::RangePolicy<ExecutionSpace>(host_space, 0, 1),
      KOKKOS_LAMBDA(int) {
        for (int k = 0; k < num_edges; ++k)
        {
          auto e = permute(k);

          int i = union_find.representative(edges_host(e).source);
          int j = union_find.representative(edges_host(e).target);

          parents_host(set_edges_host(i)) = e;
          parents_host(set_edges_host(j)) = e;

          // For the algorithm to work properly, we assume that the union-find
          // algorithm will assign the smaller index to the merged component.
          union_find.merge(i, j);

          set_edges_host(union_find.representative(i)) = e;
        }
        parents_host(set_edges_host(union_find.representative(0))) = -1; // root
      });
}

template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace>
dendrogramUnionFind(ExecutionSpace const &exec_space,
                    Kokkos::View<WeightedEdge *, MemorySpace> edges)
{
  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::dendrogram_union_find");

  Kokkos::Profiling::ProfilingSection profile_edge_sort(
      "ArborX::Dendrogram::edge_sort");
  profile_edge_sort.start();
  auto permute = computeEdgesPermutation(exec_space, edges);
  profile_edge_sort.stop();

  auto const num_vertices = edges.size() + 1;

  Kokkos::View<int *, MemorySpace> parents(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::Dendrogram::edge_parents"),
      2 * num_vertices - 1);

  Kokkos::Profiling::pushRegion(
      "ArborX::Dendrogram::dendrogram_union_find::copy_to_host");

  auto edges_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, edges);
  auto permute_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, permute);
  auto parents_host = Kokkos::create_mirror_view(parents);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion(
      "ArborX::Dendrogram::dendrogram_union_find::union_find");

  dendrogramUnionFindHost(edges_host, permute_host, parents_host);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion(
      "ArborX::Dendrogram::dendrogram_union_find::copy_to_device");

  Kokkos::deep_copy(exec_space, parents, parents_host);

  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::popRegion();

  return parents;
}

} // namespace ArborX::Details

#endif
