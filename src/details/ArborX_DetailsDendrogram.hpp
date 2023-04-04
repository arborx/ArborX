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

#ifndef ARBORX_DETAILS_DENDROGRAM_HPP
#define ARBORX_DETAILS_DENDROGRAM_HPP

#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_DetailsUnionFind.hpp>
#include <ArborX_DetailsUtils.hpp> // iota

#include <Kokkos_Core.hpp>

namespace ArborX::Details
{

struct UnweightedEdge
{
  int source;
  int target;
};

template <typename Edges, typename Parents>
void dendrogramUnionFindHost(Edges sorted_edges_host, Parents &parents_host)
{
  Kokkos::Profiling::pushRegion(
      "ArborX::Dendrogram::dendrogram_union_find::union_find_host");

  using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;
  ExecutionSpace host_space;

  int const num_edges = sorted_edges_host.size();
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
  constexpr int UNDEFINED = -1;
  Kokkos::deep_copy(host_space, set_edges_host, UNDEFINED);

  // Fence all execution spaces to make sure all the data is ready
  Kokkos::fence("ArborX::Dendrogram::dendrogramUnionFindHost"
                " (global fence before performing union-find on host)");

  Kokkos::Profiling::pushRegion(
      "ArborX::Dendrogram::dendrogram_union_find::union_find");
  for (int e = 0; e < num_edges; ++e)
  {
    int i = union_find.representative(sorted_edges_host(e).source);
    int j = union_find.representative(sorted_edges_host(e).target);

    for (int k : {i, j})
    {
      auto edge_child = set_edges_host(k);
      if (edge_child != UNDEFINED)
        parents_host(edge_child) = e;
      else
        parents_host(vertices_offset + k) = e;
    }

    union_find.merge(i, j);

    set_edges_host(union_find.representative(i)) = e;
  }
  parents_host(num_edges - 1) = -1; // root
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::popRegion();
}

template <typename ExecutionSpace, typename MemorySpace>
void dendrogramUnionFind(
    ExecutionSpace const &exec_space,
    Kokkos::View<UnweightedEdge const *, MemorySpace> sorted_edges,
    Kokkos::View<int *, MemorySpace> &parents)
{
  Kokkos::Profiling::pushRegion("ArborX::Dendrogram::dendrogram_union_find");

  Kokkos::Profiling::pushRegion(
      "ArborX::Dendrogram::dendrogram_union_find::copy_to_host");

  auto sorted_edges_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, sorted_edges);
  auto parents_host = Kokkos::create_mirror_view(
      Kokkos::view_alloc(Kokkos::WithoutInitializing), parents);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion(
      "ArborX::Dendrogram::dendrogram_union_find::union_find");

  dendrogramUnionFindHost(sorted_edges_host, parents_host);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion(
      "ArborX::Dendrogram::dendrogram_union_find::copy_to_device");

  Kokkos::deep_copy(exec_space, parents, parents_host);

  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::popRegion();
}

} // namespace ArborX::Details

#endif
