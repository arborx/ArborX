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

#include <ArborX_MinimumSpanningTree.hpp>

#include <Kokkos_Sort.hpp>

#include <iostream>

template <typename MemorySpace>
void printEdges(
    Kokkos::View<ArborX::Experimental::WeightedEdge *, MemorySpace> edges)
{
  auto edges_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, edges);
  Kokkos::sort(edges_host);
  for (int i = 0; i < (int)edges_host.size(); ++i)
  {
    auto const &edge = edges_host(i);
    auto min_index = std::min(edge.source, edge.target);
    auto max_index = std::max(edge.source, edge.target);
    printf("#%d: (%d, %d, %.2f)\n", i, min_index, max_index, edge.weight);
  }
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;

  Kokkos::View<ArborX::Point<2> *, MemorySpace> cloud("Example::point_cloud",
                                                      6);
  auto cloud_host = Kokkos::create_mirror_view(cloud);
  // 4 |     5
  // 3 |   2 4
  // 2 |         3
  // 1 | 1
  // 0 |   0
  //    ----------
  //     0 1 2 3 4
  cloud_host[0] = {1, 0};
  cloud_host[1] = {0, 1};
  cloud_host[2] = {1, 3};
  cloud_host[3] = {4, 2};
  cloud_host[4] = {2, 3};
  cloud_host[5] = {2, 4};
  Kokkos::deep_copy(cloud, cloud_host);

  ArborX::Experimental::MinimumSpanningTree<MemorySpace> emst(ExecutionSpace{},
                                                              cloud);
  auto edges = emst.edges;

  // Expected output:
  // #0: (2, 4, 1.00)
  // #1: (4, 5, 1.00)
  // #2: (0, 1, 1.41)
  // #3: (3, 4, 2.24)
  // #4: (1, 2, 2.24)
  printEdges(edges);

  return 0;
}
