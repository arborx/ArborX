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

#include <ArborX_DBSCAN.hpp>

#include <iostream>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

template <typename MemorySpace>
void printLabels(Kokkos::View<int *, MemorySpace> labels)
{
  auto labels_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, labels);
  for (int i = 0; i < (int)labels_host.size(); ++i)
    std::cout << labels_host(i) << " ";
  std::cout << '\n';
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  Kokkos::View<ArborX::Point *, MemorySpace> cloud("point_cloud", 10);
  auto cloud_host = Kokkos::create_mirror_view(cloud);
  // 4 |       0 7
  // 3 |       5 6
  // 2 |     9
  // 1 | 4 3
  // 0 | 1 2     8
  //    ----------
  //     0 1 2 3 4
  cloud_host[0] = {4.f, 3.f, 0.f};
  cloud_host[1] = {0.f, 0.f, 0.f};
  cloud_host[2] = {0.f, 1.f, 0.f};
  cloud_host[3] = {1.f, 1.f, 0.f};
  cloud_host[4] = {1.f, 0.f, 0.f};
  cloud_host[5] = {3.f, 3.f, 0.f};
  cloud_host[6] = {3.f, 4.f, 0.f};
  cloud_host[7] = {4.f, 4.f, 0.f};
  cloud_host[8] = {4.f, 0.f, 0.f};
  cloud_host[9] = {2.f, 2.f, 0.f};
  Kokkos::deep_copy(cloud, cloud_host);

  // Running with minpts = 2 and eps = 1 would produce two clusters consisting
  // of points with indices [1, 2, 3, 4] and [0, 5, 6, 7]. The corresponding
  // entries in the labels array would be the same.
  // Expected output:
  //   0 1 1 1 1 0 0 0 -1 -1
  auto labels = ArborX::dbscan(ExecutionSpace{}, cloud, 1.f, 2);
  printLabels(labels);

  // Running with minpts = 5 and eps = 1 would produce no clusters. All the
  // entries in the labels array would be marked as noise, -1.
  // Expected output:
  //   -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
  labels = ArborX::dbscan(ExecutionSpace{}, cloud, 1.f, 5);
  printLabels(labels);

  // Running with minpts = 2 and eps = 1.5 would produce a single cluster
  // consisting of points with indices [0, 1, 2, 3, 4, 5, 6, 7, 9].
  // Expected output:
  //   0 0 0 0 0 0 0 0 -1 0
  labels = ArborX::dbscan(ExecutionSpace{}, cloud, 1.5f, 2);
  printLabels(labels);

  // Running with minpts = 4 and eps = 1.5 would produce two clusters
  // consisting of points with indices [1, 2, 3, 4] and [0, 5, 6, 7]. The point
  // with index 9 is a border point and may be assigned to the first, or the
  // second cluster.
  // Expected output:
  //   0 1 1 1 1 0 0 0 -1 0
  // or
  //   0 1 1 1 1 0 0 0 -1 1
  labels = ArborX::dbscan(ExecutionSpace{}, cloud, 1.5f, 4);
  printLabels(labels);

  return 0;
}
