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

#include <ArborX.hpp>
#include <ArborX_Triangle.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;

  using Point = ArborX::Point<2>;
  using Triangle = ArborX::Triangle<2>;

  // Vertices:
  // 6_____7_____8
  // |\    |\    |
  // |  \  |  \  |
  // 3____\4____\5
  // |\    |\    |
  // |  \  |  \  |
  // 0____\1____\2
  std::vector<Point> v(9);
  v[0] = {0, 0};
  v[1] = {1, 0};
  v[2] = {2, 0};
  v[3] = {0, 1};
  v[4] = {1, 1};
  v[5] = {2, 1};
  v[6] = {0, 2};
  v[7] = {1, 2};
  v[8] = {2, 2};

  // Triangles:
  // _____________
  // |\  5 |\  7 |
  // |4 \  |6 \  |
  // |____\|____\|
  // |\  1 |\  3 |
  // |0 \  |2 \  |
  // |____\|____\|
  Kokkos::View<Triangle *, MemorySpace> triangles("Example::triangles", 8);
  auto triangles_host = Kokkos::create_mirror_view(triangles);
  triangles_host[0] = {v[0], v[1], v[3]};
  triangles_host[1] = {v[1], v[4], v[3]};
  triangles_host[2] = {v[1], v[2], v[4]};
  triangles_host[3] = {v[2], v[5], v[4]};
  triangles_host[4] = {v[3], v[4], v[6]};
  triangles_host[5] = {v[4], v[7], v[6]};
  triangles_host[6] = {v[4], v[5], v[7]};
  triangles_host[7] = {v[5], v[8], v[7]};
  Kokkos::deep_copy(triangles, triangles_host);

  // Query points:
  // _____________
  // |\  x |\   x|
  // |  \  |  \  |
  // |____\|____\|
  // |\    |\    |
  // |  \ x|x \  |
  // _____\|____\|
  Kokkos::View<decltype(ArborX::intersects(Point())) *, MemorySpace> queries(
      "Example::queries", 4);
  auto queries_host = Kokkos::create_mirror_view(queries);
  queries_host[0] = ArborX::intersects(Point{0.8f, 0.3f});
  queries_host[1] = ArborX::intersects(Point{1.1f, 0.5f});
  queries_host[2] = ArborX::intersects(Point{0.6f, 1.8f});
  queries_host[3] = ArborX::intersects(Point{1.9f, 1.9f});
  Kokkos::deep_copy(queries, queries_host);

  ExecutionSpace space;

  ArborX::BoundingVolumeHierarchy const tree(
      space, ArborX::Experimental::attach_indices(triangles));

  // The query will resize indices and offsets accordingly
  Kokkos::View<unsigned *, MemorySpace> indices("Example::indices", 0);
  Kokkos::View<int *, MemorySpace> offsets("Example::offsets", 0);
  tree.query(space, queries, ArborX::Details::LegacyDefaultCallback{}, indices,
             offsets);

  auto offsets_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets);
  auto indices_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices);

  // Expected output:
  //   offsets: 0 1 2 3 4
  //   indices: 1 2 5 7
  std::cout << "offsets: ";
  std::copy(offsets_host.data(), offsets_host.data() + offsets.size(),
            std::ostream_iterator<int>(std::cout, " "));
  std::cout << "\nindices: ";
  std::copy(indices_host.data(), indices_host.data() + indices.size(),
            std::ostream_iterator<int>(std::cout, " "));
  std::cout << "\n";

  return 0;
}
