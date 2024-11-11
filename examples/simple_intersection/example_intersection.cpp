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

#include <Kokkos_Core.hpp>

#include <iostream>

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;

  using Box = ArborX::Box<2>;
  using Point = ArborX::Point<2>;

  Kokkos::View<Box *, MemorySpace> boxes("Example::boxes", 4);
  auto boxes_host = Kokkos::create_mirror_view(boxes);
  boxes_host[0] = {{0, 0}, {1, 1}};
  boxes_host[1] = {{1, 0}, {2, 1}};
  boxes_host[2] = {{0, 1}, {1, 2}};
  boxes_host[3] = {{1, 1}, {2, 2}};
  Kokkos::deep_copy(boxes, boxes_host);

  // -----------
  // |    | 1  |
  // |    |  0 |
  // |----2----|
  // |    |    |
  // |    |    |
  // -----------
  Kokkos::View<decltype(ArborX::intersects(Point{})) *, MemorySpace> queries(
      "Example::queries", 3);
  auto queries_host = Kokkos::create_mirror_view(queries);
  queries_host[0] = ArborX::intersects(Point{1.8, 1.5});
  queries_host[1] = ArborX::intersects(Point{1.3, 1.7});
  queries_host[2] = ArborX::intersects(Point{1, 1});
  Kokkos::deep_copy(queries, queries_host);

  ExecutionSpace space;

  using Value = ArborX::PairValueIndex<Box>;

  ArborX::BoundingVolumeHierarchy const tree(
      space, ArborX::Experimental::attach_indices(boxes));

  // The query will resize values and offsets accordingly
  Kokkos::View<Value *, MemorySpace> values("Example::values", 0);
  Kokkos::View<int *, MemorySpace> offsets("Example::offsets", 0);
  tree.query(space, queries, values, offsets);

  auto offsets_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets);
  auto values_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, values);

  // Expected output:
  //   offsets: 0 1 2 6
  //   indices: 3 3 0 2 1 3
  // The order of the last 4 indices may vary.
  std::cout << "offsets: ";
  std::copy(offsets_host.data(), offsets_host.data() + offsets.size(),
            std::ostream_iterator<int>(std::cout, " "));
  std::cout << "\nindices:";
  for (int i = 0; i < values_host.extent_int(0); ++i)
    std::cout << " " << values_host(i).index;
  std::cout << '\n';

  return 0;
}
