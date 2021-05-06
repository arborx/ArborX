/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
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

// Do intersection queries using the same objects for the queries as the objects
// used in BVH construction that are located on a regular spaced
// three-dimensional grid.
//
// i-2  i-1  i  i+1
//
//  o    o   o   o   j+1
//          ---
//  o    o | x | o   j
//          ---
//  o    o   o   o   j-1
//
//  o    o   o   o   j-2
//

template <typename DeviceType>
Kokkos::View<ArborX::Box *, typename DeviceType::memory_space>
create_bounding_boxes(
    typename DeviceType::execution_space const &execution_space)
{
  float Lx = 100.0;
  float Ly = 100.0;
  float Lz = 100.0;
  int nx = 11;
  int ny = 11;
  int nz = 11;
  int n = nx * ny * nz;
  float hx = Lx / (nx - 1);
  float hy = Ly / (ny - 1);
  float hz = Lz / (nz - 1);

  auto index = [nx, ny](int i, int j, int k) {
    return i + j * nx + k * (nx * ny);
  };

  Kokkos::View<ArborX::Box *, typename DeviceType::memory_space> bounding_boxes(
      "bounding_boxes", n);
  auto bounding_boxes_host = Kokkos::create_mirror_view(bounding_boxes);

  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k)
      {
        ArborX::Point p_lower{{(i - .25) * hx, (j - .25) * hy, (k - .25) * hz}};
        ArborX::Point p_upper{{(i + .25) * hx, (j + .25) * hy, (k + .25) * hz}};
        bounding_boxes_host[index(i, j, k)] = {p_lower, p_upper};
      }
  Kokkos::deep_copy(execution_space, bounding_boxes, bounding_boxes_host);

  return bounding_boxes;
}

int main()
{
  Kokkos::initialize();
  {
    using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    using MemorySpace = typename ExecutionSpace::memory_space;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
    ExecutionSpace execution_space;

    std::cout << "Create grid with bounding boxes" << '\n';
    Kokkos::View<ArborX::Box *, MemorySpace> bounding_boxes =
        create_bounding_boxes<DeviceType>(execution_space);
    std::cout << "Bounding boxes set up." << '\n';

    std::cout << "Creating BVH tree." << '\n';
    ArborX::BVH<MemorySpace> const tree(execution_space, bounding_boxes);
    std::cout << "BVH tree set up." << '\n';

    std::cout << "Filling queries." << '\n';
    auto const n = bounding_boxes.size();
    Kokkos::View<decltype(ArborX::intersects(ArborX::Box{})) *, MemorySpace>
        queries("queries", n);
    Kokkos::parallel_for(
        "fill_queries",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space, 0, n),
        KOKKOS_LAMBDA(int i) {
          queries(i) = ArborX::intersects(bounding_boxes(i));
        });
    std::cout << "Queries set up." << '\n';

    std::cout << "Starting the queries." << '\n';
    // The query will resize indices and offsets accordingly
    Kokkos::View<int *, MemorySpace> indices("indices", 0);
    Kokkos::View<int *, MemorySpace> offsets("offsets", 0);

    ArborX::query(tree, execution_space, queries, indices, offsets);
    std::cout << "Queries done." << '\n';

    std::cout << "Starting checking results." << '\n';
    auto offsets_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets);
    auto indices_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices);

    if (offsets_host.size() != n + 1)
      Kokkos::abort("Wrong dimensions for the offsets View!\n");
    for (int i = 0; i < static_cast<int>(n + 1); ++i)
      if (offsets_host(i) != i)
        Kokkos::abort("Wrong entry in the offsets View!\n");

    if (indices_host.size() != n)
      Kokkos::abort("Wrong dimensions for the indices View!\n");
    for (int i = 0; i < static_cast<int>(n); ++i)
      if (indices_host(i) != i)
        Kokkos::abort("Wrong entry in the indices View!\n");
    std::cout << "Checking results successful." << '\n';
  }

  Kokkos::finalize();
}
