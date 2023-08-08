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

// Perform intersection queries using the same objects for the queries as the
// objects used in BVH construction that are located on a regular spaced
// three-dimensional grid.
// Each box will only intersect with itself.
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
class Boxes
{
public:
  // Create non-intersecting boxes on a 3D cartesian grid
  // used both for queries and predicates.
  Boxes(typename DeviceType::execution_space const &execution_space)
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

    _boxes = Kokkos::View<ArborX::Box *, typename DeviceType::memory_space>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "boxes"), n);
    auto boxes_host = Kokkos::create_mirror_view(_boxes);

    for (int i = 0; i < nx; ++i)
      for (int j = 0; j < ny; ++j)
        for (int k = 0; k < nz; ++k)
        {
          ArborX::Point p_lower{
              {(i - .25f) * hx, (j - .25f) * hy, (k - .25f) * hz}};
          ArborX::Point p_upper{
              {(i + .25f) * hx, (j + .25f) * hy, (k + .25f) * hz}};
          boxes_host[index(i, j, k)] = {p_lower, p_upper};
        }
    Kokkos::deep_copy(execution_space, _boxes, boxes_host);
  }

  // Return the number of boxes.
  KOKKOS_FUNCTION int size() const { return _boxes.size(); }

  // Return the box with index i.
  KOKKOS_FUNCTION ArborX::Box const &get_box(int i) const { return _boxes(i); }

private:
  Kokkos::View<ArborX::Box *, typename DeviceType::memory_space> _boxes;
};

// For creating the bounding volume hierarchy given a Boxes object, we
// need to define the memory space, how to get the total number of objects,
// and how to access a specific box. Since there are corresponding functions in
// the Boxes class, we just resort to them.
template <typename DeviceType>
struct ArborX::AccessTraits<Boxes<DeviceType>, ArborX::PrimitivesTag>
{
  using memory_space = typename DeviceType::memory_space;
  static KOKKOS_FUNCTION int size(Boxes<DeviceType> const &boxes)
  {
    return boxes.size();
  }
  static KOKKOS_FUNCTION auto get(Boxes<DeviceType> const &boxes, int i)
  {
    return boxes.get_box(i);
  }
};

// For performing the queries given a Boxes object, we need to define memory
// space, how to get the total number of queries, and what the query with index
// i should look like. Since we are using self-intersection (which boxes
// intersect with the given one), the functions here very much look like the
// ones in ArborX::AccessTraits<Boxes<DeviceType>, ArborX::PrimitivesTag>.
template <typename DeviceType>
struct ArborX::AccessTraits<Boxes<DeviceType>, ArborX::PredicatesTag>
{
  using memory_space = typename DeviceType::memory_space;
  static KOKKOS_FUNCTION int size(Boxes<DeviceType> const &boxes)
  {
    return boxes.size();
  }
  static KOKKOS_FUNCTION auto get(Boxes<DeviceType> const &boxes, int i)
  {
    return intersects(boxes.get_box(i));
  }
};

// Now that we have encapsulated the objects and queries to be used within the
// Boxes class, we can continue with performing the actual search.
int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
  ExecutionSpace execution_space;

  std::cout << "Create grid with bounding boxes" << '\n';
  Boxes<DeviceType> boxes(execution_space);
  std::cout << "Bounding boxes set up." << '\n';

  std::cout << "Creating BVH tree." << '\n';
  ArborX::BVH<MemorySpace> const tree(execution_space, boxes);
  std::cout << "BVH tree set up." << '\n';

  std::cout << "Starting the queries." << '\n';
  // The query will resize indices and offsets accordingly
  Kokkos::View<int *, MemorySpace> indices("indices", 0);
  Kokkos::View<int *, MemorySpace> offsets("offsets", 0);

  ArborX::query(tree, execution_space, boxes, indices, offsets);
  std::cout << "Queries done." << '\n';

  std::cout << "Starting checking results." << '\n';
  auto offsets_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets);
  auto indices_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices);

  unsigned int const n = boxes.size();
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
