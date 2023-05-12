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

#include "distributed_setup.hpp"
#include <mpi.h>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

namespace Example
{
template <class Points>
struct Intersects
{
  Points points;
  float radius;
};
template <class Points>
Intersects(Points const &, int) -> Intersects<Points>;

struct IndexAndRank
{
  int index;
  int rank;
};

} // namespace Example

template <class Points>
struct ArborX::AccessTraits<Example::Intersects<Points>, ArborX::PredicatesTag>
{
  static KOKKOS_FUNCTION std::size_t size(Example::Intersects<Points> const &x)
  {
    return x.points.extent(0);
  }
  static KOKKOS_FUNCTION auto get(Example::Intersects<Points> const &x,
                                  std::size_t i)
  {
    return ArborX::intersects(ArborX::Sphere(x.points(i), x.radius));
  }
  using memory_space = MemorySpace;
};

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    MPI_Comm comm = MPI_COMM_WORLD;
    ExecutionSpace exec;
    Kokkos::View<ArborX::Point *, MemorySpace> points_device;
    ArborX::DistributedTree<MemorySpace> tree =
        ArborXExample::create_tree(comm, exec, points_device);

    Kokkos::View<Example::IndexAndRank *, MemorySpace> values("values", 0);
    Kokkos::View<int *, MemorySpace> offsets("offsets", 0);
    tree.query(exec, Example::Intersects{points_device, 1.}, values, offsets);

    auto host_values =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, values);
    auto host_offsets =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets);
    int comm_rank;
    MPI_Comm_rank(comm, &comm_rank);
    for (unsigned int i = 0; i + 1 < host_offsets.size(); ++i)
    {
      std::cout << "Results for query " << i << " on MPI rank " << comm_rank
                << '\n';
      for (int j = host_offsets(i); j < host_offsets(i + 1); ++j)
        std::cout << "point " << host_values(j).index << ", rank "
                  << host_values(j).rank << std::endl;
    }
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
