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

#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

#include <mpi.h>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

int global_mpi_rank;

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

template <typename DeviceType>
struct InlinePrintCallback
{
  Kokkos::View<ArborX::Point *, DeviceType> points;
  int mpi_rank;

  InlinePrintCallback(Kokkos::View<ArborX::Point *, DeviceType> const &points_,
                      int mpi_rank_)
      : points(points_)
      , mpi_rank(mpi_rank_)
  {}

  template <typename Predicate, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate,
                                  int primitive_index,
                                  OutputFunctor const &out) const
  {
    auto data = ArborX::getData(predicate);
    auto const &point = points(primitive_index);
    printf("Intersection for query %d from MPI rank %d on MPI rank %d for "
           "point %f,%f,%f with index %d\n",
           data.index, data.rank, mpi_rank, point[0], point[1], point[2],
           primitive_index);

    out({primitive_index, mpi_rank});
  }
};

} // namespace Example

template <class Points>
struct ArborX::AccessTraits<Example::Intersects<Points>, ArborX::PredicatesTag>
{
  static KOKKOS_FUNCTION std::size_t size(Example::Intersects<Points> const &x)
  {
    return x.points.extent(0);
  }
  static KOKKOS_FUNCTION auto get(Example::Intersects<Points> const &x, int i)
  {
    return attach(ArborX::intersects(ArborX::Sphere(x.points(i), x.radius)),
                  Example::IndexAndRank{i, global_mpi_rank});
  }
  using memory_space = MemorySpace;
};

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    MPI_Comm comm = MPI_COMM_WORLD;
    int comm_rank;
    MPI_Comm_rank(comm, &comm_rank);
    global_mpi_rank = comm_rank;
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    ArborX::Point lower_left_corner = {static_cast<float>(comm_rank),
                                       static_cast<float>(comm_rank),
                                       static_cast<float>(comm_rank)};
    ArborX::Point center = {static_cast<float>(comm_rank) + .5f,
                            static_cast<float>(comm_rank) + .5f,
                            static_cast<float>(comm_rank) + .5f};
    std::vector<ArborX::Point> points = {lower_left_corner, center};
    auto points_device = Kokkos::create_mirror_view_and_copy(
        MemorySpace{},
        Kokkos::View<ArborX::Point *, Kokkos::HostSpace,
                     Kokkos::MemoryUnmanaged>(points.data(), points.size()));

    ExecutionSpace exec;
    ArborX::DistributedTree<MemorySpace> tree(comm, exec, points_device);

    Kokkos::View<Example::IndexAndRank *, MemorySpace> values("values", 0);
    Kokkos::View<int *, MemorySpace> offsets("offsets", 0);
    tree.query(
        exec, Example::Intersects{points_device, 1.},
        Example::InlinePrintCallback<MemorySpace>(points_device, comm_rank),
        values, offsets);

    auto host_values =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, values);
    auto host_offsets =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets);
    for (unsigned int i = 0; i + 1 < host_offsets.size(); ++i)
    {
      std::cout << "Results for query " << i << " on MPI rank " << comm_rank
                << '\n';
      for (unsigned int j = host_offsets(i); j < host_offsets(i + 1); ++j)
        std::cout << "point " << host_values(j).index << ", rank "
                  << host_values(j).rank << std::endl;
    }
  }
  Kokkos::finalize();
  MPI_Finalize();
  return 0;
}
