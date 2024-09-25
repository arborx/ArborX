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
#include <iterator>
#include <vector>

#include <mpi.h>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

struct PairIndexRank
{
  unsigned index;
  int rank;
};

std::ostream &operator<<(std::ostream &os, PairIndexRank const &x)
{
  os << "(" << x.index << ", " << x.rank << ")";
  return os;
}

struct CallbackWithRank
{
  int _rank;

  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const &,
                                  ArborX::PairValueIndex<Value> const &value,
                                  OutputFunctor const &out) const
  {
    out({value.index, _rank});
  }
};

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  {
    MPI_Comm comm = MPI_COMM_WORLD;

    int comm_rank;
    MPI_Comm_rank(comm, &comm_rank);
    int comm_size;
    MPI_Comm_size(comm, &comm_size);

    using Point = ArborX::Point<3>;

    // ranks: | 0 | 1 | 2 |
    //        -------------
    //        |   |   |  x|
    //        |   |   |x  |
    //        |   |  x|   |
    //        |   |x  |   |
    //        |  x|   |   |
    //        |x  |   |   |
    Point lower_left_corner = {(float)comm_rank, (float)comm_rank,
                               (float)comm_rank};
    Point center = {comm_rank + .5f, comm_rank + .5f, comm_rank + .5f};
    std::vector<Point> points = {lower_left_corner, center};

    auto points_device = Kokkos::create_mirror_view_and_copy(
        MemorySpace{},
        Kokkos::View<Point *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
            points.data(), points.size()));

    ExecutionSpace exec;
    ArborX::DistributedTree tree(
        comm, exec, ArborX::Experimental::attach_indices(points_device));

    Kokkos::View<PairIndexRank *, MemorySpace> values("Example::values", 0);
    Kokkos::View<int *, MemorySpace> offsets("Example::offsets", 0);
    tree.query(exec, ArborX::Experimental::make_nearest(points_device, 3),
               ArborX::Experimental::declare_callback_constrained(
                   CallbackWithRank{comm_rank}),
               values, offsets);

    if (comm_rank == 0)
    {
      auto offsets_host =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets);
      auto values_host =
          Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, values);

      // Expected output for 2+ ranks:
      //   offsets: 0 3 6
      //   values : (0, 0) (1, 0) (0, 1) (1, 0) (0, 0) (0, 1)
      // The order of the last 4 values may vary.
      std::cout << "offsets: ";
      std::copy(offsets_host.data(), offsets_host.data() + offsets.size(),
                std::ostream_iterator<int>(std::cout, " "));
      std::cout << "\nvalues : ";
      std::copy(values_host.data(), values_host.data() + values.size(),
                std::ostream_iterator<PairIndexRank>(std::cout, " "));
      std::cout << "\n";
    }
  }

  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}
