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

#include <mpi.h>

namespace ArborXExample
{

template <typename ExecutionSpace, typename MemorySpace>
ArborX::DistributedTree<MemorySpace>
create_tree(MPI_Comm comm, ExecutionSpace const &exec,
            Kokkos::View<ArborX::Point *, MemorySpace> &points_device)
{
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  ArborX::Point lower_left_corner = {static_cast<float>(comm_rank),
                                     static_cast<float>(comm_rank),
                                     static_cast<float>(comm_rank)};
  ArborX::Point center = {static_cast<float>(comm_rank) + .5f,
                          static_cast<float>(comm_rank) + .5f,
                          static_cast<float>(comm_rank) + .5f};
  std::vector<ArborX::Point> points = {lower_left_corner, center};
  points_device = Kokkos::create_mirror_view_and_copy(
      MemorySpace{},
      Kokkos::View<ArborX::Point *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
          points.data(), points.size()));

  ArborX::DistributedTree<MemorySpace> tree(comm, exec, points_device);
  return tree;
}
} // namespace ArborXExample
