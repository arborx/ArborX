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

#ifndef ARBORX_DISTRIBUTED_DBSCAN_HPP
#define ARBORX_DISTRIBUTED_DBSCAN_HPP

#include <ArborX_DBSCAN.hpp>
#include <detail/ArborX_AccessTraits.hpp>

#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <mpi.h>

namespace ArborX::Experimental
{

template <typename ExecutionSpace, typename Primitives, typename Coordinate,
          typename Labels>
void dbscan(MPI_Comm comm, ExecutionSpace const &exec_space,
            Primitives const &primitives, Coordinate eps, int core_min_size,
            Labels &labels,
            DBSCAN::Parameters const &parameters = DBSCAN::Parameters())
{
  Kokkos::Profiling::ScopedRegion guard("ArborX::DistributedDBSCAN");

  namespace KokkosExt = ArborX::Details::KokkosExt;

  using Points = Details::AccessValues<Primitives>;
  using MemorySpace = typename Points::memory_space;

  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value,
      "Primitives must be accessible from the execution space");

  ARBORX_ASSERT(eps > 0);
  ARBORX_ASSERT(core_min_size >= 2);

  using Point = typename Points::value_type;
  static_assert(GeometryTraits::is_point_v<Point>);
  static_assert(
      std::is_same_v<typename GeometryTraits::coordinate_type<Point>::type,
                     Coordinate>);

  Points points{primitives}; // NOLINT
  auto const n = points.size();

  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  std::vector<int> counts(comm_size);
  counts[comm_rank] = n;
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, counts.data(), 1, MPI_INT,
                comm);
  std::vector<int> offsets(comm_size + 1);
  offsets[0] = 0;
  for (int i = 0; i < comm_size; ++i)
    offsets[i + 1] = counts[i] + offsets[i];

  auto const total_n = offsets.back();
  auto rank_offset = offsets[comm_rank];

  Kokkos::View<Point *, MemorySpace> data(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::DistributedDBSCAN::data"),
      total_n);
  auto slice = Kokkos::make_pair(offsets[comm_rank], offsets[comm_rank + 1]);
  if constexpr (Kokkos::is_view_v<Primitives>)
  {
    Kokkos::deep_copy(exec_space, Kokkos::subview(data, slice), primitives);
  }
  else
  {
    Kokkos::parallel_for(
        "ArborX::DistributedDBSCAN::fill_data",
        Kokkos::RangePolicy(exec_space, 0, n),
        KOKKOS_LAMBDA(int i) { data(rank_offset + i) = points(i); });
  }

#ifdef ARBORX_ENABLE_GPU_AWARE_MPI
  auto send_data = data;
#else
  Kokkos::DefaultHostExecutionSpace host_exec;
  auto send_data = Kokkos::create_mirror_view(
      Kokkos::view_alloc(host_exec, Kokkos::WithoutInitializing), data);
  Kokkos::deep_copy(exec_space, Kokkos::subview(send_data, slice),
                    Kokkos::subview(data, slice));
#endif

  auto const value_size = sizeof(Point);
  std::for_each(counts.begin(), counts.end(),
                [value_size](auto &x) { x *= value_size; });
  std::for_each(offsets.begin(), offsets.end(),
                [value_size](auto &x) { x *= value_size; });
  MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, send_data.data(),
                 counts.data(), offsets.data(), MPI_BYTE, MPI_COMM_WORLD);

#ifndef ARBORX_ENABLE_GPU_AWARE_MPI
  Kokkos::deep_copy(exec_space, data, send_data);
#endif

  auto local_labels = dbscan(exec_space, data, eps, core_min_size, parameters);

  Kokkos::resize(Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing),
                 labels, n);
  Kokkos::parallel_for(
      "ArborX::DistributedDBSCAN::copy_labels",
      Kokkos::RangePolicy(exec_space, 0, n),
      KOKKOS_LAMBDA(int i) { labels(i) = local_labels(rank_offset + i); });
}

} // namespace ArborX::Experimental

#endif
