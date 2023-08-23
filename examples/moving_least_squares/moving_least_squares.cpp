/****************************************************************************
 * Copyright (c) 2023 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

// Example taken from DataTransferKit
// (https://github.com/ORNL-CEES/DataTransferKit)
// with MLS resolution from
// (http://dx.doi.org/10.1016/j.jcp.2015.11.055)
// and
// (A conservative mesh-free approach for fluid-structure interface problems)

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

#include "../../benchmarks/point_clouds/point_clouds.hpp"
#include "DetailsRadialBasisFunctions.hpp"
#include "MovingLeastSquares.hpp"
#include <mpi.h>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

// Function to approximate
KOKKOS_INLINE_FUNCTION float manufactured_solution(ArborX::Point const &p)
{
  return Kokkos::cos(5 * p[2]) * p[0] + p[1] + 1;
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  Kokkos::ScopeGuard guard(argc, argv);

  ExecutionSpace space{};
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  int mpi_size, mpi_rank;
  MPI_Comm_size(mpi_comm, &mpi_size);
  MPI_Comm_rank(mpi_comm, &mpi_rank);

  static constexpr std::size_t total_source_points = 1024 * 512;
  std::size_t local_source_points_num = total_source_points / mpi_size;
  static constexpr std::size_t total_target_points = 1024;
  std::size_t local_target_points_num = total_target_points / mpi_size;
  static constexpr double cube_side = 5;

  Kokkos::View<ArborX::Point *, MemorySpace> source_points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::source_points"),
      local_source_points_num);
  auto source_points_host = Kokkos::create_mirror_view(source_points);
  Kokkos::View<ArborX::Point *, MemorySpace> target_points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::target_points"),
      local_source_points_num);
  auto target_points_host = Kokkos::create_mirror_view(target_points);

  // source and target points are within a 5x5x5 cube
  if (mpi_rank == 0)
  {
    Kokkos::View<ArborX::Point *, Kokkos::HostSpace> all_source_points(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::all_source_points"),
        total_source_points);
    filledBoxCloud(cube_side / 2, all_source_points);
    MPI_Scatter(
        all_source_points.data(), local_source_points_num * 3 * sizeof(float),
        MPI_BYTE, source_points_host.data(),
        local_source_points_num * 3 * sizeof(float), MPI_BYTE, 0, mpi_comm);

    Kokkos::View<ArborX::Point *, Kokkos::HostSpace> all_target_points(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::all_target_points"),
        total_target_points);
    filledBoxCloud(cube_side / 2, all_target_points);
    MPI_Scatter(
        all_target_points.data(), local_target_points_num * 3 * sizeof(float),
        MPI_BYTE, target_points_host.data(),
        local_target_points_num * 3 * sizeof(float), MPI_BYTE, 0, mpi_comm);
  }
  else
  {
    MPI_Scatter(nullptr, local_source_points_num * 3 * sizeof(float), MPI_BYTE,
                source_points_host.data(),
                local_source_points_num * 3 * sizeof(float), MPI_BYTE, 0,
                mpi_comm);

    MPI_Scatter(nullptr, local_target_points_num * 3 * sizeof(float), MPI_BYTE,
                target_points_host.data(),
                local_target_points_num * 3 * sizeof(float), MPI_BYTE, 0,
                mpi_comm);
  }

  Kokkos::deep_copy(space, source_points, source_points_host);
  Kokkos::deep_copy(space, target_points, target_points_host);

  // Create the transform from a point cloud to another
  MovingLeastSquares<MemorySpace, float> mls(mpi_comm, space, source_points,
                                             target_points, Details::degree<2>,
                                             Details::wendland<0>);

  // Compute source values
  Kokkos::View<float *, MemorySpace> source_values("Example::source_values",
                                                   local_source_points_num);
  Kokkos::parallel_for(
      "Example::source_evaluation",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, local_source_points_num),
      KOKKOS_LAMBDA(int const i) {
        source_values(i) = manufactured_solution(source_points(i));
      });

  // Compute target values from source ones
  auto target_values = mls.apply(space, source_values);
  auto target_values_host = Kokkos::create_mirror_view(target_values);
  Kokkos::deep_copy(space, target_values_host, target_values);

  // Compute target values via evaluation
  Kokkos::View<float *, MemorySpace> target_values_exact(
      "Example::target_values_exact", local_target_points_num);
  Kokkos::parallel_for(
      "Example::target_evaluation",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, local_target_points_num),
      KOKKOS_LAMBDA(int const i) {
        target_values_exact(i) = manufactured_solution(target_points(i));
      });

  // Compute local error
  static constexpr float epsilon = std::numeric_limits<float>::epsilon();
  using ErrType = typename Kokkos::MaxLoc<float, std::size_t>::value_type;
  ErrType error{0, 0};
  Kokkos::parallel_reduce(
      "Example::error_computation",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, local_target_points_num),
      KOKKOS_LAMBDA(int const i, ErrType &loc_error) {
        float abs_error =
            Kokkos::abs(target_values(i) - target_values_exact(i));
        float abs_value = Kokkos::abs(target_values_exact(i)) +
                          epsilon;

        if (loc_error.val < abs_error / abs_value)
        {
          loc_error.val = abs_error / abs_value;
          loc_error.loc = i;
        }
      },
      Kokkos::MaxLoc<float, std::size_t>(error));

  std::tuple<float, ArborX::Point, float> error_obj{
      error.val, target_points_host(error.loc), target_values_host(error.loc)};

  if (mpi_rank == 0)
  {
    std::vector<decltype(error_obj)> all_error_obj(mpi_size);
    MPI_Gather(&error_obj, sizeof(decltype(error_obj)), MPI_BYTE,
               all_error_obj.data(), sizeof(decltype(error_obj)), MPI_BYTE, 0,
               mpi_comm);

    for (int i = 0; i < mpi_size; i++)
      if (std::get<0>(error_obj) < std::get<0>(all_error_obj[i]))
        error_obj = all_error_obj[i];

    float error = std::get<0>(error_obj), approx = std::get<2>(error_obj);
    auto point = std::get<1>(error_obj);
    std::cout << "Maximum error: " << error << " at point " << point[0] << ", "
              << point[1] << ", " << point[2]
              << "\nTrue value: " << manufactured_solution(point)
              << "\nComputed: " << approx << std::endl;
  }
  else
  {
    MPI_Gather(&error_obj, sizeof(decltype(error_obj)), MPI_BYTE, nullptr,
               sizeof(decltype(error_obj)), MPI_BYTE, 0, mpi_comm);
  }

  MPI_Finalize();
  return 0;
}
