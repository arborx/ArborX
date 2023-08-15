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

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

#include <sstream>

#include "mls.hpp"
#include <mpi.h>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

struct RBFWendland_0
{
  KOKKOS_INLINE_FUNCTION static float apply(float x)
  {
    return (1.f - x) * (1.f - x);
  }
};

struct MVPolynomialBasis_3D
{
  static constexpr std::size_t size = 10;

  KOKKOS_INLINE_FUNCTION static Kokkos::Array<float, size>
  basis(ArborX::Point const &p)
  {
    return {{1.f, p[0], p[1], p[2], p[0] * p[0], p[0] * p[1], p[0] * p[2],
             p[1] * p[1], p[1] * p[2], p[2] * p[2]}};
  }
};

// Function to approximate
KOKKOS_INLINE_FUNCTION float manufactured_solution(ArborX::Point const &p)
{
  return Kokkos::sin(p[0]) * p[2] + p[1];
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  Kokkos::ScopeGuard guard(argc, argv);

  constexpr std::size_t cube_side = 20;
  constexpr std::size_t source_points_num = cube_side * cube_side * cube_side;
  constexpr std::size_t target_points_num = 4;

  ExecutionSpace space{};
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  int mpi_size, mpi_rank;
  MPI_Comm_size(mpi_comm, &mpi_size);
  MPI_Comm_rank(mpi_comm, &mpi_rank);

  std::size_t local_source_points_num = source_points_num / mpi_size;

  Kokkos::View<ArborX::Point *, MemorySpace> source_points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::source_points"),
      local_source_points_num);
  Kokkos::View<ArborX::Point *, MemorySpace> target_points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::target_points"),
      target_points_num);
  auto target_points_host = Kokkos::create_mirror_view(target_points);

  // Generate source points (Organized within a [-10, 10]^3 cube)
  std::size_t thickness = cube_side / mpi_size;
  Kokkos::parallel_for(
      "Example::source_points_init",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(space, {0, 0, 0},
                                             {cube_side, cube_side, thickness}),
      KOKKOS_LAMBDA(int const i, int const j, int const k) {
        source_points(i * cube_side * thickness + j * thickness +
                      k) = ArborX::Point{
            20.f * (float(i) / (cube_side - 1) - .5f),
            20.f * (float(j) / (cube_side - 1) - .5f),
            20.f * (float(k + thickness * mpi_rank) / (cube_side - 1) - .5f)};
      });

  // Generate target points
  target_points_host(0) = ArborX::Point{1.f, 0.f, 1.f};
  target_points_host(1) = ArborX::Point{5.f, 5.f, 5.f};
  target_points_host(2) = ArborX::Point{-5.f, 5.f, 3.f};
  target_points_host(3) = ArborX::Point{1.f, -3.3f, 7.f};
  Kokkos::deep_copy(space, target_points, target_points_host);

  // Create the transform from a point cloud to another
  MLS<float, MVPolynomialBasis_3D, RBFWendland_0, ExecutionSpace, MemorySpace>
      mls(space, mpi_comm, source_points, target_points);

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

  // Compute target values via evaluation
  Kokkos::View<float *, MemorySpace> target_values_exact(
      "Example::target_values_exact", target_points_num);
  Kokkos::parallel_for(
      "Example::target_evaluation",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, target_points_num),
      KOKKOS_LAMBDA(int const i) {
        target_values_exact(i) = manufactured_solution(target_points(i));
      });

  // Show difference
  auto target_values_host = Kokkos::create_mirror_view(target_values);
  Kokkos::deep_copy(space, target_values_host, target_values);
  auto target_values_exact_host =
      Kokkos::create_mirror_view(target_values_exact);
  Kokkos::deep_copy(space, target_values_exact_host, target_values_exact);

  std::stringstream ss{};
  float error = 0.f;
  for (int i = 0; i < target_points_num; i++)
  {
    error = Kokkos::max(
        Kokkos::abs(target_values_host(i) - target_values_exact_host(i)) /
            Kokkos::abs(target_values_exact_host(i)),
        error);

    ss << mpi_rank << ": ==== Target " << i << '\n'
       << mpi_rank << ": Interpolation: " << target_values_host(i) << '\n'
       << mpi_rank << ": Real value   : " << target_values_exact_host(i)
       << '\n';
  }
  ss << mpi_rank << ": Maximum relative error: " << error << std::endl;

  std::cout << ss.str();

  MPI_Finalize();
  return 0;
}
