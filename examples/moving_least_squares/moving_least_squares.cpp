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

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>
#include <tuple>
#include <vector>

#include "../../benchmarks/point_clouds/point_clouds.hpp"
#include "DetailsRadialBasisFunctions.hpp"
#include "MovingLeastSquares.hpp"
#include <mpi.h>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

using HostExecutionSpace = Kokkos::DefaultHostExecutionSpace;
using HostMemorySpace = HostExecutionSpace::memory_space;

// Function to approximate
struct Step
{
  KOKKOS_INLINE_FUNCTION static float eval(ArborX::Point const &p)
  {
    return !Kokkos::signbit(p[0]) * 1.f;
  }

  template <class... Properties>
  static Kokkos::View<float *, Properties...>
  map(ExecutionSpace const &space,
      Kokkos::View<ArborX::Point *, Properties...> const &ps)
  {
    Kokkos::View<float *, Properties...> evals("Example::evals", ps.extent(0));
    Kokkos::parallel_for(
        "Example::evaluation",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, ps.extent(0)),
        KOKKOS_LAMBDA(int const i) { evals(i) = eval(ps(i)); });
    return evals;
  }
};

Kokkos::Array<Kokkos::View<ArborX::Point *, MemorySpace>, 2>
createPointClouds(HostExecutionSpace const &hspace, ExecutionSpace const &space,
                  MPI_Comm comm, std::size_t points_num)
{
  int mpi_size, mpi_rank;
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &mpi_rank);

  Kokkos::Array<Kokkos::View<ArborX::Point *, HostMemorySpace>, 2>
      point_clouds_host{Kokkos::View<ArborX::Point *, HostMemorySpace>(
                            Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                               "Example::points_cloud_0"),
                            points_num),
                        Kokkos::View<ArborX::Point *, HostMemorySpace>(
                            Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                               "Example::points_cloud_1"),
                            points_num)};

  if (mpi_rank == 0)
  {
    Kokkos::Array<Kokkos::View<ArborX::Point *, HostMemorySpace>, 2>
        all_point_clouds{Kokkos::View<ArborX::Point *, HostMemorySpace>(
                             Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                "Example::all_points_cloud_0"),
                             points_num * mpi_size),
                         Kokkos::View<ArborX::Point *, HostMemorySpace>(
                             Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                "Example::all_points_cloud_1"),
                             points_num * mpi_size)};

    filledBoxCloud(.5, all_point_clouds[0]);
    filledBoxCloud(.5, all_point_clouds[1]);

    MPI_Scatter(all_point_clouds[0].data(), points_num * 3 * sizeof(float),
                MPI_BYTE, point_clouds_host[0].data(),
                points_num * 3 * sizeof(float), MPI_BYTE, 0, comm);
    MPI_Scatter(all_point_clouds[1].data(), points_num * 3 * sizeof(float),
                MPI_BYTE, point_clouds_host[1].data(),
                points_num * 3 * sizeof(float), MPI_BYTE, 0, comm);
  }
  else
  {
    MPI_Scatter(nullptr, 0, MPI_BYTE, point_clouds_host[0].data(),
                points_num * 3 * sizeof(float), MPI_BYTE, 0, comm);
    MPI_Scatter(nullptr, 0, MPI_BYTE, point_clouds_host[1].data(),
                points_num * 3 * sizeof(float), MPI_BYTE, 0, comm);
  }

  Kokkos::parallel_for(
      "Example::flatten_points",
      Kokkos::RangePolicy<HostExecutionSpace>(hspace, 0, points_num),
      KOKKOS_LAMBDA(int const i) {
        point_clouds_host[0](i)[2] = 0;
        point_clouds_host[1](i)[2] = 0;
      });

  Kokkos::Array<Kokkos::View<ArborX::Point *, MemorySpace>, 2> point_clouds{
      Kokkos::View<ArborX::Point *, MemorySpace>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "Example::points_cloud_0"),
          points_num),
      Kokkos::View<ArborX::Point *, MemorySpace>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "Example::points_cloud_1"),
          points_num)};
  Kokkos::deep_copy(space, point_clouds[0], point_clouds_host[0]);
  Kokkos::deep_copy(space, point_clouds[1], point_clouds_host[1]);

  return point_clouds;
}

template <typename Deg, typename RBF>
Kokkos::Array<MovingLeastSquares<MemorySpace, float>, 2> createMLSObjects(
    MPI_Comm comm, ExecutionSpace const &space,
    Kokkos::View<ArborX::Point *, MemorySpace> const &point_clouds_0,
    Kokkos::View<ArborX::Point *, MemorySpace> const &point_clouds_1,
    Deg const &deg, RBF const &rbf)
{
  return {MovingLeastSquares<MemorySpace, float>(comm, space, point_clouds_0,
                                                 point_clouds_1, deg, rbf),
          MovingLeastSquares<MemorySpace, float>(comm, space, point_clouds_1,
                                                 point_clouds_0, deg, rbf)};
}

void doError(MPI_Comm comm, ExecutionSpace const &space,
             Kokkos::View<ArborX::Point *, MemorySpace> const &points,
             Kokkos::View<float *, MemorySpace> const &approx,
             Kokkos::View<float *, MemorySpace> const &values)
{
  int mpi_size, mpi_rank;
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &mpi_rank);

  // Compute local error
  using ErrType = typename Kokkos::MaxLoc<float, std::size_t>::value_type;
  ErrType error{0, 0};
  float error_sum = 0;
  Kokkos::parallel_reduce(
      "Example::error_computation",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, approx.extent(0)),
      KOKKOS_LAMBDA(int const i, ErrType &loc_error, float &loc_error_sum) {
        float abs_error = Kokkos::abs(approx(i) - values(i));

        loc_error_sum += abs_error;
        if (loc_error.val < abs_error)
        {
          loc_error.val = abs_error;
          loc_error.loc = i;
        }
      },
      Kokkos::MaxLoc<float, std::size_t>(error), Kokkos::Sum<float>(error_sum));

  auto approx_host = Kokkos::create_mirror_view(approx);
  auto points_host = Kokkos::create_mirror_view(points);
  Kokkos::deep_copy(space, approx_host, approx);
  Kokkos::deep_copy(space, points_host, points);

  std::tuple<float, ArborX::Point, float> error_obj{
      error.val, points_host(error.loc), approx_host(error.loc)};

  // Compute global error
  if (mpi_rank == 0)
  {
    float error_sum_global;
    std::vector<decltype(error_obj)> all_error_obj(mpi_size);
    MPI_Gather(&error_obj, sizeof(decltype(error_obj)), MPI_BYTE,
               all_error_obj.data(), sizeof(decltype(error_obj)), MPI_BYTE, 0,
               comm);
    MPI_Reduce(&error_sum, &error_sum_global, 1, MPI_FLOAT, MPI_SUM, 0, comm);

    for (int i = 0; i < mpi_size; i++)
      if (std::get<0>(error_obj) < std::get<0>(all_error_obj[i]))
        error_obj = all_error_obj[i];

    float error = std::get<0>(error_obj), approx = std::get<2>(error_obj);
    auto point = std::get<1>(error_obj);
    std::cout << "Mean error: "
              << error_sum_global / (points.extent(0) * mpi_size)
              << "\nMaximum error: " << error << " at point " << point[0]
              << ", " << point[1] << "\n  True value:  " << Step::eval(point)
              << "\n  Computed:    " << approx << std::endl;
  }
  else
  {
    MPI_Gather(&error_obj, sizeof(decltype(error_obj)), MPI_BYTE, nullptr,
               sizeof(decltype(error_obj)), MPI_BYTE, 0, comm);
    MPI_Reduce(&error_sum, nullptr, 1, MPI_FLOAT, MPI_SUM, 0, comm);
  }
}

Kokkos::View<float *, MemorySpace>
doOne(MPI_Comm comm, ExecutionSpace const &space,
      Kokkos::View<ArborX::Point *, MemorySpace> const &tgt,
      Kokkos::View<float *, MemorySpace> const &values,
      Kokkos::View<float *, MemorySpace> const &true_values,
      MovingLeastSquares<MemorySpace, float> &mls)
{
  auto tgt_values = mls.apply(space, values);
  doError(comm, space, tgt, tgt_values, true_values);
  return tgt_values;
}

int main(int argc, char *argv[])
{
  static constexpr std::size_t total_points = 1024 * 128;
  static constexpr std::size_t num_back_forth = 50;
  static constexpr auto deg = Details::degree<4>;
  static constexpr auto rbf = Details::wu<2>;

  MPI_Init(&argc, &argv);
  Kokkos::ScopeGuard guard(argc, argv);

  ExecutionSpace space{};
  HostExecutionSpace host_space{};
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  int mpi_size, mpi_rank;
  MPI_Comm_size(mpi_comm, &mpi_size);
  MPI_Comm_rank(mpi_comm, &mpi_rank);

  auto point_clouds =
      createPointClouds(host_space, space, mpi_comm, total_points / mpi_size);

  // Create the transform from a point cloud to another
  auto mlss = createMLSObjects(mpi_comm, space, point_clouds[0],
                               point_clouds[1], deg, rbf);

  Kokkos::Array<Kokkos::View<float *, MemorySpace>, 2> true_values{
      Step::map(space, point_clouds[0]), Step::map(space, point_clouds[1])};

  Kokkos::View<float *, MemorySpace> source_values = true_values[0];
  for (int i = 0; i < num_back_forth * 2; i++)
  {
    if (mpi_rank == 0)
      std::cout << "=== TURN " << i + 1 << std::endl;

    Kokkos::View<ArborX::Point *, MemorySpace> target =
        point_clouds[1 - (i % 2)];
    Kokkos::View<float *, MemorySpace> tgt_true_values =
        true_values[1 - (i % 2)];
    MovingLeastSquares<MemorySpace, float> &mls = mlss[i % 2];

    source_values =
        doOne(mpi_comm, space, target, source_values, tgt_true_values, mls);

    if (mpi_rank == 0)
      std::cout << "===\n" << std::endl;
  }

  MPI_Finalize();
  return 0;
}
