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

#include <ArborXBenchmark_TimeMonitor.hpp>
#include <ArborX_DistributedTree.hpp>
#include <ArborX_Triangle.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <fstream>
#include <iostream>
#include <utility> // pair
#include <vector>

#include <mpi.h>

constexpr int DIM = 3;

template <typename Coordinate>
auto readTriangles(std::string const &filename)
{
  std::ifstream input(filename);
  ARBORX_ASSERT(input.good());

  int num_points = 0;
  int num_triangles = 0;
  input >> num_points;
  input >> num_triangles;

  // Read in points
  std::vector<ArborX::Point<DIM, Coordinate>> points;
  points.resize(num_points);
  for (int i = 0; i < num_points; ++i)
    for (int d = 0; d < DIM; ++d)
      input >> points[i][d];

  // Read in triangles
  std::vector<std::array<int, DIM>> triangles;
  triangles.resize(num_triangles);
  {
    auto it = std::istream_iterator<int>(input);
    for (int i = 0; i < num_triangles; ++i)
      triangles[i] = {*it++ - 1, *it++ - 1, *it++ - 1};
  }

  input.close();

  return std::make_pair(points, triangles);
}

template <typename Coordinate>
auto buildMesh(MPI_Comm comm,
               std::vector<ArborX::Point<3, Coordinate>> const &points,
               std::vector<std::array<int, 3>> const &triangles)
{
  using Triangle = ArborX::Triangle<DIM, Coordinate>;

  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  int const num_triangles = triangles.size();

  Coordinate const d = 0.8;
  int rows_per_dim = 40;

  int N = std::cbrt(comm_size);
  int Nx = comm_rank / (N * N);
  int Ny = (comm_rank - Nx * (N * N)) / N;
  int Nz = comm_rank - Nx * (N * N) - Ny * N;

  auto rank_offset_x = Nx * rows_per_dim * d;
  auto rank_offset_y = Ny * rows_per_dim * d;
  auto rank_offset_z = Nz * rows_per_dim * d;

  std::vector<Triangle> primitives;
  std::vector<Triangle> predicates;

  auto shift_point = [](std::array<Coordinate, DIM> const &offset,
                        ArborX::Point<DIM, Coordinate> const &point) {
    return ArborX::Point<DIM, Coordinate>{
        point[0] + offset[0], point[1] + offset[1], point[2] + offset[2]};
  };

  primitives.reserve(3 * rows_per_dim * rows_per_dim);
  predicates.reserve(3 * rows_per_dim * rows_per_dim);
  for (int i = 0; i < rows_per_dim; ++i)
    for (int j = 0; j < rows_per_dim; ++j)
      for (int k = 0; k < rows_per_dim; ++k)
      {
        if (i > 0 && i < rows_per_dim - 1 && j > 0 && j < rows_per_dim - 1 &&
            k > 0 && k < rows_per_dim - 1)
          continue; // Skip interior points

        std::array<Coordinate, DIM> offset = {rank_offset_x + i * d,
                                              rank_offset_y + j * d,
                                              rank_offset_z + k * d};

        if ((i + j + k) % 2 == 0)
          for (int t = 0; t < num_triangles; ++t)
            predicates.emplace_back(
                Triangle{shift_point(offset, points[triangles[t][0]]),
                         shift_point(offset, points[triangles[t][1]]),
                         shift_point(offset, points[triangles[t][2]])});
        else
          for (int t = 0; t < num_triangles; ++t)
            primitives.emplace_back(
                Triangle{shift_point(offset, points[triangles[t][0]]),
                         shift_point(offset, points[triangles[t][1]]),
                         shift_point(offset, points[triangles[t][2]])});
      }

  return std::make_pair(primitives, predicates);
}

// Callback to store the result indices
struct ExtractIndex
{
  template <typename Query, typename Value, typename Output>
  KOKKOS_FUNCTION void operator()(Query const &, Value const &value,
                                  Output const &out) const
  {
    out(value.index);
  }
};

template <typename Coordinate>
int main_(MPI_Comm const comm)
{
  ArborXBenchmark::TimeMonitor time_monitor;

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  using Box = ArborX::Box<DIM, Coordinate>;

  auto [points, triangles] = readTriangles<Coordinate>("unit_sphere.txt");
  auto [primitives_v, predicates_v] = buildMesh(comm, points, triangles);

  ExecutionSpace space;

  Kokkos::View<Box *, MemorySpace> primitives(
      Kokkos::view_alloc("Benchmark::primitives", Kokkos::WithoutInitializing),
      primitives_v.size());
  auto primitives_host = Kokkos::create_mirror_view(primitives);
  for (size_t i = 0; i < primitives_v.size(); ++i)
  {
    Box box;
    ArborX::Details::expand(box, primitives_v[i]);
    primitives_host[i] = box;
  }
  Kokkos::deep_copy(space, primitives, primitives_host);

  Kokkos::View<Box *, MemorySpace> predicates(
      Kokkos::view_alloc("Benchmark::predicates", Kokkos::WithoutInitializing),
      predicates_v.size());
  auto predicates_host = Kokkos::create_mirror_view(predicates);
  for (size_t i = 0; i < predicates_v.size(); ++i)
  {
    Box box;
    ArborX::Details::expand(box, predicates_v[i]);
    predicates_host[i] = box;
  }
  Kokkos::deep_copy(space, predicates, predicates_host);

  auto construction_time = time_monitor.getNewTimer("construction");
  MPI_Barrier(comm);
  construction_time->start();
  ArborX::DistributedTree<MemorySpace, ArborX::PairValueIndex<Box, int>> tree(
      comm, space, ArborX::Experimental::attach_indices<int>(primitives));
  construction_time->stop();

  Kokkos::View<int *, MemorySpace> offsets("Benchmark::offsets", 0);
  Kokkos::View<int *, MemorySpace> indices("Benchmark::indices", 0);
  auto query_time = time_monitor.getNewTimer("query");
  MPI_Barrier(comm);
  query_time->start();
  tree.query(space, ArborX::Experimental::make_intersects(predicates),
             ExtractIndex{}, indices, offsets);
  query_time->stop();

  time_monitor.summarize(comm);

  return 0;
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  MPI_Comm const comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  if (comm_rank == 0)
  {
    std::cout << "ArborX version: " << ArborX::version() << std::endl;
    std::cout << "ArborX hash   : " << ArborX::gitCommitHash() << std::endl;
    std::cout << "Kokkos version: " << ArborX::Details::KokkosExt::version()
              << std::endl;
  }

  // Strip "--help" and "--kokkos-help" from the flags passed to Kokkos if we
  // are not on MPI rank 0 to prevent Kokkos from printing the help message
  // multiple times.
  if (comm_rank != 0)
  {
    auto *help_it = std::find_if(argv, argv + argc, [](std::string const &x) {
      return x == "--help" || x == "--kokkos-help";
    });
    if (help_it != argv + argc)
    {
      std::swap(*help_it, *(argv + argc - 1));
      --argc;
    }
  }
  Kokkos::initialize(argc, argv);

  main_<float>(comm);

  Kokkos::finalize();
  MPI_Finalize();

  return EXIT_SUCCESS;
}
