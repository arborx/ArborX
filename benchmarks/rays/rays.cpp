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

#include "ArborX_Predicates.hpp"
#include <ArborX_DistributedTree.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <fstream>

#include <mpi.h>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

constexpr int n_rays = 307200;

template <typename MemorySpace>
std::pair<Kokkos::View<ArborX::Experimental::Ray[n_rays], MemorySpace>,
          Kokkos::View<double[n_rays], MemorySpace>>
read_data()
{
  std::string filename = "rays_cam-0-0_test_full.csv";
  Kokkos::View<ArborX::Experimental::Ray[n_rays], MemorySpace> rays("rays");
  auto rays_host = Kokkos::create_mirror_view(rays);
  Kokkos::View<double[n_rays], MemorySpace> temperatures("temperatures");
  auto temperatures_host = Kokkos::create_mirror_view(temperatures);

  std::ifstream file;
  file.open(filename);
  std::string line;
  std::getline(file, line);
  // Counter to keep track of the current line
  int line_number = 0;
  while (std::getline(file, line))
  {
    std::size_t pos = 0;
    std::size_t last_pos = 0;
    std::size_t line_length = line.length();
    unsigned int i = 0;
    ArborX::Point point;
    ArborX::Experimental::Vector direction;
    double value = 0.;
    while (last_pos < line_length + 1)
    {
      pos = line.find_first_of(',', last_pos);
      // If no comma was found then we read until the end of the file
      if (pos == std::string::npos)
      {
        pos = line_length;
      }

      if (pos != last_pos)
      {
        char *end = line.data() + pos;
        if (i < 3)
        {
          point[i] = std::strtod(line.data() + last_pos, &end);
        }
        else if (i < 6)
        {
          direction[i - 3] = std::strtod(line.data() + last_pos, &end);
        }
        else
        {
          value = std::strtod(line.data() + last_pos, &end);
        }

        ++i;
      }

      last_pos = pos + 1;
    }

    ArborX::Experimental::Ray ray{point, direction};
    rays_host(line_number) = ray;
    temperatures_host(line_number) = value;
    ++line_number;
  }
  Kokkos::deep_copy(rays, rays_host);
  Kokkos::deep_copy(temperatures, temperatures_host);

  return {rays, temperatures};
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
    std::cout << "Kokkos version: " << KokkosExt::version() << std::endl;
  }

  Kokkos::initialize(argc, argv);

  {
    // Read the experimental data
    auto [tmp_rays, temperatures] = read_data<MemorySpace>();
    // We cannot do a lambda capture of structured binding on clang without
    // C++20
    Kokkos::View<ArborX::Experimental::Ray[n_rays], MemorySpace> rays =
        tmp_rays;

    // Build the queries
    Kokkos::View<ArborX::Nearest<ArborX::Experimental::Ray> *, MemorySpace>
        queries(Kokkos::view_alloc("queries", Kokkos::WithoutInitializing),
                rays.size());
    Kokkos::parallel_for(
        "Ray::Build_queries",
        Kokkos::RangePolicy<ExecutionSpace>(0, rays.extent(0)),
        KOKKOS_LAMBDA(int i) {
          queries(i) = ArborX::nearest<ArborX::Experimental::Ray>(rays(i), 1);
        });

    // Create the mesh
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    constexpr double dim_x = 0.4;
    constexpr double dim_y = 0.4;
    constexpr double dim_z = 0.2;
    // Each processor has one million cells
    constexpr int n_divisions_x = 100;
    constexpr int n_divisions_y = 100;
    constexpr int n_divisions_z = 100;
    float delta_x = dim_x;
    float delta_y = dim_y / comm_size;
    float delta_z = dim_z;
    float offset_y = comm_rank * delta_y;
    Kokkos::View<ArborX::Box[n_divisions_x * n_divisions_y * n_divisions_z],
                 MemorySpace>
        bounding_boxes(
            Kokkos::view_alloc("bounding_boxes", Kokkos::WithoutInitializing));
    Kokkos::parallel_for(
        "build_mesh",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0}, {n_divisions_x, n_divisions_y, n_divisions_z}),
        KOKKOS_LAMBDA(int i, int j, int k) {
          bounding_boxes(i + j * n_divisions_x +
                         k * n_divisions_x * n_divisions_y) =
              ArborX::Box({i * delta_x / n_divisions_x,
                           offset_y + j * delta_y / n_divisions_y,
                           k * delta_z / n_divisions_z},
                          {(i + 1) * delta_x / n_divisions_x,
                           offset_y + (j + 1) * delta_y / n_divisions_y,
                           (k + 1) * delta_z / n_divisions_z});
        });

    // Wait for the processors to be done with the setup phase
    Kokkos::fence();
    MPI_Barrier(comm);

    // Build the distributed tree
    Kokkos::Profiling::pushRegion("Ray::Setup");
    ArborX::DistributedTree<MemorySpace> distributed_tree(
        comm, ExecutionSpace{}, bounding_boxes);
    Kokkos::Profiling::popRegion();

    // Query
    Kokkos::View<int *, MemorySpace> offsets("offsets", 0);
    Kokkos::View<ArborX::PairIndexRank *, MemorySpace> indices_ranks(
        "Testing::values", 0);
    Kokkos::Profiling::pushRegion("Ray::Query");
    distributed_tree.query(ExecutionSpace{}, queries, indices_ranks, offsets);
    Kokkos::Profiling::popRegion();
  }

  Kokkos::finalize();

  MPI_Finalize();

  return 0;
}
