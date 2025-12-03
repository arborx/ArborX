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
#include <ArborX_Point.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <cmath>    // sqrt, cbrt
#include <iostream> // cout
#include <random>
#include <vector>

#include <mpi.h>

namespace bpo = boost::program_options;

struct Parameters
{
  int n_values;
  int n_queries;
  int n_neighbors;
  float shift;
  int partition_dim;
  bool perform_knn_search;
  bool perform_radius_search;
  bool shift_queries;
};

template <typename Point, typename MemorySpace>
void buildProblem(MPI_Comm comm, Kokkos::View<Point *, MemorySpace> &values,
                  Kokkos::View<Point *, MemorySpace> &queries,
                  int partition_dim, bool shift_queries, float shift)
{
  using Coordinate = ArborX::GeometryTraits::coordinate_type_t<Point>;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  int const n_queries = queries.size();
  int const n_values = values.size();

  Coordinate a = 0;
  Coordinate offset_x = 0;
  Coordinate offset_y = 0;
  Coordinate offset_z = 0;
  int i_max = 0;
  // Change the geometry of the problem. In 1D, all the point clouds are
  // aligned on a line. In 2D, the point clouds create a board and in 3D,
  // they create a box.
  switch (partition_dim)
  {
  case 1:
  {
    i_max = comm_size;
    offset_x = 2 * shift * comm_rank;
    a = n_values;
    break;
  }
  case 2:
  {
    i_max = std::ceil(std::sqrt(comm_size));
    int i = comm_rank % i_max;
    int j = comm_rank / i_max;
    offset_x = 2 * shift * i;
    offset_y = 2 * shift * j;
    a = std::sqrt(n_values);
    break;
  }
  case 3:
  {
    i_max = std::ceil(std::cbrt(comm_size));
    int j_max = i_max;
    int i = comm_rank % i_max;
    int j = (comm_rank / i_max) % j_max;
    int k = comm_rank / (i_max * j_max);
    offset_x = 2 * shift * i;
    offset_y = 2 * shift * j;
    offset_z = 2 * shift * k;
    a = std::cbrt(n_values);
    break;
  }
  default:
  {
    throw std::runtime_error("partition_dim should be 1, 2, or 3");
  }
  }

  // Generate random points uniformly distributed within a box.
  std::uniform_real_distribution<Coordinate> distribution(-1, 1);
  std::default_random_engine generator;
  auto random = [&distribution, &generator]() {
    return distribution(generator);
  };

  // The boxes in which the points are placed have side length two, centered
  // around offset_[xyz] and scaled by a.
  Kokkos::View<Point *, MemorySpace> random_points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Benchmark::points"),
      std::max(n_values, n_queries));
  auto random_points_host = Kokkos::create_mirror_view(random_points);
  for (int i = 0; i < random_points.extent_int(0); ++i)
    random_points_host(i) = {{a * (offset_x + random()),
                              a * (offset_y + random()) * (partition_dim > 1),
                              a * (offset_z + random()) * (partition_dim > 2)}};
  Kokkos::deep_copy(random_points, random_points_host);

  Kokkos::deep_copy(
      values,
      Kokkos::subview(random_points, Kokkos::pair<int, int>(0, n_values)));

  if (!shift_queries)
  {
    // By default, random points are "reused" between building the tree and
    // performing queries.
    Kokkos::deep_copy(
        queries,
        Kokkos::subview(random_points, Kokkos::pair<int, int>(0, n_queries)));
  }
  else
  {
    // For the queries, we shrink the global box by a factor three, and
    // move it by a third of the global size towards the global center.
    auto queries_host = Kokkos::create_mirror_view(queries);

    int const max_offset = 2 * shift * i_max;
    for (int i = 0; i < n_queries; ++i)
      queries_host(i) = {{a * ((offset_x + random()) / 3 + max_offset / 3),
                          a * ((offset_y + random()) / 3 + max_offset / 3) *
                              (partition_dim > 1),
                          a * ((offset_z + random()) / 3 + max_offset / 3) *
                              (partition_dim > 2)}};
    Kokkos::deep_copy(queries, queries_host);
  }
}

template <typename Coordinate>
void main_(MPI_Comm comm, Parameters const &params)
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  using Point = ArborX::Point<3, Coordinate>;

  auto const n_values = params.n_values;
  auto const n_queries = params.n_queries;
  auto const n_neighbors = params.n_neighbors;
  auto const shift = params.shift;
  auto const partition_dim = params.partition_dim;
  auto const perform_knn_search = params.perform_knn_search;
  auto const perform_radius_search = params.perform_radius_search;
  auto const shift_queries = params.shift_queries;

  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  Kokkos::View<Point *, MemorySpace> random_values(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Benchmark::values"),
      n_values);
  Kokkos::View<Point *, MemorySpace> random_queries(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Benchmark::queries"),
      n_queries);
  buildProblem(comm, random_values, random_queries, partition_dim,
               shift_queries, shift);

  ArborXBenchmark::TimeMonitor time_monitor;

  auto construction_time = time_monitor.getNewTimer("construction");
  MPI_Barrier(comm);
  construction_time->start();
  ArborX::DistributedTree distributed_tree(comm, ExecutionSpace{},
                                           random_values);
  construction_time->stop();

  std::ostream &os = std::cout;
  if (comm_rank == 0)
    os << "construction done\n";

  if (perform_knn_search)
  {
    Kokkos::View<int *, MemorySpace> offsets("Benchmark::offsets", 0);
    Kokkos::View<Point *, MemorySpace> values("Benchmark::values", 0);

    auto knn_time = time_monitor.getNewTimer("knn");
    MPI_Barrier(comm);
    knn_time->start();
    distributed_tree.query(
        ExecutionSpace{},
        ArborX::Experimental::make_nearest(random_queries, n_neighbors), values,
        offsets);
    knn_time->stop();

    if (comm_rank == 0)
      os << "knn done\n";
  }

  if (perform_radius_search)
  {
    Kokkos::View<int *, MemorySpace> offsets("Testing::offsets", 0);
    Kokkos::View<Point *, MemorySpace> values("Testing::values", 0);

    // Radius is computed so that the number of results per query for a
    // uniformly distributed primitives in a [-a,a]^d box is approximately
    // n_neighbors.
    Coordinate r = 0;
    switch (partition_dim)
    {
    case 1:
      // Derivation: n_values*(2*r)/(2a) = n_neighbors
      r = static_cast<Coordinate>(n_neighbors);
      break;
    case 2:
      // Derivation: n_values*(pi*r^2)/(2a)^2 = n_neighbors
      r = std::sqrt(n_neighbors * 4.f / Kokkos::numbers::pi_v<Coordinate>);
      break;
    case 3:
      // Derivation: n_values*(4/3*pi*r^3)/(2a)^3 = n_neighbors
      r = std::cbrt(n_neighbors * 6.f / Kokkos::numbers::pi_v<Coordinate>);
      break;
    }

    auto radius_time = time_monitor.getNewTimer("radius");
    MPI_Barrier(comm);
    radius_time->start();
    distributed_tree.query(
        ExecutionSpace{},
        ArborX::Experimental::make_intersects(random_queries, r), values,
        offsets);
    radius_time->stop();

    if (comm_rank == 0)
      os << "radius done\n";
  }
  time_monitor.summarize(comm);
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
  // multiply.
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

  Parameters params;
  std::string precision;
  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ( "help", "produce help message" )
      ( "values", bpo::value<int>(&params.n_values)->default_value(20000), "Number of indexable values (source) per MPI rank." )
      ( "queries", bpo::value<int>(&params.n_queries)->default_value(5000), "Number of queries (target) per MPI rank." )
      ( "neighbors", bpo::value<int>(&params.n_neighbors)->default_value(10), "Desired number of results per query." )
      ( "shift", bpo::value<float>(&params.shift)->default_value(1.f), "Shift of the point clouds. '0' means the clouds are built "
                                                                       "at the same place, while '1' places the clouds next to each"
                                                                       "other. Negative values and values larger than one "
                                                                       "mean that the clouds are separated." )
      ( "partition_dim", bpo::value<int>(&params.partition_dim)->default_value(3), "Number of dimension used by the partitioning of the global "
                                                                                   "point cloud. 1 -> local clouds are aligned on a line, 2 -> "
                                                                                   "local clouds form a board, 3 -> local clouds form a box." )
      ( "precision", bpo::value<std::string>(&precision)->default_value("float"), "Precision (float | double)" )
      ( "do-not-perform-knn-search", "skip kNN search" )
      ( "do-not-perform-radius-search", "skip radius search" )
      ( "shift-queries" , "By default, points are reused for the queries. Enabling this option shrinks the local box queries are created "
                          "in to a third of its size and moves it to the center of the global box. The result is a huge imbalance for the "
                          "number of queries that need to be processed by each processor.")
      ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);
  bpo::notify(vm);

  if (vm.count("help") > 0)
  {
    if (comm_rank == 0)
      std::cout << desc << '\n';
    return 0;
  }

  params.perform_knn_search = !vm.count("do-not-perform-knn-search");
  params.perform_radius_search = !vm.count("do-not-perform-radius-search");
  params.shift_queries = !vm.count("shift-queries");

  if (comm_rank == 0)
  {
    std::cout << std::boolalpha;
    std::cout << "\nRunning with arguments:"
              << "\nperform knn search      : " << params.perform_knn_search
              << "\nperform radius search   : " << params.perform_radius_search
              << "\n#points/MPI process     : " << params.n_values
              << "\n#queries/MPI process    : " << params.n_queries
              << "\nsize of shift           : " << params.shift
              << "\ndimension               : " << params.partition_dim
              << "\nshift-queries           : " << params.shift_queries << '\n';
  }

  Kokkos::ScopeGuard guard(argc, argv);

  if (precision == "float")
    main_<float>(comm, params);
  else
    main_<double>(comm, params);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
