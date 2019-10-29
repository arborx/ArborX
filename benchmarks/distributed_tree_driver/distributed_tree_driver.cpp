/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_DistributedSearchTree.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <algorithm>
#include <chrono>
#include <cmath> // cbrt
#include <iomanip>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <mpi.h>

struct HelpPrinted
{
};

// The TimeMonitor class can be used to measure for a series of events, i.e. it
// represents a set of timers of type Timer. It is a poor man's drop-in
// replacement for Teuchos::TimeMonitor
class TimeMonitor
{
  using container_type = std::vector<std::pair<std::string, double>>;
  using entry_reference_type = container_type::reference;
  container_type _data;

public:
  class Timer
  {
    entry_reference_type _entry;
    bool _started;
    std::chrono::high_resolution_clock::time_point _tick;

  public:
    Timer(entry_reference_type ref)
        : _entry{ref}
        , _started{false}
    {
    }
    void start()
    {
      assert(!_started);
      _tick = std::chrono::high_resolution_clock::now();
      _started = true;
    }
    void stop()
    {
      assert(_started);
      std::chrono::duration<double> duration =
          std::chrono::high_resolution_clock::now() - _tick;
      // NOTE I have put much thought into whether we should use the
      // operator+= and keep track of how many times the timer was
      // restarted.  To be honest I have not even looked was the original
      // TimeMonitor behavior is :)
      _entry.second = duration.count();
      _started = false;
    }
  };
  // NOTE Original code had the pointer semantics.  Can change in the future.
  // The smart pointer is a distraction.  The main problem here is that the
  // reference stored by the timer is invalidated if the time monitor gets
  // out of scope.
  std::unique_ptr<Timer> getNewTimer(std::string name)
  {
    // FIXME Consider searching whether there already is an entry with the
    // same name.
    _data.emplace_back(std::move(name), 0.);
    return std::make_unique<Timer>(_data.back());
  }
  void summarize(MPI_Comm comm, std::ostream &os = std::cout)
  {
    int comm_size;
    MPI_Comm_size(comm, &comm_size);
    int comm_rank;
    MPI_Comm_rank(comm, &comm_rank);
    int n_timers = _data.size();

    os << std::left << std::scientific;

    // Initialize with length of "Timer Name"
    std::string const timer_name = "Timer Name";
    std::size_t const max_section_length = std::accumulate(
        _data.begin(), _data.end(), timer_name.size(),
        [](std::size_t current_max, entry_reference_type section) {
          return std::max(current_max, section.first.size());
        });

    if (comm_size == 1)
    {
      std::string const header_without_timer_name = " | GlobalTime";
      std::stringstream dummy_string_stream;
      dummy_string_stream << std::setprecision(os.precision())
                          << std::scientific << " | " << 1.;
      int const header_width =
          max_section_length + std::max<int>(header_without_timer_name.size(),
                                             dummy_string_stream.str().size());

      os << std::string(header_width, '=') << "\n\n";
      os << "TimeMonitor results over 1 processor\n\n";
      os << std::setw(max_section_length) << timer_name
         << header_without_timer_name << '\n';
      os << std::string(header_width, '-') << '\n';
      for (int i = 0; i < n_timers; ++i)
      {
        os << std::setw(max_section_length) << _data[i].first << " | "
           << _data[i].second << '\n';
      }
      os << std::string(header_width, '=') << '\n';
      return;
    }
    std::vector<double> all_entries(comm_size * n_timers);
    std::transform(
        _data.begin(), _data.end(), all_entries.begin() + comm_rank * n_timers,
        [](std::pair<std::string, double> const &x) { return x.second; });
    // FIXME No guarantee that all processors have the same timers!
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_entries.data(),
                  n_timers, MPI_DOUBLE, comm);
    std::string const header_without_timer_name =
        " | MinOverProcs | MeanOverProcs | MaxOverProcs";
    if (comm_rank == 0)
    {
      os << std::string(max_section_length + header_without_timer_name.size(),
                        '=')
         << "\n\n";
      os << "TimeMonitor results over " << comm_size << " processors\n";
      os << std::setw(max_section_length) << timer_name
         << header_without_timer_name << '\n';
      os << std::string(max_section_length + header_without_timer_name.size(),
                        '-')
         << '\n';
    }
    std::vector<double> tmp(comm_size);
    for (int i = 0; i < n_timers; ++i)
    {
      for (int j = 0; j < comm_size; ++j)
      {
        tmp[j] = all_entries[j * n_timers + i];
      }
      auto min = *std::min_element(tmp.begin(), tmp.end());
      auto max = *std::max_element(tmp.begin(), tmp.end());
      auto mean = std::accumulate(tmp.begin(), tmp.end(), 0.) / comm_size;
      if (comm_rank == 0)
      {
        os << std::setw(max_section_length) << _data[i].first << " | " << min
           << " |  " << mean << " | " << max << '\n';
      }
    }
    if (comm_rank == 0)
    {
      os << std::string(max_section_length + header_without_timer_name.size(),
                        '=')
         << '\n';
    }
  }
};

namespace bpo = boost::program_options;

template <class NO>
int main_(std::vector<std::string> const &args, const MPI_Comm comm)
{
  TimeMonitor time_monitor;

  using DeviceType = typename NO::device_type;
  using ExecutionSpace = typename DeviceType::execution_space;

  int n_values;
  int n_queries;
  int n_neighbors;
  double shift;
  int partition_dim;
  bool perform_knn_search = true;
  bool perform_radius_search = true;

  bpo::options_description desc("Allowed options");
  // clang-format off
    desc.add_options()
        ( "help", "produce help message" )
        ( "values", bpo::value<int>(&n_values)->default_value(50000), "Number of indexable values (source) per MPI rank." )
        ( "queries", bpo::value<int>(&n_queries)->default_value(20000), "Number of queries (target) per MPI rank." )
        ( "neighbors", bpo::value<int>(&n_neighbors)->default_value(10), "Desired number of results per query." )
        ( "shift", bpo::value<double>(&shift)->default_value(1.), "Shift of the point clouds. '0' means the clouds are built "
	                                                          "at the same place, while '1' places the clouds next to each"
								  "other. Negative values and values larger than one "
                                                                  "mean that the clouds are separated." )
        ( "partition_dim", bpo::value<int>(&partition_dim)->default_value(3), "Number of dimension used by the partitioning of the global "
                                                                              "point cloud. 1 -> local clouds are aligned on a line, 2 -> "
                                                                              "local clouds form a board, 3 -> local clouds form a box." )
        ( "do-not-perform-knn-search", "skip kNN search" )
        ( "do-not-perform-radius-search", "skip radius search" )
        ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(args).options(desc).run(), vm);
  bpo::notify(vm);

  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  if (vm.count("help"))
  {
    if (comm_rank == 0)
      std::cout << desc << '\n';
    throw HelpPrinted();
  }

  if (vm.count("do-not-perform-knn-search"))
    perform_knn_search = false;
  if (vm.count("do-not-perform-radius-search"))
    perform_radius_search = false;

  if (comm_rank == 0)
  {
    std::cout << std::boolalpha;
    std::cout << "\nRunning with arguments:\n"
              << "perform knn search      : " << perform_knn_search << '\n'
              << "perform radius search   : " << perform_radius_search << '\n'
              << "#points/MPI process     : " << n_values << '\n'
              << "#queries/MPI process    : " << n_queries << '\n'
              << "size of shift           : " << shift << '\n'
              << "dimension               : " << partition_dim << '\n'
              << '\n';
  }

  Kokkos::View<ArborX::Point *, DeviceType> random_points("random_points", 0);
  {
    // Random points are "reused" between building the tree and performing
    // queries. Note that this means that for the points in the middle of
    // the local domains there won't be any communication.
    auto n = std::max(n_values, n_queries);
    Kokkos::resize(random_points, n);

    auto random_points_host = Kokkos::create_mirror_view(random_points);

    // Generate random points uniformly distributed within a box.
    auto const a = std::cbrt(n_values);
    std::uniform_real_distribution<double> distribution(-a, +a);
    std::default_random_engine generator;
    auto random = [&distribution, &generator]() {
      return distribution(generator);
    };

    double offset_x = 0.;
    double offset_y = 0.;
    double offset_z = 0.;
    // Change the geometry of the problem. In 1D, all the point clouds are
    // aligned on a line. In 2D, the point clouds create a board and in 3D,
    // they create a box.
    switch (partition_dim)
    {
    case 1:
    {
      offset_x = 2 * a * shift * comm_rank;

      break;
    }
    case 2:
    {
      int i_max = std::ceil(std::sqrt(comm_size));
      int i = comm_rank % i_max;
      int j = comm_rank / i_max;
      offset_x = 2 * a * shift * i;
      offset_y = 2 * a * shift * j;

      break;
    }
    case 3:
    {
      int i_max = std::ceil(std::cbrt(comm_size));
      int j_max = i_max;
      int i = comm_rank % i_max;
      int j = (comm_rank / i_max) % j_max;
      int k = comm_rank / (i_max * j_max);
      offset_x = 2 * a * shift * i;
      offset_y = 2 * a * shift * j;
      offset_z = 2 * a * shift * k;

      break;
    }
    default:
    {
      throw std::runtime_error("partition_dim should be 1, 2, or 3");
    }
    }

    for (int i = 0; i < n; ++i)
      random_points_host(i) = {
          {offset_x + random(), offset_y + random(), offset_z + random()}};
    Kokkos::deep_copy(random_points, random_points_host);
  }

  Kokkos::View<ArborX::Box *, DeviceType> bounding_boxes(
      Kokkos::ViewAllocateWithoutInitializing("bounding_boxes"), n_values);
  Kokkos::parallel_for("bvh_driver:construct_bounding_boxes",
                       Kokkos::RangePolicy<ExecutionSpace>(0, n_values),
                       KOKKOS_LAMBDA(int i) {
                         double const x = random_points(i)[0];
                         double const y = random_points(i)[1];
                         double const z = random_points(i)[2];
                         bounding_boxes(i) = {{{x - 1., y - 1., z - 1.}},
                                              {{x + 1., y + 1., z + 1.}}};
                       });

  auto construction = time_monitor.getNewTimer("construction");
  MPI_Barrier(comm);
  construction->start();
  ArborX::DistributedSearchTree<DeviceType> distributed_tree(comm,
                                                             bounding_boxes);
  construction->stop();

  std::ostream &os = std::cout;
  if (comm_rank == 0)
    os << "contruction done\n";

  if (perform_knn_search)
  {
    Kokkos::View<ArborX::Nearest<ArborX::Point> *, DeviceType> queries(
        Kokkos::ViewAllocateWithoutInitializing("queries"), n_queries);
    Kokkos::parallel_for("bvh_driver:setup_knn_search_queries",
                         Kokkos::RangePolicy<ExecutionSpace>(0, n_queries),
                         KOKKOS_LAMBDA(int i) {
                           queries(i) = ArborX::nearest<ArborX::Point>(
                               random_points(i), n_neighbors);
                         });

    Kokkos::View<int *, DeviceType> offset("offset", 0);
    Kokkos::View<int *, DeviceType> indices("indices", 0);
    Kokkos::View<int *, DeviceType> ranks("ranks", 0);

    auto knn = time_monitor.getNewTimer("knn");
    MPI_Barrier(comm);
    knn->start();
    distributed_tree.query(queries, indices, offset, ranks);
    knn->stop();

    if (comm_rank == 0)
      os << "knn done\n";
  }

  if (perform_radius_search)
  {
    Kokkos::View<decltype(ArborX::intersects(ArborX::Sphere{})) *, DeviceType>
        queries(Kokkos::ViewAllocateWithoutInitializing("queries"), n_queries);
    // radius chosen in order to control the number of results per query
    // NOTE: minus "1+sqrt(3)/2 \approx 1.37" matches the size of the boxes
    // inserted into the tree (mid-point between half-edge and
    // half-diagonal)
    double const r =
        2. * std::cbrt(static_cast<double>(n_neighbors) * 3. / (4. * M_PI)) -
        (1. + std::sqrt(3.)) / 2.;
    Kokkos::parallel_for(
        "bvh_driver:setup_radius_search_queries",
        Kokkos::RangePolicy<ExecutionSpace>(0, n_queries),
        KOKKOS_LAMBDA(int i) {
          queries(i) = ArborX::intersects(ArborX::Sphere{random_points(i), r});
        });

    Kokkos::View<int *, DeviceType> offset("offset", 0);
    Kokkos::View<int *, DeviceType> indices("indices", 0);
    Kokkos::View<int *, DeviceType> ranks("ranks", 0);

    auto radius = time_monitor.getNewTimer("radius");
    MPI_Barrier(comm);
    radius->start();
    distributed_tree.query(queries, indices, offset, ranks);
    radius->stop();

    if (comm_rank == 0)
      os << "radius done\n";
  }
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
  }

  // Strip "--help" and "--kokkos-help" from the flags passed to Kokkos if we
  // are not on MPI rank 0 to prevent Kokkos from printing the help message
  // multiply.
  auto kokkos_argv = argv;
  auto kokkos_argc = argc;
  if (comm_rank != 0)
  {
    auto help_it = std::find_if(kokkos_argv, kokkos_argv + kokkos_argc,
                                [](std::string const &x) {
                                  return x == "--help" || x == "--kokkos-help";
                                });
    if (help_it != kokkos_argv + kokkos_argc)
    {
      std::swap(*help_it, *(kokkos_argv + kokkos_argc - 1));
      --kokkos_argc;
    }
  }
  Kokkos::initialize(kokkos_argc, kokkos_argv);

  bool success = true;

  try
  {
    std::string node;
    // NOTE Lame trick to get a valid default value
#if defined(KOKKOS_ENABLE_CUDA)
    node = "cuda";
#elif defined(KOKKOS_ENABLE_OPENMP)
    node = "openmp";
#elif defined(KOKKOS_ENABLE_SERIAL)
    node = "serial";
#endif
    bpo::options_description desc("Parallel setting:");
    desc.add_options()("node", bpo::value<std::string>(&node),
                       "node type (serial | openmp | cuda)");
    bpo::variables_map vm;
    bpo::parsed_options parsed = bpo::command_line_parser(argc, argv)
                                     .options(desc)
                                     .allow_unregistered()
                                     .run();
    bpo::store(parsed, vm);
    std::vector<std::string> pass_further =
        bpo::collect_unrecognized(parsed.options, bpo::include_positional);

    if (comm_rank == 0 && std::find_if(pass_further.begin(), pass_further.end(),
                                       [](std::string const &x) {
                                         return x == "--help";
                                       }) != pass_further.end())
    {
      std::cout << desc << '\n';
    }

    if (node == "serial")
    {
#ifdef KOKKOS_ENABLE_SERIAL
      using Node = Kokkos::Serial;
      main_<Node>(pass_further, comm);
#else
      throw std::runtime_error("Serial node type is disabled");
#endif
    }
    else if (node == "openmp")
    {
#ifdef KOKKOS_ENABLE_OPENMP
      using Node = Kokkos::OpenMP;
      main_<Node>(pass_further, comm);
#else
      throw std::runtime_error("OpenMP node type is disabled");
#endif
    }
    else if (node == "cuda")
    {
#ifdef KOKKOS_ENABLE_CUDA
      using Node = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>;
      main_<Node>(pass_further, comm);
#else
      throw std::runtime_error("CUDA node type is disabled");
#endif
    }
    else
    {
      throw std::runtime_error("Unrecognized node type");
    }
  }
  catch (HelpPrinted const &)
  {
    // Do nothing, it was a successful run. Just clean up things below.
  }
  catch (std::exception const &e)
  {
    std::cerr << "processor " << comm_rank
              << " caught a std::exception: " << e.what() << '\n';
    success = false;
  }
  catch (...)
  {
    std::cerr << "processor " << comm_rank
              << " caught some kind of exception\n";
    success = false;
  }

  Kokkos::finalize();

  MPI_Finalize();

  return (success ? EXIT_SUCCESS : EXIT_FAILURE);
}
