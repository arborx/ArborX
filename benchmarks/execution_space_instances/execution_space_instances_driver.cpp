/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_LinearBVH.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <algorithm>
#include <chrono>
#include <cmath> // sqrt, cbrt
#include <iomanip>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

struct HelpPrinted
{
};

template <typename DeviceType>
struct NearestNeighborsSearches
{
  Kokkos::View<ArborX::Point *, DeviceType> points;
  int k;
};
template <typename DeviceType>
struct RadiusSearches
{
  Kokkos::View<ArborX::Point *, DeviceType> points;
  double radius;
};

namespace ArborX
{
template <typename DeviceType>
struct AccessTraits<RadiusSearches<DeviceType>, ArborX::PredicatesTag>
{
  using memory_space = typename DeviceType::memory_space;
  static KOKKOS_FUNCTION std::size_t
  size(RadiusSearches<DeviceType> const &pred)
  {
    return pred.points.extent(0);
  }
  static KOKKOS_FUNCTION auto get(RadiusSearches<DeviceType> const &pred,
                                  std::size_t i)
  {
    return ArborX::intersects(ArborX::Sphere{pred.points(i), pred.radius});
  }
};

template <typename DeviceType>
struct AccessTraits<NearestNeighborsSearches<DeviceType>, ArborX::PredicatesTag>
{
  using memory_space = typename DeviceType::memory_space;
  static KOKKOS_FUNCTION std::size_t
  size(NearestNeighborsSearches<DeviceType> const &pred)
  {
    return pred.points.extent(0);
  }
  static KOKKOS_FUNCTION auto
  get(NearestNeighborsSearches<DeviceType> const &pred, std::size_t i)
  {
    return ArborX::nearest(pred.points(i), pred.k);
  }
};
} // namespace ArborX

namespace bpo = boost::program_options;

template <class NO>
int main_(std::vector<std::string> const &args)
{
  using DeviceType = typename NO::device_type;
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;

  int n_spaces;
  int n_values;
  int n_queries;
  int n_neighbors;
  double shift;
  int partition_dim;
  bool perform_knn_search = true;
  bool perform_radius_search = false;
  bool shift_queries = true;

  bpo::options_description desc("Allowed options");
  // clang-format off
    desc.add_options()
        ( "help", "produce help message" )
	( "spaces", bpo::value<int>(&n_spaces)->default_value(1), "Number of execution space instances." )
        ( "values", bpo::value<int>(&n_values)->default_value(20000), "Number of indexable values (source) per execution space instance." )
        ( "queries", bpo::value<int>(&n_queries)->default_value(5000), "Number of queries (target) per execution space instance." )
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
        ( "shift-queries" , "By default, points are reused for the queries. Enabling this option shrinks the local box queries are created "
                            "in to a third of its size and moves it to the center of the global box. The result is a huge imbalance for the "
                            "number of queries that need to be processed by each execution space instance.")
        ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(args).options(desc).run(), vm);
  bpo::notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << '\n';
    throw HelpPrinted();
  }

  if (vm.count("do-not-perform-knn-search"))
    perform_knn_search = false;
  if (vm.count("do-not-perform-radius-search"))
    perform_radius_search = false;
  if (vm.count("shift-queries"))
    shift_queries = true;

  std::cout << std::boolalpha;
  std::cout << "\nRunning with arguments:\n"
            << "number of execution space instances : " << n_spaces << '\n'
            << "perform knn search                  : " << perform_knn_search
            << '\n'
            << "perform radius search               : " << perform_radius_search
            << '\n'
            << "#points/execution space instance    : " << n_values << '\n'
            << "#queries/execution space instance   : " << n_queries << '\n'
            << "size of shift                       : " << shift << '\n'
            << "dimension                           : " << partition_dim << '\n'
            << "shift-queries                       : " << shift_queries << '\n'
            << '\n';

  std::vector<ExecutionSpace> instances(n_spaces);

  Kokkos::View<ArborX::Point *, DeviceType> random_values(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Testing::values"),
      n_values * n_spaces);
  Kokkos::View<ArborX::Point *, DeviceType> random_queries(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Testing::queries"),
      n_queries * n_spaces);
  for (int instance = 0; instance < n_spaces; ++instance)
  {
    double a = 0.;
    double offset_x = 0.;
    double offset_y = 0.;
    double offset_z = 0.;
    int i_max = 0;
    // Change the geometry of the problem. In 1D, all the point clouds are
    // aligned on a line. In 2D, the point clouds create a board and in 3D,
    // they create a box.
    switch (partition_dim)
    {
    case 1:
    {
      i_max = n_spaces;
      offset_x = 2 * shift * instance;
      a = n_values;

      break;
    }
    case 2:
    {
      i_max = std::ceil(std::sqrt(n_spaces));
      int i = instance % i_max;
      int j = instance / i_max;
      offset_x = 2 * shift * i;
      offset_y = 2 * shift * j;
      a = std::sqrt(n_values);

      break;
    }
    case 3:
    {
      i_max = std::ceil(std::cbrt(n_spaces));
      int j_max = i_max;
      int i = instance % i_max;
      int j = (instance / i_max) % j_max;
      int k = instance / (i_max * j_max);
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
    std::uniform_real_distribution<double> distribution(-1., 1.);
    std::default_random_engine generator;
    auto random = [&distribution, &generator]() {
      return distribution(generator);
    };

    // The boxes in which the points are placed have side length two, centered
    // around offset_[xyz] and scaled by a.
    Kokkos::View<ArborX::Point *, DeviceType> random_points(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "Testing::points"),
        std::max(n_values, n_queries));
    auto random_points_host = Kokkos::create_mirror_view(random_points);
    for (int i = 0; i < random_points.extent_int(0); ++i)
      random_points_host(i) = {
          {a * (offset_x + random()),
           a * (offset_y + random()) * (partition_dim > 1),
           a * (offset_z + random()) * (partition_dim > 2)}};
    Kokkos::deep_copy(random_points, random_points_host);

    Kokkos::deep_copy(
        Kokkos::subview(random_values,
                        Kokkos::pair<int, int>(instance * n_values,
                                               (instance + 1) * n_values)),
        Kokkos::subview(random_points, Kokkos::pair<int, int>(0, n_values)));

    if (!shift_queries)
    {
      // By default, random points are "reused" between building the tree and
      // performing queries.
      Kokkos::deep_copy(
          Kokkos::subview(random_queries,
                          Kokkos::pair<int, int>(instance * n_queries,
                                                 (instance + 1) * n_queries)),
          Kokkos::subview(random_points, Kokkos::pair<int, int>(0, n_queries)));
    }
    else
    {
      // For the queries, we shrink the global box by a factor three, and
      // move it by a third of the global size towards the global center.
      auto subview = Kokkos::subview(
          random_queries, Kokkos::pair<int, int>(instance * n_queries,
                                                 (instance + 1) * n_queries));
      auto random_queries_host = Kokkos::create_mirror_view(subview);

      int const max_offset = 2 * shift * i_max;
      for (int i = 0; i < n_queries; ++i)
        random_queries_host(i) = {
            {a * ((offset_x + random()) / 3 + max_offset / 3),
             a * ((offset_y + random()) / 3 + max_offset / 3) *
                 (partition_dim > 1),
             a * ((offset_z + random()) / 3 + max_offset / 3) *
                 (partition_dim > 2)}};
      Kokkos::deep_copy(
          Kokkos::subview(random_queries,
                          Kokkos::pair<int, int>(instance * n_queries,
                                                 (instance + 1) * n_queries)),
          random_queries_host);
    }
  }

  Kokkos::View<ArborX::Box *, DeviceType> bounding_boxes(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "Testing::bounding_boxes"),
      n_values * n_spaces);
  Kokkos::parallel_for(
      "bvh_driver:construct_bounding_boxes",
      Kokkos::RangePolicy<ExecutionSpace>(0, n_values * n_spaces),
      KOKKOS_LAMBDA(int i) {
        double const x = random_values(i)[0];
        double const y = random_values(i)[1];
        double const z = random_values(i)[2];
        bounding_boxes(i) = {{{x - 1., y - 1., z - 1.}},
                             {{x + 1., y + 1., z + 1.}}};
      });

  const auto create_and_query =
      [perform_knn_search, n_neighbors, perform_radius_search, partition_dim](
          ExecutionSpace const &exec_space,
          Kokkos::View<ArborX::Box *, DeviceType> const &subboxes,
          Kokkos::View<ArborX::Point *, DeviceType> const &subqueries) {
        ArborX::BVH<MemorySpace> tree(exec_space, subboxes);

        if (perform_knn_search)
        {
          Kokkos::View<int *, DeviceType> offsets("Testing::offsets", 0);
          Kokkos::View<int *, DeviceType> values("Testing::values", 0);

          tree.query(
              exec_space,
              NearestNeighborsSearches<DeviceType>{subqueries, n_neighbors},
              values, offsets);
        }

        if (perform_radius_search)
        {
          // Radius is computed so that the number of results per query for a
          // uniformly distributed primitives in a [-a,a]^d box is approximately
          // n_neighbors. The primivites are boxes and not points. Thus, the
          // radius we would have chosen for the case of point primitives has to
          // be adjusted to account for box-box interaction. The radius is
          // decreased by an average of the lengths of a half-edge and a
          // half-diagonal to account for that (approximately). An exact
          // calculation would require computing an integral.
          double r = 0.;
          switch (partition_dim)
          {
          case 1:
            // Derivation (first term): n_values*(2*r)/(2a) = n_neighbors
            r = static_cast<double>(n_neighbors) - 1.;
            break;
          case 2:
            // Derivation (first term): n_values*(M_PI*r^2)/(2a)^2 = n_neighbors
            r = std::sqrt(static_cast<double>(n_neighbors) * 4. / M_PI) -
                (1. + std::sqrt(2.)) / 2;
            break;
          case 3:
            // Derivation (first term): n_values*(4/3*M_PI*r^3)/(2a)^3 =
            // n_neighbors
            r = std::cbrt(static_cast<double>(n_neighbors) * 6. / M_PI) -
                (1. + std::cbrt(3.)) / 2;
            break;
          }

          Kokkos::View<int *, DeviceType> offsets("Testing::offsets", 0);
          Kokkos::View<int *, DeviceType> values("Testing::values", 0);

          tree.query(exec_space, RadiusSearches<DeviceType>{subqueries, r},
                     values, offsets);
        }
      };

  Kokkos::fence();
  Kokkos::Timer total_time;
  total_time.reset();

  for (int instance = 0; instance < n_spaces; ++instance)
  {
    create_and_query(
        instances[instance],
        Kokkos::subview(bounding_boxes,
                        Kokkos::pair<int, int>(n_values * instance,
                                               n_values * (instance + 1))),
        Kokkos::subview(random_queries,
                        Kokkos::pair<int, int>(n_queries * instance,
                                               n_queries * (instance + 1))));
  }

  Kokkos::fence();
  std::cout << "Multiple instances running in " << total_time.seconds()
            << " seconds" << std::endl;
  total_time.reset();

  create_and_query(ExecutionSpace{}, bounding_boxes, random_queries);

  Kokkos::fence();
  std::cout << "Single instance running in " << total_time.seconds()
            << " seconds" << std::endl;

  return 0;
}

int main(int argc, char *argv[])
{
  std::cout << "ArborX version: " << ArborX::version() << std::endl;
  std::cout << "ArborX hash   : " << ArborX::gitCommitHash() << std::endl;

  Kokkos::initialize(argc, argv);

  bool success = true;

  try
  {
    std::string node;
    // NOTE Lame trick to get a valid default value
#if defined(KOKKOS_ENABLE_HIP)
    node = "hip";
#elif defined(KOKKOS_ENABLE_CUDA)
    node = "cuda";
#elif defined(KOKKOS_ENABLE_OPENMP)
    node = "openmp";
#elif defined(KOKKOS_ENABLE_THREADS)
    node = "threads";
#elif defined(KOKKOS_ENABLE_SERIAL)
    node = "serial";
#endif
    bpo::options_description desc("Parallel setting:");
    desc.add_options()("node", bpo::value<std::string>(&node),
                       "node type (serial | openmp | threads | cuda)");
    bpo::variables_map vm;
    bpo::parsed_options parsed = bpo::command_line_parser(argc, argv)
                                     .options(desc)
                                     .allow_unregistered()
                                     .run();
    bpo::store(parsed, vm);
    std::vector<std::string> pass_further =
        bpo::collect_unrecognized(parsed.options, bpo::include_positional);
    bpo::notify(vm);

    if (std::find_if(pass_further.begin(), pass_further.end(),
                     [](std::string const &x) { return x == "--help"; }) !=
        pass_further.end())
    {
      std::cout << desc << '\n';
    }

    if (node != "serial" && node != "openmp" && node != "cuda" &&
        node != "threads" && node != "hip")
      throw std::runtime_error("Unrecognized node type: \"" + node + "\"");

    if (node == "serial")
    {
#ifdef KOKKOS_ENABLE_SERIAL
      using Node = Kokkos::Serial;
      main_<Node>(pass_further);
#else
      throw std::runtime_error("Serial node type is disabled");
#endif
    }
    if (node == "openmp")
    {
#ifdef KOKKOS_ENABLE_OPENMP
      using Node = Kokkos::OpenMP;
      main_<Node>(pass_further);
#else
      throw std::runtime_error("OpenMP node type is disabled");
#endif
    }
    if (node == "threads")
    {
#ifdef KOKKOS_ENABLE_THREADS
      using Node = Kokkos::Threads;
      main_<Node>(pass_further);
#else
      throw std::runtime_error("Threads node type is disabled");
#endif
    }
    if (node == "cuda")
    {
#ifdef KOKKOS_ENABLE_CUDA
      using Node = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>;
      main_<Node>(pass_further);
#else
      throw std::runtime_error("CUDA node type is disabled");
#endif
    }
    if (node == "hip")
    {
#ifdef KOKKOS_ENABLE_HIP
      using Node = Kokkos::Device<Kokkos::Experimental::HIP,
                                  Kokkos::Experimental::HIPSpace>;
      main_<Node>(pass_further);
#else
      throw std::runtime_error("HIP node type is disabled");
#endif
    }
  }
  catch (HelpPrinted const &)
  {
    // Do nothing, it was a successful run. Just clean up things below.
  }
  catch (std::exception const &e)
  {
    std::cerr << "caught a std::exception: " << e.what() << '\n';
    success = false;
  }
  catch (...)
  {
    std::cerr << "caught some kind of exception\n";
    success = false;
  }

  Kokkos::finalize();

  return (success ? EXIT_SUCCESS : EXIT_FAILURE);
}
