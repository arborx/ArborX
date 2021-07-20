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
struct IntersectionSearches
{
  Kokkos::View<ArborX::Point *, DeviceType> points;
};

namespace ArborX
{
template <typename DeviceType>
struct AccessTraits<IntersectionSearches<DeviceType>, ArborX::PredicatesTag>
{
  using memory_space = typename DeviceType::memory_space;
  static KOKKOS_FUNCTION std::size_t
  size(IntersectionSearches<DeviceType> const &pred)
  {
    return pred.points.extent(0);
  }
  static KOKKOS_FUNCTION auto get(IntersectionSearches<DeviceType> const &pred,
                                  std::size_t i)
  {
    return ArborX::intersects(pred.points(i));
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

  bpo::options_description desc("Allowed options");
  // clang-format off
    desc.add_options()
        ( "help", "produce help message" )
	( "spaces", bpo::value<int>(&n_spaces)->default_value(1), "Number of execution space instances." )
        ( "values", bpo::value<int>(&n_values)->default_value(20000), "Number of indexable values (source) per execution space instance." )
        ( "queries", bpo::value<int>(&n_queries)->default_value(5000), "Number of queries (target) per execution space instance." )
        ( "neighbors", bpo::value<int>(&n_neighbors)->default_value(10), "Desired number of results per query." )
        ( "shift", bpo::value<double>(&shift)->default_value(2.), "Shift of the point clouds. '0' means the clouds are built "
	                                                          "at the same place, while '1' places the clouds next to each"
								  "other. Negative values and values larger than one "
                                                                  "mean that the clouds are separated." )
        ( "partition_dim", bpo::value<int>(&partition_dim)->default_value(3), "Number of dimension used by the partitioning of the global "
                                                                              "point cloud. 1 -> local clouds are aligned on a line, 2 -> "
                                                                              "local clouds form a board, 3 -> local clouds form a box." )
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

  std::cout << std::boolalpha;
  std::cout << "\nRunning with arguments:\n"
            << "number of execution space instances : " << n_spaces << '\n'
            << "#points/execution space instance    : " << n_values << '\n'
            << "#queries/execution space instance   : " << n_queries << '\n'
            << "size of shift                       : " << shift << '\n'
            << "dimension                           : " << partition_dim << '\n'
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

    // Random points are "reused" between building the tree and performing
    // queries.
    Kokkos::deep_copy(
        Kokkos::subview(random_queries,
                        Kokkos::pair<int, int>(instance * n_queries,
                                               (instance + 1) * n_queries)),
        Kokkos::subview(random_points, Kokkos::pair<int, int>(0, n_queries)));
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
        bounding_boxes(i) = {{{x - .1, y - .1, z - .1}},
                             {{x + .1, y + .1, z + .1}}};
      });

  const auto create_and_query =
      [n_neighbors, partition_dim](
          ExecutionSpace const &exec_space,
          Kokkos::View<ArborX::Box *, DeviceType> const &subboxes,
          Kokkos::View<ArborX::Point *, DeviceType> const &subqueries, 
	  std::vector<int> &output_offsets,
	  std::vector<int> &output_values) {
        ArborX::BVH<MemorySpace> tree(exec_space, subboxes);

        Kokkos::View<int *, DeviceType> offsets("Testing::offsets", 0);
        Kokkos::View<int *, DeviceType> values("Testing::values", 0);

        tree.query(exec_space, IntersectionSearches<DeviceType>{subqueries},
                   values, offsets);

#ifndef NDEBUG
	{
          output_offsets.resize(offsets.size());
          Kokkos::deep_copy(exec_space, 
                            Kokkos::View<int*, Kokkos::HostSpace>(
                              output_offsets.data(), offsets.size()),
			    offsets);
	}
        {
          output_values.resize(values.size());
          Kokkos::deep_copy(exec_space,
                            Kokkos::View<int*, Kokkos::HostSpace>(
                              output_values.data(), values.size()),
                            values);
        }
#endif
      };

  Kokkos::fence();
  Kokkos::Timer total_time;
  total_time.reset();

  std::vector<std::vector<int>> all_offsets_individual(n_spaces);
  std::vector<std::vector<int>> all_values_individual(n_spaces);

  for (int instance = 0; instance < n_spaces; ++instance)
  {
    create_and_query(
        instances[instance],
        Kokkos::subview(bounding_boxes,
                        Kokkos::pair<int, int>(n_values * instance,
                                               n_values * (instance + 1))),
        Kokkos::subview(random_queries,
                        Kokkos::pair<int, int>(n_queries * instance,
                                               n_queries * (instance + 1))),
	all_offsets_individual[instance],
	all_values_individual[instance]
	);
  }

  Kokkos::fence();
  std::cout << "Multiple instances running in " << total_time.seconds()
            << " seconds" << std::endl;

#ifndef NDEBUG
  std::vector<int> compare_offsets_individual;  
  {
    int combined_size = 0;
    for (auto const &vec: all_offsets_individual)
    {
      for (unsigned int i=0; i+1<vec.size(); ++i)
	compare_offsets_individual.push_back(combined_size+vec[i]);
      combined_size += vec.size()-1;
    }
    compare_offsets_individual.push_back(combined_size);
  }
  std::vector<int> compare_values_individual;
  {
    for (unsigned int j=0; j<all_values_individual.size(); ++j)
      for (auto const el: all_values_individual[j])
	compare_values_individual.push_back(n_values*j+el);
  }
#endif

  total_time.reset();

  std::vector<int> all_offsets_combined;
  std::vector<int> all_values_combined;

  create_and_query(ExecutionSpace{}, bounding_boxes, random_queries, all_offsets_combined, all_values_combined);

  Kokkos::fence();
  std::cout << "Single instance running in " << total_time.seconds()
            << " seconds" << std::endl;
#ifndef NDEBUG
  std::cout << "Checking results...";
  assert(compare_offsets_individual.size() == all_offsets_combined.size());
  for (unsigned int i=0; i < all_offsets_combined.size(); ++i)
  {
    assert(all_offsets_combined[i]==compare_offsets_individual[i]);
  }
  assert(compare_values_individual.size() == all_values_combined.size());
  for (unsigned int i=0; i < all_values_combined.size(); ++i)
  {
    assert(all_values_combined[i]==compare_values_individual[i]);
  }
  std::cout << "done" << std::endl;
#endif

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
