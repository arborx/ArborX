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

template <typename ExecutionSpace>
class InstanceManager
{
public:
  InstanceManager(int const n_instances) { _instances.resize(n_instances); }
  const std::vector<ExecutionSpace> &get_instances() const
  {
    return _instances;
  }

private:
  std::vector<ExecutionSpace> _instances;
};

#ifdef KOKKOS_ENABLE_CUDA
template <>
class InstanceManager<Kokkos::Cuda>
{
public:
  InstanceManager(int const n_instances)
  {
    _streams.resize(n_instances);
    _instances.reserve(n_instances);
    for (unsigned int i = 0; i < n_instances; ++i)
    {
      cudaStreamCreate(&_streams[i]);
      _instances.emplace_back(_streams[i]);
    }
  }

  ~InstanceManager()
  {
    for (unsigned int i = 0; i < _streams.size(); ++i)
      cudaStreamDestroy(_streams[i]);
  }

  const std::vector<Kokkos::Cuda> &get_instances() const { return _instances; }

private:
  std::vector<Kokkos::Cuda> _instances;
  std::vector<cudaStream_t> _streams;
};
#endif

struct HelpPrinted
{
};

template <typename Queries>
struct QueriesWithIndex
{
  Queries _queries;
};

namespace ArborX
{

template <typename Queries>
struct AccessTraits<QueriesWithIndex<Queries>, ArborX::PredicatesTag>
{
  using memory_space = typename Queries::memory_space;
  static size_t size(QueriesWithIndex<Queries> const &q)
  {
    return q._queries.extent(0);
  }
  static KOKKOS_FUNCTION auto get(QueriesWithIndex<Queries> const &q, size_t i)
  {
    return attach(ArborX::intersects(q._queries(i)), i);
  }
};

} // namespace ArborX

template <typename DeviceType>
struct CountCallback
{
  Kokkos::View<int *, DeviceType> count_;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int) const
  {
    auto const i = ArborX::getData(query);
    Kokkos::atomic_fetch_add(&count_(i), 1);
  }
};

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
  double shift;
  bool separate_trees;

  bpo::options_description desc("Allowed options");
  // clang-format off
    desc.add_options()
        ( "help", "produce help message" )
	( "spaces", bpo::value<int>(&n_spaces)->default_value(1), "Number of execution space instances." )
        ( "separate-trees", bpo::value<bool>(&separate_trees)->default_value(false), "Create separate trees for the execution space instances." )
        ( "values", bpo::value<int>(&n_values)->default_value(20000), "Number of indexable values (source) per execution space instance." )
        ( "queries", bpo::value<int>(&n_queries)->default_value(5000), "Number of queries (target) per execution space instance." )
        ( "shift", bpo::value<double>(&shift)->default_value(2.), "Shift of the point clouds. '0' means the clouds are built "
	                                                          "at the same place, while '1' places the clouds next to each"
								  "other. Negative values and values larger than one "
                                                                  "mean that the clouds are separated." )
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
            << "separate trees                      : " << separate_trees
            << '\n'
            << "#points/execution space instance    : " << n_values << '\n'
            << "#queries/execution space instance   : " << n_queries << '\n'
            << "size of shift                       : " << shift << '\n'
            << '\n';

  InstanceManager<ExecutionSpace> instance_manager(n_spaces);
  const std::vector<ExecutionSpace> &instances =
      instance_manager.get_instances();

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
    // All the point clouds form a box.
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
      random_points_host(i) = {{a * (offset_x + random()),
                                a * (offset_y + random()),
                                a * (offset_z + random())}};
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

  const auto create =
      [](ExecutionSpace const &exec_space,
         Kokkos::View<ArborX::Box *, DeviceType> const &subboxes) {
        Kokkos::Profiling::pushRegion("TestExecutionSpace::tree_construction");
        ArborX::BVH<MemorySpace> tree(exec_space, subboxes);
        Kokkos::Profiling::popRegion();
        return tree;
      };

  const auto query =
      [](ExecutionSpace const &exec_space,
         Kokkos::View<ArborX::Point *, DeviceType> const &subpoints,
         ArborX::BVH<MemorySpace> const &tree,
         Kokkos::View<int *, DeviceType> &n_neighbors_per_query) {
        Kokkos::Profiling::pushRegion("TestExecutionSpace::query");
        CountCallback<DeviceType> callbacks{n_neighbors_per_query};
        QueriesWithIndex<Kokkos::View<ArborX::Point *, DeviceType>> queries{
            subpoints};
        tree.query(
            exec_space, queries, callbacks,
            ArborX::Experimental::TraversalPolicy().setPredicateSorting(false));
        Kokkos::Profiling::popRegion();
      };

  Kokkos::fence();
  Kokkos::Timer total_time;
  total_time.reset();

  Kokkos::Profiling::pushRegion("TestExecutionSpace::separate_instances");
  std::vector<ArborX::BVH<MemorySpace>> trees;
  trees.reserve(separate_trees ? n_spaces : 1);

  std::vector<Kokkos::View<int *, DeviceType>> n_neighbors_per_query(
      n_spaces,
      Kokkos::View<int *, DeviceType>("Testing::num_neigh", n_queries));

  if (separate_trees)
    for (int instance = 0; instance < n_spaces; ++instance)
      trees.push_back(create(
          instances[instance],
          Kokkos::subview(bounding_boxes,
                          Kokkos::pair<int, int>(n_values * instance,
                                                 n_values * (instance + 1)))));
  else
  {
    trees.push_back(create(instances[0], bounding_boxes));
    Kokkos::fence();
  }
  for (int instance = 0; instance < n_spaces; ++instance)
    query(instances[instance],
          Kokkos::subview(random_queries,
                          Kokkos::pair<int, int>(n_queries * instance,
                                                 n_queries * (instance + 1))),
          trees[separate_trees ? instance : 0],
          n_neighbors_per_query[instance]);
  Kokkos::fence();
  Kokkos::Profiling::popRegion();
  std::cout << "Multiple instances running in " << total_time.seconds()
            << " seconds" << std::endl;

  total_time.reset();

  Kokkos::View<int *, DeviceType> combined_n_neighbors_per_query(
      "Testing::num_neigh", n_spaces * n_queries);

  Kokkos::Profiling::pushRegion("run_combined_instance");
  auto tree = create(ExecutionSpace{}, bounding_boxes);
  query(ExecutionSpace{}, random_queries, tree, combined_n_neighbors_per_query);

  Kokkos::fence();
  Kokkos::Profiling::popRegion();
  std::cout << "Combined instance running in " << total_time.seconds()
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
