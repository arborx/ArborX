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

#include <ArborX_LinearBVH.hpp>
#include <ArborX_Sphere.hpp>
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
  std::vector<ExecutionSpace> const &get_instances() const
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
    for (auto &stream : _streams)
    {
      cudaStreamCreate(&stream);
      _instances.emplace_back(stream);
    }
  }

  ~InstanceManager()
  {
    for (auto &stream : _streams)
      cudaStreamDestroy(stream);
  }

  std::vector<Kokkos::Cuda> const &get_instances() const { return _instances; }

private:
  std::vector<Kokkos::Cuda> _instances;
  std::vector<cudaStream_t> _streams;
};
#endif

template <typename MemorySpace>
struct CountCallback
{
  Kokkos::View<int *, MemorySpace> _counts;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int) const
  {
    auto const i = ArborX::getData(query);
    Kokkos::atomic_increment(&_counts(i));
  }
};

int main(int argc, char *argv[])
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  Kokkos::ScopeGuard guard(argc, argv);

  std::cout << "ArborX version: " << ArborX::version() << std::endl;
  std::cout << "ArborX hash   : " << ArborX::gitCommitHash() << std::endl;

  namespace bpo = boost::program_options;

  int num_exec_spaces;
  int num_primitives;
  int num_problems;
  int num_predicates;

  bpo::options_description desc("Allowed options");
  // clang-format off
    desc.add_options()
        ( "help", "produce help message" )
        ( "num-spaces", bpo::value<int>(&num_exec_spaces)->default_value(1), "Number of execution space instances." )
        ( "num-problems", bpo::value<int>(&num_problems)->default_value(1), "Number of subproblems." )
        ( "values", bpo::value<int>(&num_primitives)->default_value(20000), "Number of indexable values (source) per subproblem." )
        ( "queries", bpo::value<int>(&num_predicates)->default_value(5000), "Number of queries (target) per subproblem." )
        ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);
  bpo::notify(vm);

  float const r = 0.1f;
  float const shift = 2.f;

  if (vm.count("help") > 0)
  {
    std::cout << desc << '\n';
    return 1;
  }

  std::cout << std::boolalpha;
  std::cout << "\nRunning with arguments:"
            << "\nnumber of execution space instances : " << num_exec_spaces
            << "\nnumber of problems                  : " << num_problems
            << "\n#points/problem                     : " << num_primitives
            << "\n#queries/problem                    : " << num_predicates
            << '\n';

  // Generate random points uniformly distributed within a box.
  std::uniform_real_distribution<float> distribution(-1., 1.);
  std::default_random_engine generator;
  auto random = [&distribution, &generator]() {
    return distribution(generator);
  };

  Kokkos::View<ArborX::Point *, MemorySpace> primitives(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "primitives"),
      num_primitives * num_problems);
  Kokkos::View<decltype(ArborX::attach(ArborX::intersects(ArborX::Sphere{}),
                                       int{})) *,
               MemorySpace>
      predicates(Kokkos::view_alloc(Kokkos::WithoutInitializing, "predicates"),
                 num_predicates * num_problems);
  for (int p = 0; p < num_problems; ++p)
  {
    // Points are placed in a box [offset_x-1, offset_x+1] x [-1, 1] x [-1, 1]
    float offset_x = p * shift;

    Kokkos::View<ArborX::Point *, MemorySpace> points(
        "points", std::max(num_primitives, num_predicates));
    auto points_host = Kokkos::create_mirror_view(points);
    for (int i = 0; i < (int)points.extent(0); ++i)
      points_host(i) = {offset_x + random(), random(), random()};
    Kokkos::deep_copy(points, points_host);

    Kokkos::deep_copy(
        Kokkos::subview(
            primitives,
            Kokkos::make_pair(p * num_primitives, (p + 1) * num_primitives)),
        Kokkos::subview(points, Kokkos::make_pair(0, num_primitives)));
    Kokkos::parallel_for(
        "construct_predicates",
        Kokkos::RangePolicy<ExecutionSpace>(
            ExecutionSpace{}, p * num_predicates, (p + 1) * num_predicates),
        KOKKOS_LAMBDA(int i) {
          predicates(i) =
              ArborX::attach(ArborX::intersects(ArborX::Sphere{
                                 points(i - p * num_predicates), r}),
                             i);
        });
  }

  InstanceManager<ExecutionSpace> instance_manager(num_exec_spaces);
  auto const &instances = instance_manager.get_instances();

  std::vector<ArborX::BVH<MemorySpace>> trees;
  for (int p = 0; p < num_problems; ++p)
  {
    auto const &exec_space = instances[p % num_exec_spaces];

    trees.emplace_back(
        exec_space, Kokkos::subview(primitives, Kokkos::pair<int, int>(
                                                    p * num_primitives,
                                                    (p + 1) * num_primitives)));
  }
  ArborX::BVH<MemorySpace> tree(instances[0], primitives);

  Kokkos::View<int *, MemorySpace> counts("counts",
                                          num_predicates * num_problems);

  Kokkos::fence();
  Kokkos::Timer query_time;
  query_time.reset();
  for (int p = 0; p < num_problems; ++p)
  {
    auto const &exec_space = instances[p % num_exec_spaces];

    trees[p].query(
        exec_space,
        Kokkos::subview(predicates,
                        Kokkos::pair<int, int>(p * num_predicates,
                                               (p + 1) * num_predicates)),
        CountCallback<MemorySpace>{counts},
        ArborX::Experimental::TraversalPolicy().setPredicateSorting(false));
    // ArborX::Experimental::TraversalPolicy().setPredicateSorting(true));
  }
  Kokkos::fence();
  std::cout << "Time multiple(s): " << query_time.seconds() << '\n';

  Kokkos::deep_copy(counts, 0);
  query_time.reset();

  tree.query(
      instances[0], predicates, CountCallback<MemorySpace>{counts},
      ArborX::Experimental::TraversalPolicy().setPredicateSorting(false));

  Kokkos::fence();
  std::cout << "Time single(s): " << query_time.seconds() << '\n';

  return EXIT_SUCCESS;
}
