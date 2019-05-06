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

#include "../test/ArborX_BoostRTreeHelpers.hpp"
#include <ArborX_LinearBVH.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <chrono>
#include <cmath> // cbrt
#include <cstdlib>
#include <random>

#include <benchmark/benchmark.h>
#include <point_clouds.hpp>

#if defined(KOKKOS_ENABLE_SERIAL)
class BoostRTree
{
public:
  using DeviceType = Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>;
  using device_type = DeviceType;

  BoostRTree(Kokkos::View<ArborX::Point *, DeviceType> points)
  {
    _tree = BoostRTreeHelpers::makeRTree(points);
  }

  template <typename Query>
  void query(Kokkos::View<Query *, DeviceType> queries,
             Kokkos::View<int *, DeviceType> &indices,
             Kokkos::View<int *, DeviceType> &offset)
  {
    std::tie(offset, indices) =
        BoostRTreeHelpers::performQueries(_tree, queries);
  }

  template <typename Query>
  void query(Kokkos::View<Query *, DeviceType> queries,
             Kokkos::View<int *, DeviceType> &indices,
             Kokkos::View<int *, DeviceType> &offset, int)
  {
    std::tie(offset, indices) =
        BoostRTreeHelpers::performQueries(_tree, queries);
  }

private:
  BoostRTreeHelpers::RTree<ArborX::Point> _tree;
};
#endif

template <typename DeviceType>
Kokkos::View<ArborX::Point *, DeviceType>
constructPoints(int n_values, PointCloudType point_cloud_type)
{
  Kokkos::View<ArborX::Point *, DeviceType> random_points(
      Kokkos::ViewAllocateWithoutInitializing("random_points"), n_values);
  // Generate random points uniformely distributed within a box.  The edge
  // length of the box chosen such that object density (here objects will be
  // boxes 2x2x2 centered around a random point) will remain constant as
  // problem size is changed.
  auto const a = std::cbrt(n_values);
  generatePointCloud(point_cloud_type, a, random_points);

  return random_points;
}

template <typename DeviceType>
Kokkos::View<ArborX::Nearest<ArborX::Point> *, DeviceType>
makeNearestQueries(int n_values, int n_queries, int n_neighbors,
                   PointCloudType target_point_cloud_type)
{
  Kokkos::View<ArborX::Point *, DeviceType> random_points(
      Kokkos::ViewAllocateWithoutInitializing("random_points"), n_queries);
  auto const a = std::cbrt(n_values);
  generatePointCloud(target_point_cloud_type, a, random_points);

  Kokkos::View<ArborX::Nearest<ArborX::Point> *, DeviceType> queries(
      Kokkos::ViewAllocateWithoutInitializing("queries"), n_queries);
  using ExecutionSpace = typename DeviceType::execution_space;
  Kokkos::parallel_for(
      "bvh_driver:setup_knn_search_queries",
      Kokkos::RangePolicy<ExecutionSpace>(0, n_queries), KOKKOS_LAMBDA(int i) {
        queries(i) =
            ArborX::nearest<ArborX::Point>(random_points(i), n_neighbors);
      });
  Kokkos::fence();
  return queries;
}

template <typename DeviceType>
Kokkos::View<ArborX::Within *, DeviceType>
makeSpatialQueries(int n_values, int n_queries, int n_neighbors,
                   PointCloudType target_point_cloud_type)
{
  Kokkos::View<ArborX::Point *, DeviceType> random_points(
      Kokkos::ViewAllocateWithoutInitializing("random_points"), n_queries);
  auto const a = std::cbrt(n_values);
  generatePointCloud(target_point_cloud_type, a, random_points);

  Kokkos::View<ArborX::Within *, DeviceType> queries(
      Kokkos::ViewAllocateWithoutInitializing("queries"), n_queries);
  // radius chosen in order to control the number of results per query
  // NOTE: minus "1+sqrt(3)/2 \approx 1.37" matches the size of the boxes
  // inserted into the tree (mid-point between half-edge and half-diagonal)
  double const r =
      2. * std::cbrt(static_cast<double>(n_neighbors) * 3. / (4. * M_PI)) -
      (1. + std::sqrt(3.)) / 2.;
  using ExecutionSpace = typename DeviceType::execution_space;
  Kokkos::parallel_for("bvh_driver:setup_radius_search_queries",
                       Kokkos::RangePolicy<ExecutionSpace>(0, n_queries),
                       KOKKOS_LAMBDA(int i) {
                         queries(i) = ArborX::within(random_points(i), r);
                       });
  Kokkos::fence();
  return queries;
}

template <class TreeType>
void BM_construction(benchmark::State &state)
{
  using DeviceType = typename TreeType::device_type;
  int const n_values = state.range(0);
  PointCloudType point_cloud_type = static_cast<PointCloudType>(state.range(1));
  auto points = constructPoints<DeviceType>(n_values, point_cloud_type);

  for (auto _ : state)
  {
    auto const start = std::chrono::high_resolution_clock::now();
    TreeType index(points);
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
}

template <class TreeType>
void BM_knn_search(benchmark::State &state)
{
  using DeviceType = typename TreeType::device_type;
  int const n_values = state.range(0);
  int const n_queries = state.range(1);
  int const n_neighbors = state.range(2);
  PointCloudType const source_point_cloud_type =
      static_cast<PointCloudType>(state.range(3));
  PointCloudType const target_point_cloud_type =
      static_cast<PointCloudType>(state.range(4));

  TreeType index(
      constructPoints<DeviceType>(n_values, source_point_cloud_type));
  auto const queries = makeNearestQueries<DeviceType>(
      n_values, n_queries, n_neighbors, target_point_cloud_type);

  for (auto _ : state)
  {
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    Kokkos::View<int *, DeviceType> indices("indices", 0);
    auto const start = std::chrono::high_resolution_clock::now();
    index.query(queries, indices, offset);
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
}

template <class TreeType>
void BM_radius_search(benchmark::State &state)
{
  using DeviceType = typename TreeType::device_type;
  int const n_values = state.range(0);
  int const n_queries = state.range(1);
  int const n_neighbors = state.range(2);
  int const buffer_size = state.range(3);
  PointCloudType const source_point_cloud_type =
      static_cast<PointCloudType>(state.range(4));
  PointCloudType const target_point_cloud_type =
      static_cast<PointCloudType>(state.range(5));

  TreeType index(
      constructPoints<DeviceType>(n_values, source_point_cloud_type));
  auto const queries = makeSpatialQueries<DeviceType>(
      n_values, n_queries, n_neighbors, target_point_cloud_type);

  bool first_pass = true;
  for (auto _ : state)
  {
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    Kokkos::View<int *, DeviceType> indices("indices", 0);
    auto const start = std::chrono::high_resolution_clock::now();
    index.query(queries, indices, offset, buffer_size);
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());

    if (first_pass)
    {
      auto offset_clone = ArborX::clone(offset);
      ArborX::adjacentDifference(offset, offset_clone);
      double const max = ArborX::max(offset_clone);
      double const avg = ArborX::lastElement(offset) / n_queries;
      auto offset_clone_subview = Kokkos::subview(
          offset_clone, std::make_pair(1, offset_clone.extent_int(0)));
      double const min = ArborX::min(offset_clone_subview);

      std::ostream &os = std::cout;
      os << "min number of neighbors " << min << "\n";
      os << "max number of neighbors " << max << "\n";
      os << "avg number of neighbors " << avg << "\n";

      first_pass = false;
    }
  }
}

class KokkosScopeGuard
{
public:
  KokkosScopeGuard(int &argc, char *argv[]) { Kokkos::initialize(argc, argv); }
  ~KokkosScopeGuard() { Kokkos::finalize(); }
};

#define REGISTER_BENCHMARK(TreeType)                                           \
  BENCHMARK_TEMPLATE(BM_construction, TreeType)                                \
      ->Args({n_values, source_point_cloud_type})                              \
      ->UseManualTime()                                                        \
      ->Unit(benchmark::kMicrosecond);                                         \
  BENCHMARK_TEMPLATE(BM_knn_search, TreeType)                                  \
      ->Args({n_values, n_queries, n_neighbors, source_point_cloud_type,       \
              target_point_cloud_type})                                        \
      ->UseManualTime()                                                        \
      ->Unit(benchmark::kMicrosecond);                                         \
  BENCHMARK_TEMPLATE(BM_radius_search, TreeType)                               \
      ->Args({n_values, n_queries, n_neighbors, buffer_size,                   \
              source_point_cloud_type, target_point_cloud_type})               \
      ->UseManualTime()                                                        \
      ->Unit(benchmark::kMicrosecond);

int main(int argc, char *argv[])
{
  KokkosScopeGuard guard(argc, argv);

  namespace bpo = boost::program_options;
  bpo::options_description desc("Allowed options");
  int n_values = 50000;
  int n_queries = 20000;
  int n_neighbors = 10;
  int buffer_size = 0;
  std::string source_pt_cloud = "filled_box";
  std::string target_pt_cloud = "filled_box";
  // clang-format off
    desc.add_options()
        ( "help", "produce help message" )
        ( "values", bpo::value<int>(&n_values)->default_value(50000), "number of indexable values (source)" )
        ( "queries", bpo::value<int>(&n_queries)->default_value(20000), "number of queries (target)" )
        ( "neighbors", bpo::value<int>(&n_neighbors)->default_value(10), "desired number of results per query" )
        ( "buffer", bpo::value<int>(&buffer_size)->default_value(0), "size for buffer optimization in radius search" )
        ( "source-point-cloud-type", bpo::value<std::string>(&source_pt_cloud)->default_value("filled_box"), "shape of the source point cloud"  )
        ( "target-point-cloud-type", bpo::value<std::string>(&target_pt_cloud)->default_value("filled_box"), "shape of the target point cloud"  )
    ;
  // clang-format on
  bpo::variables_map vm;
  bpo::parsed_options opts = bpo::command_line_parser(argc, argv)
                                 .options(desc)
                                 .allow_unregistered()
                                 .run();
  bpo::store(opts, vm);
  bpo::notify(vm);

  if (vm.count("help"))
  {
    // Full list of options consists of Kokkos + Boost.Program_options +
    // Google Benchmark and we still need to call benchmark::Initialize() to
    // get those printed to the standard output.
    std::cout << desc << "\n";
    int ac = 2;
    char *av[] = {(char *)"ignored", (char *)"--help"};
    // benchmark::Initialize() calls exit(0) when `--help` so register
    // Kokkos::finalize() to be called on normal program termination.
    std::atexit(Kokkos::finalize);
    benchmark::Initialize(&ac, av);
    return 1;
  }
  else
  {
    benchmark::Initialize(&argc, argv);
    // Throw if some of the arguments have not been recognized.
    std::ignore = bpo::command_line_parser(argc, argv)
                      .options(bpo::options_description(""))
                      .run();
  }

  // Google benchmark only supports integer arguments (see
  // https://github.com/google/benchmark/issues/387), so we map the string to
  // an enum.
  std::map<std::string, PointCloudType> to_point_cloud_enum;
  to_point_cloud_enum["filled_box"] = PointCloudType::filled_box;
  to_point_cloud_enum["hollow_box"] = PointCloudType::hollow_box;
  to_point_cloud_enum["filled_sphere"] = PointCloudType::filled_sphere;
  to_point_cloud_enum["hollow_sphere"] = PointCloudType::hollow_sphere;
  int source_point_cloud_type = to_point_cloud_enum.at(source_pt_cloud);
  int target_point_cloud_type = to_point_cloud_enum.at(target_pt_cloud);

#ifdef KOKKOS_ENABLE_SERIAL
  using Serial = Kokkos::Serial::device_type;
  REGISTER_BENCHMARK(ArborX::BVH<Serial>);
#endif

#ifdef KOKKOS_ENABLE_OPENMP
  using OpenMP = Kokkos::OpenMP::device_type;
  REGISTER_BENCHMARK(ArborX::BVH<OpenMP>);
#endif

#ifdef KOKKOS_ENABLE_CUDA
  // using Cuda = Kokkos::Cuda::device_type; // <- FIXME segfault
  using Cuda = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaUVMSpace>;
  REGISTER_BENCHMARK(ArborX::BVH<Cuda>);
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
  REGISTER_BENCHMARK(BoostRTree);
#endif

  benchmark::RunSpecifiedBenchmarks();

  return EXIT_SUCCESS;
}
