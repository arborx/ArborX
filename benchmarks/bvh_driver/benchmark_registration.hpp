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

#ifndef BENCHMARK_REGISTRATION_HPP
#define BENCHMARK_REGISTRATION_HPP

#include <ArborXBenchmark_PointClouds.hpp>
#include <ArborX_Point.hpp>
#include <details/ArborX_Predicates.hpp>

#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath> // cbrt

#include <benchmark/benchmark.h>

template <class TreeType>
struct is_boost_rtree : std::false_type
{};
template <typename Geometry>
struct is_boost_rtree<BoostExt::RTree<Geometry>> : std::true_type
{};
template <typename Geometry>
inline constexpr bool is_boost_rtree_v = is_boost_rtree<Geometry>::value;

struct Spec
{
  using PointCloudType = ArborXBenchmark::PointCloudType;

  std::string backends;
  int n_values;
  int n_queries;
  int n_neighbors;
  bool sort_predicates;
  int buffer_size;
  PointCloudType source_point_cloud_type;
  PointCloudType target_point_cloud_type;

  Spec() = default;
  Spec(std::string const &spec_string)
  {
    std::istringstream ss(spec_string);
    std::string token;

    // clang-format off
    getline(ss, token, '/');  backends = token;
    getline(ss, token, '/');  n_values = std::stoi(token);
    getline(ss, token, '/');  n_queries = std::stoi(token);
    getline(ss, token, '/');  n_neighbors = std::stoi(token);
    getline(ss, token, '/');  sort_predicates = static_cast<bool>(std::stoi(token));
    getline(ss, token, '/');  buffer_size = std::stoi(token);
    getline(ss, token, '/');  source_point_cloud_type = static_cast<PointCloudType>(std::stoi(token));
    getline(ss, token, '/');  target_point_cloud_type = static_cast<PointCloudType>(std::stoi(token));
    // clang-format on

    if (!(backends == "all" || backends == "serial" || backends == "openmp" ||
          backends == "threads" || backends == "cuda" || backends == "rtree" ||
          backends == "hip" || backends == "sycl" ||
          backends == "openmptarget"))
      throw std::runtime_error("Backend " + backends + " invalid!");
  }

  std::string create_label_construction(std::string const &tree_name) const
  {
    std::string s = std::string("BM_construction<") + tree_name + ">";
    for (auto const &var :
         {n_values, static_cast<int>(source_point_cloud_type)})
      s += "/" + std::to_string(var);
    return s;
  }

  std::string create_label_radius_search(std::string const &tree_name,
                                         std::string const &flavor = "") const
  {
    std::string s = std::string("BM_radius_") +
                    (flavor.empty() ? "" : flavor + "_") + "search<" +
                    tree_name + ">";
    for (auto const &var :
         {n_values, n_queries, n_neighbors, static_cast<int>(sort_predicates),
          buffer_size, static_cast<int>(source_point_cloud_type),
          static_cast<int>(target_point_cloud_type)})
      s += "/" + std::to_string(var);
    return s;
  };

  std::string create_label_knn_search(std::string const &tree_name,
                                      std::string const &flavor = "") const
  {
    std::string s = std::string("BM_knn_") +
                    (flavor.empty() ? "" : flavor + "_") + "search<" +
                    tree_name + ">";
    for (auto const &var :
         {n_values, n_queries, n_neighbors, static_cast<int>(sort_predicates),
          static_cast<int>(source_point_cloud_type),
          static_cast<int>(target_point_cloud_type)})
      s += "/" + std::to_string(var);
    return s;
  };
};

template <typename DeviceType>
auto constructPoints(int n_values,
                     ArborXBenchmark::PointCloudType point_cloud_type)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace exec;

  using Point = ArborX::Point<3>;
  Kokkos::View<Point *, DeviceType> random_points(
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                         "Benchmark::random_points"),
      n_values);
  // Generate random points uniformly distributed within a box.  The edge
  // length of the box chosen such that object density (here objects will be
  // boxes 2x2x2 centered around a random point) will remain constant as
  // problem size is changed.
  auto const a = std::cbrt(n_values);
  ArborXBenchmark::generatePointCloud(exec, point_cloud_type, a, random_points);

  return random_points;
}

template <typename TreeType, typename ExecutionSpace, typename Primitives>
auto makeTree(ExecutionSpace const &space, Primitives const &primitives)
{
  if constexpr (is_boost_rtree_v<TreeType>)
    return TreeType(space, primitives);
  else
    return TreeType(space, ArborX::Experimental::attach_indices(primitives));
}

template <typename DeviceType>
auto makeSpatialQueries(int n_values, int n_queries, int n_neighbors,
                        ArborXBenchmark::PointCloudType target_point_cloud_type)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace exec;

  using Point = ArborX::Point<3>;
  Kokkos::View<Point *, DeviceType> random_points(
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                         "Benchmark::random_points"),
      n_queries);
  auto const a = std::cbrt(n_values);
  ArborXBenchmark::generatePointCloud(exec, target_point_cloud_type, a,
                                      random_points);

  // Radius is computed so that the number of results per query for a uniformly
  // distributed points in a [-a,a]^3 box is approximately n_neighbors.
  // Calculation: n_values*(4/3*pi*r^3)/(2a)^3 = n_neighbors
  double const r = std::cbrt(static_cast<double>(n_neighbors) * 6. /
                             Kokkos::numbers::pi_v<double>);

  return ArborX::Experimental::make_intersects(random_points, r);
}

template <typename DeviceType>
auto makeNearestQueries(int n_values, int n_queries, int n_neighbors,
                        ArborXBenchmark::PointCloudType target_point_cloud_type)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace exec;

  using Point = ArborX::Point<3>;
  Kokkos::View<Point *, DeviceType> random_points(
      Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                         "Benchmark::random_points"),
      n_queries);
  auto const a = std::cbrt(n_values);
  ArborXBenchmark::generatePointCloud(exec, target_point_cloud_type, a,
                                      random_points);

  return ArborX::Experimental::make_nearest(random_points, n_neighbors);
}

template <typename DeviceType>
struct CountCallback
{
  Kokkos::View<int *, DeviceType> count_;

  template <typename Query, typename Value>
  KOKKOS_FUNCTION void operator()(Query const &query, Value) const
  {
    auto const i = ArborX::getData(query);
    Kokkos::atomic_increment(&count_(i));
  }
};

template <typename ExecutionSpace, class TreeType>
void BM_construction(benchmark::State &state, Spec const &spec)
{
  using DeviceType =
      Kokkos::Device<ExecutionSpace, typename TreeType::memory_space>;

  ExecutionSpace exec_space;

  auto const points =
      constructPoints<DeviceType>(spec.n_values, spec.source_point_cloud_type);

  for (auto _ : state)
  {
    exec_space.fence();
    auto const start = std::chrono::high_resolution_clock::now();

    auto index = makeTree<TreeType>(exec_space, points);

    exec_space.fence();
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
  state.counters["rate"] = benchmark::Counter(
      spec.n_values, benchmark::Counter::kIsIterationInvariantRate);
}

template <typename ExecutionSpace, class TreeType>
void BM_radius_search(benchmark::State &state, Spec const &spec)
{
  using DeviceType =
      Kokkos::Device<ExecutionSpace, typename TreeType::memory_space>;

  ExecutionSpace exec_space;

  auto index = makeTree<TreeType>(
      exec_space,
      constructPoints<DeviceType>(spec.n_values, spec.source_point_cloud_type));
  auto const queries = makeSpatialQueries<DeviceType>(
      spec.n_values, spec.n_queries, spec.n_neighbors,
      spec.target_point_cloud_type);

  for (auto _ : state)
  {
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    Kokkos::View<int *, DeviceType> indices("indices", 0);

    exec_space.fence();
    auto const start = std::chrono::high_resolution_clock::now();

    ArborX::query(index, exec_space, queries, indices, offset,
                  ArborX::Experimental::TraversalPolicy()
                      .setPredicateSorting(spec.sort_predicates)
                      .setBufferSize(spec.buffer_size));

    exec_space.fence();
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
  state.counters["rate"] = benchmark::Counter(
      spec.n_queries, benchmark::Counter::kIsIterationInvariantRate);
}

template <typename ExecutionSpace, class TreeType>
void BM_radius_callback_search(benchmark::State &state, Spec const &spec)
{
  using DeviceType =
      Kokkos::Device<ExecutionSpace, typename TreeType::memory_space>;

  ExecutionSpace exec_space;

  auto index = makeTree<TreeType>(
      ExecutionSpace{},
      constructPoints<DeviceType>(spec.n_values, spec.source_point_cloud_type));
  auto const queries = makeSpatialQueries<DeviceType>(
      spec.n_values, spec.n_queries, spec.n_neighbors,
      spec.target_point_cloud_type);

  for (auto _ : state)
  {
    Kokkos::View<int *, DeviceType> num_neigh("Testing::num_neigh",
                                              spec.n_queries);
    CountCallback<DeviceType> callback{num_neigh};

    exec_space.fence();
    auto const start = std::chrono::high_resolution_clock::now();

    index.query(exec_space, ArborX::Experimental::attach_indices<int>(queries),
                callback,
                ArborX::Experimental::TraversalPolicy().setPredicateSorting(
                    spec.sort_predicates));

    exec_space.fence();
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
  state.counters["rate"] = benchmark::Counter(
      spec.n_queries, benchmark::Counter::kIsIterationInvariantRate);
}

template <typename ExecutionSpace, class TreeType>
void BM_knn_search(benchmark::State &state, Spec const &spec)
{
  using DeviceType =
      Kokkos::Device<ExecutionSpace, typename TreeType::memory_space>;

  ExecutionSpace exec_space;

  auto index = makeTree<TreeType>(
      exec_space,
      constructPoints<DeviceType>(spec.n_values, spec.source_point_cloud_type));
  auto const queries = makeNearestQueries<DeviceType>(
      spec.n_values, spec.n_queries, spec.n_neighbors,
      spec.target_point_cloud_type);

  for (auto _ : state)
  {
    Kokkos::View<int *, DeviceType> offset("Benchmark::offset", 0);
    Kokkos::View<int *, DeviceType> indices("Benchmark::indices", 0);

    exec_space.fence();
    auto const start = std::chrono::high_resolution_clock::now();

    ArborX::query(index, exec_space, queries, indices, offset,
                  ArborX::Experimental::TraversalPolicy().setPredicateSorting(
                      spec.sort_predicates));

    exec_space.fence();
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
  state.counters["rate"] = benchmark::Counter(
      spec.n_queries, benchmark::Counter::kIsIterationInvariantRate);
}

template <typename ExecutionSpace, class TreeType>
void BM_knn_callback_search(benchmark::State &state, Spec const &spec)
{
  using DeviceType =
      Kokkos::Device<ExecutionSpace, typename TreeType::memory_space>;

  ExecutionSpace exec_space;

  auto index = makeTree<TreeType>(
      exec_space,
      constructPoints<DeviceType>(spec.n_values, spec.source_point_cloud_type));
  auto const queries = makeNearestQueries<DeviceType>(
      spec.n_values, spec.n_queries, spec.n_neighbors,
      spec.target_point_cloud_type);

  for (auto _ : state)
  {
    Kokkos::View<int *, DeviceType> num_neigh("Benchmark::num_neigh",
                                              spec.n_queries);
    CountCallback<DeviceType> callback{num_neigh};

    exec_space.fence();
    auto const start = std::chrono::high_resolution_clock::now();

    index.query(exec_space, ArborX::Experimental::attach_indices<int>(queries),
                callback,
                ArborX::Experimental::TraversalPolicy().setPredicateSorting(
                    spec.sort_predicates));

    exec_space.fence();
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
  state.counters["rate"] = benchmark::Counter(
      spec.n_queries, benchmark::Counter::kIsIterationInvariantRate);
}

template <typename ExecutionSpace, typename TreeType>
void register_benchmark_construction(Spec const &spec,
                                     std::string const &description)
{
  benchmark::RegisterBenchmark(
      spec.create_label_construction(description).c_str(),
      [=](benchmark::State &state) {
        BM_construction<ExecutionSpace, TreeType>(state, spec);
      })
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
}

template <typename ExecutionSpace, typename TreeType>
void register_benchmark_spatial_query_no_callback(
    Spec const &spec, std::string const &description)
{
  benchmark::RegisterBenchmark(
      spec.create_label_radius_search(description).c_str(),
      [=](benchmark::State &state) {
        BM_radius_search<ExecutionSpace, TreeType>(state, spec);
      })
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
}

template <typename ExecutionSpace, typename TreeType>
void register_benchmark_spatial_query_callback(Spec const &spec,
                                               std::string const &description)
{
  benchmark::RegisterBenchmark(
      spec.create_label_radius_search(description, "callback").c_str(),
      [=](benchmark::State &state) {
        BM_radius_callback_search<ExecutionSpace, TreeType>(state, spec);
      })
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
}

template <typename ExecutionSpace, typename TreeType>
void register_benchmark_nearest_query_no_callback(
    Spec const &spec, std::string const &description)
{
  benchmark::RegisterBenchmark(
      spec.create_label_knn_search(description).c_str(),
      [=](benchmark::State &state) {
        BM_knn_search<ExecutionSpace, TreeType>(state, spec);
      })
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
}

template <typename ExecutionSpace, typename TreeType>
void register_benchmark_nearest_query_callback(Spec const &spec,
                                               std::string const &description)
{
  benchmark::RegisterBenchmark(
      spec.create_label_knn_search(description, "callback").c_str(),
      [=](benchmark::State &state) {
        BM_knn_callback_search<ExecutionSpace, TreeType>(state, spec);
      })
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
}

#endif
