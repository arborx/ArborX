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

#include <ArborX_Point.hpp>
#include <ArborX_Predicates.hpp>

#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath> // cbrt

#include <benchmark/benchmark.h>
#include <point_clouds.hpp>

struct Spec
{
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
Kokkos::View<ArborX::Point *, DeviceType>
constructPoints(int n_values, PointCloudType point_cloud_type)
{
  Kokkos::View<ArborX::Point *, DeviceType> random_points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "random_points"),
      n_values);
  // Generate random points uniformly distributed within a box.  The edge
  // length of the box chosen such that object density (here objects will be
  // boxes 2x2x2 centered around a random point) will remain constant as
  // problem size is changed.
  auto const a = std::cbrt(n_values);
  generatePointCloud(point_cloud_type, a, random_points);

  return random_points;
}

template <typename DeviceType>
Kokkos::View<decltype(ArborX::intersects(ArborX::Sphere{})) *, DeviceType>
makeSpatialQueries(int n_values, int n_queries, int n_neighbors,
                   PointCloudType target_point_cloud_type)
{
  Kokkos::View<ArborX::Point *, DeviceType> random_points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "random_points"),
      n_queries);
  auto const a = std::cbrt(n_values);
  generatePointCloud(target_point_cloud_type, a, random_points);

  Kokkos::View<decltype(ArborX::intersects(ArborX::Sphere{})) *, DeviceType>
      queries(Kokkos::view_alloc(Kokkos::WithoutInitializing, "queries"),
              n_queries);
  // Radius is computed so that the number of results per query for a uniformly
  // distributed points in a [-a,a]^3 box is approximately n_neighbors.
  // Calculation: n_values*(4/3*M_PI*r^3)/(2a)^3 = n_neighbors
  double const r = std::cbrt(static_cast<double>(n_neighbors) * 6. / M_PI);
  using ExecutionSpace = typename DeviceType::execution_space;
  Kokkos::parallel_for(
      "bvh_driver:setup_radius_search_queries",
      Kokkos::RangePolicy<ExecutionSpace>(0, n_queries), KOKKOS_LAMBDA(int i) {
        queries(i) = ArborX::intersects(ArborX::Sphere{random_points(i), r});
      });
  return queries;
}

template <typename DeviceType>
Kokkos::View<ArborX::Nearest<ArborX::Point> *, DeviceType>
makeNearestQueries(int n_values, int n_queries, int n_neighbors,
                   PointCloudType target_point_cloud_type)
{
  Kokkos::View<ArborX::Point *, DeviceType> random_points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "random_points"),
      n_queries);
  auto const a = std::cbrt(n_values);
  generatePointCloud(target_point_cloud_type, a, random_points);

  Kokkos::View<ArborX::Nearest<ArborX::Point> *, DeviceType> queries(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "queries"), n_queries);
  using ExecutionSpace = typename DeviceType::execution_space;
  Kokkos::parallel_for(
      "bvh_driver:setup_knn_search_queries",
      Kokkos::RangePolicy<ExecutionSpace>(0, n_queries), KOKKOS_LAMBDA(int i) {
        queries(i) =
            ArborX::nearest<ArborX::Point>(random_points(i), n_neighbors);
      });
  return queries;
}

template <typename Queries>
struct QueriesWithIndex
{
  Queries _queries;
};

template <typename Queries>
struct ArborX::AccessTraits<QueriesWithIndex<Queries>, ArborX::PredicatesTag>
{
  using memory_space = typename Queries::memory_space;
  static KOKKOS_FUNCTION size_t size(QueriesWithIndex<Queries> const &q)
  {
    return q._queries.extent(0);
  }
  static KOKKOS_FUNCTION auto get(QueriesWithIndex<Queries> const &q, size_t i)
  {
    return attach(q._queries(i), (int)i);
  }
};

template <typename DeviceType>
struct CountCallback
{
  Kokkos::View<int *, DeviceType> count_;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int) const
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

  auto const points =
      constructPoints<DeviceType>(spec.n_values, spec.source_point_cloud_type);

  ExecutionSpace exec_space;

  for (auto _ : state)
  {
    exec_space.fence();
    auto const start = std::chrono::high_resolution_clock::now();

    TreeType index(exec_space, points);

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

  TreeType index(exec_space, constructPoints<DeviceType>(
                                 spec.n_values, spec.source_point_cloud_type));
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

  TreeType index(
      ExecutionSpace{},
      constructPoints<DeviceType>(spec.n_values, spec.source_point_cloud_type));
  auto const queries_no_index = makeSpatialQueries<DeviceType>(
      spec.n_values, spec.n_queries, spec.n_neighbors,
      spec.target_point_cloud_type);
  QueriesWithIndex<decltype(queries_no_index)> queries{queries_no_index};

  for (auto _ : state)
  {
    Kokkos::View<int *, DeviceType> num_neigh("Testing::num_neigh",
                                              spec.n_queries);
    CountCallback<DeviceType> callback{num_neigh};

    exec_space.fence();
    auto const start = std::chrono::high_resolution_clock::now();

    index.query(exec_space, queries, callback,
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

  TreeType index(exec_space, constructPoints<DeviceType>(
                                 spec.n_values, spec.source_point_cloud_type));
  auto const queries = makeNearestQueries<DeviceType>(
      spec.n_values, spec.n_queries, spec.n_neighbors,
      spec.target_point_cloud_type);

  for (auto _ : state)
  {
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    Kokkos::View<int *, DeviceType> indices("indices", 0);

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

  TreeType index(exec_space, constructPoints<DeviceType>(
                                 spec.n_values, spec.source_point_cloud_type));
  auto const queries_no_index = makeNearestQueries<DeviceType>(
      spec.n_values, spec.n_queries, spec.n_neighbors,
      spec.target_point_cloud_type);
  QueriesWithIndex<decltype(queries_no_index)> queries{queries_no_index};

  for (auto _ : state)
  {
    Kokkos::View<int *, DeviceType> num_neigh("Testing::num_neigh",
                                              spec.n_queries);
    CountCallback<DeviceType> callback{num_neigh};

    exec_space.fence();
    auto const start = std::chrono::high_resolution_clock::now();

    index.query(exec_space, queries, callback,
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
