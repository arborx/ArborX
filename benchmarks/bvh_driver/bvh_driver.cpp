/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_BoostRTreeHelpers.hpp>
#include <ArborX_DetailsBatchedQueries.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <chrono>
#include <cmath> // cbrt
#include <cstdlib>
#include <random>

#ifdef ARBORX_PERFORMANCE_TESTING
#include <mpi.h>
#endif

#include <benchmark/benchmark.h>
#include <point_clouds.hpp>

struct Spec
{
  std::string create_label_construction(std::string const &tree_name) const
  {
    std::string s = std::string("BM_construction<") + tree_name + ">";
    for (auto const &var :
         {n_values, static_cast<int>(source_point_cloud_type)})
      s += "/" + std::to_string(var);
    return s;
  }

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
  }

  std::string backends;
  int n_values;
  int n_queries;
  int n_neighbors;
  bool sort_predicates;
  int buffer_size;
  PointCloudType source_point_cloud_type;
  PointCloudType target_point_cloud_type;
};

Spec create_spec_from_string(std::string const &spec_string)
{
  std::istringstream ss(spec_string);
  std::string token;

  Spec spec;

  // clang-format off
    getline(ss, token, '/');  spec.backends = token;
    getline(ss, token, '/');  spec.n_values = std::stoi(token);
    getline(ss, token, '/');  spec.n_queries = std::stoi(token);
    getline(ss, token, '/');  spec.n_neighbors = std::stoi(token);
    getline(ss, token, '/');  spec.sort_predicates = static_cast<bool>(std::stoi(token));
    getline(ss, token, '/');  spec.buffer_size = std::stoi(token);
    getline(ss, token, '/');  spec.source_point_cloud_type = static_cast<PointCloudType>(std::stoi(token));
    getline(ss, token, '/');  spec.target_point_cloud_type = static_cast<PointCloudType>(std::stoi(token));
  // clang-format on

  if (!(spec.backends == "all" || spec.backends == "serial" ||
        spec.backends == "openmp" || spec.backends == "threads" ||
        spec.backends == "cuda" || spec.backends == "rtree" ||
        spec.backends == "hip"))
    throw std::runtime_error("Backend " + spec.backends + " invalid!");

  return spec;
}

template <typename DeviceType>
Kokkos::View<ArborX::Point *, DeviceType>
constructPoints(int n_values, PointCloudType point_cloud_type)
{
  Kokkos::View<ArborX::Point *, DeviceType> random_points(
      Kokkos::ViewAllocateWithoutInitializing("random_points"), n_values);
  // Generate random points uniformly distributed within a box.  The edge
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
  return queries;
}

template <typename DeviceType>
Kokkos::View<decltype(ArborX::intersects(ArborX::Sphere{})) *, DeviceType>
makeSpatialQueries(int n_values, int n_queries, int n_neighbors,
                   PointCloudType target_point_cloud_type)
{
  Kokkos::View<ArborX::Point *, DeviceType> random_points(
      Kokkos::ViewAllocateWithoutInitializing("random_points"), n_queries);
  auto const a = std::cbrt(n_values);
  generatePointCloud(target_point_cloud_type, a, random_points);

  Kokkos::View<decltype(ArborX::intersects(ArborX::Sphere{})) *, DeviceType>
      queries(Kokkos::ViewAllocateWithoutInitializing("queries"), n_queries);
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

template <typename Queries>
struct QueriesWithIndex
{
  Queries _queries;
};

template <typename Queries>
struct ArborX::AccessTraits<QueriesWithIndex<Queries>, ArborX::PredicatesTag>
{
  using memory_space = typename Queries::memory_space;
  static size_t size(QueriesWithIndex<Queries> const &q)
  {
    return q._queries.extent(0);
  }
  static KOKKOS_FUNCTION auto get(QueriesWithIndex<Queries> const &q, size_t i)
  {
    return attach(q._queries(i), (int)i);
  }
};

template <class TreeType>
void BM_construction(benchmark::State &state, Spec const &spec)
{
  using DeviceType = typename TreeType::device_type;
  auto const points =
      constructPoints<DeviceType>(spec.n_values, spec.source_point_cloud_type);

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
void BM_knn_search(benchmark::State &state, Spec const &spec)
{
  using DeviceType = typename TreeType::device_type;

  TreeType index(
      constructPoints<DeviceType>(spec.n_values, spec.source_point_cloud_type));
  auto const queries = makeNearestQueries<DeviceType>(
      spec.n_values, spec.n_queries, spec.n_neighbors,
      spec.target_point_cloud_type);

  for (auto _ : state)
  {
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    Kokkos::View<int *, DeviceType> indices("indices", 0);
    auto const start = std::chrono::high_resolution_clock::now();
    index.query(queries, indices, offset,
                ArborX::Experimental::TraversalPolicy().setPredicateSorting(
                    spec.sort_predicates));
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
}

template <typename DeviceType>
struct NearestCallback
{
  Kokkos::View<int *, DeviceType> count_;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int, float) const
  {
    auto const i = ArborX::getData(query);
    Kokkos::atomic_fetch_add(&count_(i), 1);
  }
};

template <class TreeType>
void BM_knn_callback_search(benchmark::State &state, Spec const &spec)
{
  using DeviceType = typename TreeType::device_type;

  TreeType index(
      constructPoints<DeviceType>(spec.n_values, spec.source_point_cloud_type));
  auto const queries_no_index = makeNearestQueries<DeviceType>(
      spec.n_values, spec.n_queries, spec.n_neighbors,
      spec.target_point_cloud_type);
  QueriesWithIndex<decltype(queries_no_index)> queries{queries_no_index};

  for (auto _ : state)
  {
    Kokkos::View<int *, DeviceType> num_neigh("Testing::num_neigh",
                                              spec.n_queries);
    NearestCallback<DeviceType> callback{num_neigh};

    auto const start = std::chrono::high_resolution_clock::now();
    index.query(queries, callback,
                ArborX::Experimental::TraversalPolicy().setPredicateSorting(
                    spec.sort_predicates));
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
}

template <class TreeType>
void BM_radius_search(benchmark::State &state, Spec const &spec)
{
  using DeviceType = typename TreeType::device_type;
  using ExecutionSpace = typename DeviceType::execution_space;

  TreeType index(
      constructPoints<DeviceType>(spec.n_values, spec.source_point_cloud_type));
  auto queries = makeSpatialQueries<DeviceType>(spec.n_values, spec.n_queries,
                                                spec.n_neighbors,
                                                spec.target_point_cloud_type);

  for (auto _ : state)
  {
    if (spec.sort_predicates)
    {
      using Access =
          ArborX::AccessTraits<decltype(queries), ArborX::PredicatesTag>;

      auto permute = ArborX::Details::BatchedQueries<
          DeviceType>::sortQueriesAlongZOrderCurve(ExecutionSpace{},
                                                   index.bounds(), queries);
      queries = ArborX::Details::BatchedQueries<DeviceType>::applyPermutation(
          ExecutionSpace{}, permute, queries);
      // DEBUG
      /*auto permute2 =
      ArborX::Details::BatchedQueries<DeviceType>::sortQueriesAlongZOrderCurve(ExecutionSpace{},
      index.bounds(), queries);

      Kokkos::parallel_for("check_permutation",
      Kokkos::RangePolicy<ExecutionSpace>(0, Access::size(queries)),
                        KOKKOS_LAMBDA(const int i) {
                          if (permute2(i) != i)
                          {
                            printf("%d should be %d\n", permute2(i), i);
                            Kokkos::abort("wrong permutation");
                          }
                        });*/
    }

    Kokkos::View<int *, DeviceType> offset("offset", 0);
    Kokkos::View<int *, DeviceType> indices("indices", 0);
    auto const start = std::chrono::high_resolution_clock::now();
    index.query(queries, indices, offset,
                ArborX::Experimental::TraversalPolicy()
                    .setPredicateSorting(false)
                    .setBufferSize(spec.buffer_size));
    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());
  }
}

template <typename DeviceType>
struct SpatialCallback_FirstPass_BufferOptimization
{
  Kokkos::View<int *, DeviceType> offsets_;
  Kokkos::View<int *, DeviceType> indices_;
  Kokkos::View<int *, DeviceType> counts_;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int primitive_index) const
  {
    auto const predicate_index = ArborX::getData(query);
    auto const &offset = offsets_(predicate_index);
    auto const buffer_size = *(&offset + 1) - offset;

    int const count_old =
        Kokkos::atomic_fetch_add(&counts_(predicate_index), 1);

    if (count_old < buffer_size)
      indices_(offset + count_old) = primitive_index;
  }
};

template <typename DeviceType>
struct SpatialCallback_SecondPass_BufferOptimization
{
  Kokkos::View<int *, DeviceType> offsets_;
  Kokkos::View<int *, DeviceType> indices_;
  Kokkos::View<int *, DeviceType> counts_;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int primitive_index) const
  {
    auto const predicate_index = ArborX::getData(query);
    auto const &offset = offsets_(predicate_index);
    auto const buffer_size = *(&offset + 1) - offset;

    int const count_old =
        Kokkos::atomic_fetch_add(&counts_(predicate_index), 1);

    // DEBUG
    /*    if (!(count_old < buffer_size))
          Kokkos::abort("Didn't update buffer size");*/
    indices_(offset + count_old) = primitive_index;
  }
};

template <typename DeviceType>
struct SpatialCallback
{
  Kokkos::View<int *, DeviceType> count_;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int) const
  {
    auto const i = ArborX::getData(query);
    Kokkos::atomic_fetch_add(&count_(i), 1);
  }
};

template <class TreeType>
void BM_radius_callback_search(benchmark::State &state, Spec const &spec)
{
  using DeviceType = typename TreeType::device_type;
  using ExecutionSpace = typename DeviceType::execution_space;

  TreeType index(
      constructPoints<DeviceType>(spec.n_values, spec.source_point_cloud_type));
  auto const queries_no_index = makeSpatialQueries<DeviceType>(
      spec.n_values, spec.n_queries, spec.n_neighbors,
      spec.target_point_cloud_type);

  QueriesWithIndex<decltype(queries_no_index)> queries{queries_no_index};

  Kokkos::View<int *, DeviceType> offsets("offsets", 0);
  Kokkos::View<int *, DeviceType> indices("indices", 0);
  Kokkos::View<int *, DeviceType> counts("counts", 0);

  SpatialCallback_FirstPass_BufferOptimization<DeviceType> callback_first;
  SpatialCallback_SecondPass_BufferOptimization<DeviceType> callback_second;

  for (auto _ : state)
  {
    Kokkos::View<int *, DeviceType> num_neigh("Testing::num_neigh",
                                              spec.n_queries);

    Kokkos::realloc(offsets, spec.n_queries + 1);

    Kokkos::parallel_for(
        "initialize_offsets",
        Kokkos::RangePolicy<ExecutionSpace>(0, offsets.size()),
        KOKKOS_LAMBDA(const int &i) { offsets(i) = i * spec.buffer_size; });
    Kokkos::realloc(indices, spec.n_queries * spec.buffer_size);
    Kokkos::realloc(counts, spec.n_queries);

    callback_first.offsets_ = offsets;
    callback_first.indices_ = indices;
    callback_first.counts_ = counts;
    SpatialCallback_SecondPass_BufferOptimization<DeviceType> callback_second;
    callback_second.offsets_ = offsets;
    callback_second.indices_ = indices;
    callback_second.counts_ = counts;

    if (spec.sort_predicates)
    {
      using Access =
          ArborX::AccessTraits<decltype(queries), ArborX::PredicatesTag>;

      auto permute = ArborX::Details::BatchedQueries<
          DeviceType>::sortQueriesAlongZOrderCurve(ExecutionSpace{},
                                                   index.bounds(), queries);
      auto sorted_queries =
          ArborX::Details::BatchedQueries<DeviceType>::applyPermutation(
              ExecutionSpace{}, permute, queries);
      Kokkos::parallel_for(
          "assign_sorted_queries",
          Kokkos::RangePolicy<ExecutionSpace>(0, Access::size(queries)),
          KOKKOS_LAMBDA(const int i) {
            queries._queries(i) =
                static_cast<typename decltype(queries_no_index)::value_type>(
                    sorted_queries(i));
          });
      // DEBUG
      /*auto permute2 =
      ArborX::Details::BatchedQueries<DeviceType>::sortQueriesAlongZOrderCurve(ExecutionSpace{},
      index.bounds(), queries);

      Kokkos::parallel_for("check_permutation",
      Kokkos::RangePolicy<ExecutionSpace>(0, Access::size(queries)),
                        KOKKOS_LAMBDA(const int i) {
                          if (permute2(i) != i)
                          {
                            printf("%d should be %d\n", permute2(i), i);
                            Kokkos::abort("wrong permutation");
                          }
                        });*/
    }
    auto const start = std::chrono::high_resolution_clock::now();
    Kokkos::Profiling::pushRegion("first_pass");

    index.query(queries, callback_first,
                ArborX::Experimental::TraversalPolicy()
                    .setPredicateSorting(false)
                    .setBufferSize(spec.buffer_size));
    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("intermediate");
    int total_size = 0;
    Kokkos::parallel_reduce(
        "reduce_counts", Kokkos::RangePolicy<ExecutionSpace>(0, counts.size()),
        KOKKOS_LAMBDA(const int &i, int &sum) { sum += counts(i); },
        total_size);
    Kokkos::realloc(callback_second.indices_, total_size);

    Kokkos::parallel_scan("scan_counts",
                          Kokkos::RangePolicy<ExecutionSpace>(0, counts.size()),
                          KOKKOS_LAMBDA(int i, int &update, bool final_pass) {
                            update += counts(i);
                            if (final_pass)
                              offsets(i + 1) = update;
                          });

    Kokkos::deep_copy(callback_second.counts_, 0);
    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("second_pass");
    index.query(queries, callback_second,
                ArborX::Experimental::TraversalPolicy()
                    .setPredicateSorting(false)
                    .setBufferSize(spec.buffer_size));
    Kokkos::Profiling::popRegion();

    auto const end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    state.SetIterationTime(elapsed_seconds.count());

    // DEBUG
    /*Kokkos::View<int *, DeviceType> reference_offset("offset", 0);
    Kokkos::View<int *, DeviceType> reference_indices("indices", 0);
    index.query(queries_no_index, reference_indices, reference_offset,
                ArborX::Experimental::TraversalPolicy()
                    .setPredicateSorting(spec.sort_predicates)
                    .setBufferSize(spec.buffer_size));

    assert(reference_offset.size() == callback_second.offsets_.size());
    Kokkos::parallel_for(
        "verify_callback",
        Kokkos::RangePolicy<ExecutionSpace>(0, reference_offset.size()),
        KOKKOS_LAMBDA(const int i) {
          const auto reference = reference_offset(i);
          const auto other = callback_second.offsets_(i);
          if (reference != other)
          {
            printf("reference_offsets(%d)=%d should be offsets(%d)=%d\n!", i,
                   reference, i, other);
            Kokkos::abort("Mismatch");
          }
        });

    assert(reference_indices.size() == callback_second.indices_.size());
    Kokkos::parallel_for(
        "verify_callback",
        Kokkos::RangePolicy<ExecutionSpace>(0, reference_indices.size()),
        KOKKOS_LAMBDA(const int i) {
          const auto reference = reference_indices(i);
          const auto other = callback_second.indices_(i);
          if (reference != other)
          {
            printf("reference_offsets(%d)=%d should be offsets(%d)=%d\n!", i,
                   reference, i, other);
            Kokkos::abort("Mismatch");
          }
        });*/
  }
}

class KokkosScopeGuard
{
public:
  KokkosScopeGuard(int &argc, char *argv[]) { Kokkos::initialize(argc, argv); }
  ~KokkosScopeGuard() { Kokkos::finalize(); }
};

template <typename TreeType>
void register_benchmark(std::string const &description, Spec const &spec)
{
  benchmark::RegisterBenchmark(
      spec.create_label_construction(description).c_str(),
      [=](benchmark::State &state) { BM_construction<TreeType>(state, spec); })
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark(
      spec.create_label_knn_search(description).c_str(),
      [=](benchmark::State &state) { BM_knn_search<TreeType>(state, spec); })
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
  if (description != "BoostRTree")
  {
    benchmark::RegisterBenchmark(
        spec.create_label_knn_search(description, "callback").c_str(),
        [=](benchmark::State &state) {
          BM_knn_callback_search<TreeType>(state, spec);
        })
        ->UseManualTime()
        ->Unit(benchmark::kMicrosecond);
  }
  benchmark::RegisterBenchmark(
      spec.create_label_radius_search(description).c_str(),
      [=](benchmark::State &state) { BM_radius_search<TreeType>(state, spec); })
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
  if (description != "BoostRTree")
  {
    benchmark::RegisterBenchmark(
        spec.create_label_radius_search(description, "callback").c_str(),
        [=](benchmark::State &state) {
          BM_radius_callback_search<TreeType>(state, spec);
        })
        ->UseManualTime()
        ->Unit(benchmark::kMicrosecond);
  }
}

// NOTE Motivation for this class that stores the argument count and values is
// I could not figure out how to make the parser consume arguments with
// Boost.Program_options
// Benchmark removes its own arguments from the command line arguments. This
// means, that by virtue of returning references to internal data members in
// argc() and argv() function, it will necessarily modify the members. It will
// decrease _argc, and "reduce" _argv data. Hence, we must keep a copy of _argv
// that is not modified from the outside to release memory in the destructor
// correctly.
class CmdLineArgs
{
private:
  int _argc;
  std::vector<char *> _argv;
  std::vector<char *> _owner_ptrs;

public:
  CmdLineArgs(std::vector<std::string> const &args, char const *exe)
      : _argc(args.size() + 1)
      , _owner_ptrs{new char[std::strlen(exe) + 1]}
  {
    std::strcpy(_owner_ptrs[0], exe);
    _owner_ptrs.reserve(_argc);
    for (auto const &s : args)
    {
      _owner_ptrs.push_back(new char[s.size() + 1]);
      std::strcpy(_owner_ptrs.back(), s.c_str());
    }
    _argv = _owner_ptrs;
  }

  ~CmdLineArgs()
  {
    for (auto *p : _owner_ptrs)
    {
      delete[] p;
    }
  }

  int &argc() { return _argc; }

  char **argv() { return _argv.data(); }
};

int main(int argc, char *argv[])
{
#ifdef ARBORX_PERFORMANCE_TESTING
  MPI_Init(&argc, &argv);
#endif
  Kokkos::initialize(argc, argv);

  namespace bpo = boost::program_options;
  bpo::options_description desc("Allowed options");
  Spec single_spec;
  std::string source_pt_cloud;
  std::string target_pt_cloud;
  std::vector<std::string> exact_specs;
  // clang-format off
    desc.add_options()
        ( "help", "produce help message" )
        ( "values", bpo::value<int>(&single_spec.n_values)->default_value(50000), "number of indexable values (source)" )
        ( "queries", bpo::value<int>(&single_spec.n_queries)->default_value(20000), "number of queries (target)" )
        ( "predicate-sort", bpo::value<bool>(&single_spec.sort_predicates)->default_value(true), "sort predicates" )
        ( "neighbors", bpo::value<int>(&single_spec.n_neighbors)->default_value(10), "desired number of results per query" )
        ( "buffer", bpo::value<int>(&single_spec.buffer_size)->default_value(0), "size for buffer optimization in radius search" )
        ( "source-point-cloud-type", bpo::value<std::string>(&source_pt_cloud)->default_value("filled_box"), "shape of the source point cloud"  )
        ( "target-point-cloud-type", bpo::value<std::string>(&target_pt_cloud)->default_value("filled_box"), "shape of the target point cloud"  )
        ( "exact-spec", bpo::value<std::vector<std::string>>(&exact_specs)->multitoken(), "exact specification (can be specified multiple times for batch)" )
    ;
  // clang-format on
  bpo::variables_map vm;
  bpo::parsed_options parsed = bpo::command_line_parser(argc, argv)
                                   .options(desc)
                                   .allow_unregistered()
                                   .run();
  bpo::store(parsed, vm);
  CmdLineArgs pass_further{
      bpo::collect_unrecognized(parsed.options, bpo::include_positional),
      argv[0]};
  bpo::notify(vm);

  std::cout << "ArborX version: " << ArborX::version() << std::endl;
  std::cout << "ArborX hash   : " << ArborX::gitCommitHash() << std::endl;

  if (vm.count("help") > 0)
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

  if (vm.count("exact-spec") > 0)
  {
    for (std::string option :
         {"values", "queries", "predicate-sort", "neighbors", "buffer",
          "source-point-cloud-type", "target-point-cloud-type"})
    {
      if (!vm[option].defaulted())
      {
        std::cout << "Conflicting options: 'exact-spec' and '" << option
                  << "', exiting..." << std::endl;
        return EXIT_FAILURE;
      }
    }
  }

  benchmark::Initialize(&pass_further.argc(), pass_further.argv());
  // Throw if some of the arguments have not been recognized.
  std::ignore =
      bpo::command_line_parser(pass_further.argc(), pass_further.argv())
          .options(bpo::options_description(""))
          .run();

  std::vector<Spec> specs;
  specs.reserve(exact_specs.size());
  for (auto const &spec_string : exact_specs)
    specs.push_back(create_spec_from_string(spec_string));

  if (vm.count("exact-spec") == 0)
  {
    single_spec.backends = "all";
    single_spec.source_point_cloud_type = to_point_cloud_enum(source_pt_cloud);
    single_spec.target_point_cloud_type = to_point_cloud_enum(target_pt_cloud);
    specs.push_back(single_spec);
  }

  for (auto const &spec : specs)
  {
#ifdef KOKKOS_ENABLE_SERIAL
    if (spec.backends == "all" || spec.backends == "serial")
      register_benchmark<ArborX::BVH<Kokkos::Serial::device_type>>(
          "ArborX::BVH<Serial>", spec);
#else
    if (spec.backends == "serial")
      throw std::runtime_error("Serial backend not available!");
#endif

#ifdef KOKKOS_ENABLE_OPENMP
    if (spec.backends == "all" || spec.backends == "openmp")
      register_benchmark<ArborX::BVH<Kokkos::OpenMP::device_type>>(
          "ArborX::BVH<OpenMP>", spec);
#else
    if (spec.backends == "openmp")
      throw std::runtime_error("OpenMP backend not available!");
#endif

#ifdef KOKKOS_ENABLE_THREADS
    if (spec.backends == "all" || spec.backends == "threads")
      register_benchmark<ArborX::BVH<Kokkos::Threads::device_type>>(
          "ArborX::BVH<Threads>", spec);
#else
    if (spec.backends == "threads")
      throw std::runtime_error("Threads backend not available!");
#endif

#ifdef KOKKOS_ENABLE_CUDA
    if (spec.backends == "all" || spec.backends == "cuda")
      register_benchmark<ArborX::BVH<Kokkos::Cuda::device_type>>(
          "ArborX::BVH<Cuda>", spec);
#else
    if (spec.backends == "cuda")
      throw std::runtime_error("CUDA backend not available!");
#endif

#ifdef KOKKOS_ENABLE_HIP
    if (spec.backends == "all" || spec.backends == "hip")
      register_benchmark<ArborX::BVH<Kokkos::Experimental::HIP::device_type>>(
          "ArborX::BVH<HIP>", spec);
#else
    if (spec.backends == "hip")
      throw std::runtime_error("HIP backend not available!");
#endif

#ifdef KOKKOS_ENABLE_SERIAL
/*    if (spec.backends == "all" || spec.backends == "rtree")
      register_benchmark<BoostExt::RTree<ArborX::Point>>("BoostRTree", spec);*/
#endif
  }

  benchmark::RunSpecifiedBenchmarks();

  Kokkos::finalize();
#ifdef ARBORX_PERFORMANCE_TESTING
  MPI_Finalize();
#endif

  return EXIT_SUCCESS;
}
