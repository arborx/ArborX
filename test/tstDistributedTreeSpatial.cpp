/****************************************************************************
 * Copyright (c) 2017-2023 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "ArborXTest_Cloud.hpp"
#include "ArborX_BoostRTreeHelpers.hpp"
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include <ArborX_DistributedTree.hpp>

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <random>

#include "Search_UnitTestHelpers.hpp"
#include <mpi.h>

#define BOOST_TEST_MODULE DistributedTree

namespace tt = boost::test_tools;

using ArborXTest::PairIndexRank;

BOOST_AUTO_TEST_CASE_TEMPLATE(hello_world_spatial, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using Point = ArborX::Point<3>;
  using Sphere = ArborX::Sphere<3>;

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  int const n = 4;
  std::vector<Point> points(n);
  // [  rank 0       [  rank 1       [  rank 2       [  rank 3       [
  // x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---
  // ^   ^   ^   ^
  // 0   1   2   3   ^   ^   ^   ^
  //                 0   1   2   3   ^   ^   ^   ^
  //                                 0   1   2   3   ^   ^   ^   ^
  //                                                 0   1   2   3
  for (int i = 0; i < (int)points.size(); ++i)
    points[i] = {{(float)i / n + comm_rank, 0., 0.}};

  auto const tree =
      makeDistributedTree<DeviceType>(comm, ExecutionSpace{}, points);

  // 0---0---0---0---1---1---1---1---2---2---2---2---3---3---3---3---
  // |               |               |               |               |
  // |               |               |               x   x   x   x   |
  // |               |               |               |<------0------>|
  // |               |               x   x   x   x   x               |
  // |               |               |<------1------>|               |
  // |               x   x   x   x   x               |               |
  // |               |<------2------>|               |               |
  // x   x   x   x   x               |               |               |
  // |<------3------>|               |               |               |
  // |               |               |               |               |
  Kokkos::View<decltype(ArborX::intersects(Sphere{})) *, DeviceType> queries(
      "Testing::queries", 1);
  auto queries_host = Kokkos::create_mirror_view(queries);
  queries_host(0) = ArborX::intersects(
      Sphere{{{0.5f + comm_size - 1 - comm_rank, 0., 0.}}, 0.5});
  deep_copy(queries, queries_host);

  std::vector<PairIndexRank> values;
  values.reserve(n + 1);
  for (int i = 0; i < n; ++i)
    values.push_back({n - 1 - i, comm_size - 1 - comm_rank});

  if (comm_rank > 0)
  {
    values.push_back({0, comm_size - comm_rank});
    ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree, queries,
                           make_reference_solution(values, {0, n + 1}));
  }
  else
  {
    ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree, queries,
                           make_reference_solution(values, {0, n}));
  }
}

#if 0
BOOST_AUTO_TEST_CASE_TEMPLATE(empty_tree_spatial, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using Tree = ArborX::DistributedTree<typename DeviceType::memory_space>;
  using ExecutionSpace = typename DeviceType::execution_space;

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  Tree default_initialized;
  Tree value_initialized{};
  for (auto const &tree : {
           default_initialized, value_initialized,
           makeDistributedTree<DeviceType, ArborX::Box<3>>(
               comm, ExecutionSpace{},
               {}) // constructed with empty view of boxes
       })
  {

    BOOST_TEST(tree.empty());
    BOOST_TEST(tree.size() == 0);

    BOOST_TEST(ArborX::Details::equals(tree.bounds(), {}));

    ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                           makeIntersectsQueries<DeviceType>({}),
                           make_reference_solution<PairIndexRank>({}, {0}));

    ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                           makeIntersectsQueries<DeviceType>({}),
                           make_reference_solution<PairIndexRank>({}, {0}));

    // Only rank 0 has a couple spatial queries with a spatial predicate
    if (comm_rank == 0)
    {
      ARBORX_TEST_QUERY_TREE(
          ExecutionSpace{}, tree,
          makeIntersectsQueries<DeviceType>({{}, {}}),
          make_reference_solution<PairIndexRank>({}, {0, 0, 0}));
    }
    else
    {
      ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                             makeIntersectsQueries<DeviceType>({}),
                             make_reference_solution<PairIndexRank>({}, {0}));
    }

    // All ranks but rank 0 have a single query with a spatial predicate
    if (comm_rank == 0)
    {
      ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                             makeIntersectsQueries<DeviceType>({}),
                             make_reference_solution<PairIndexRank>({}, {0}));
    }
    else
    {
      ARBORX_TEST_QUERY_TREE(
          ExecutionSpace{}, tree,
          makeIntersectsQueries<DeviceType>({
              {{{(float)comm_rank, 0.f, 0.f}}, (float)comm_size},
          }),
          make_reference_solution<PairIndexRank>({}, {0, 0}));
    }
  }
}
#endif

BOOST_AUTO_TEST_CASE_TEMPLATE(unique_leaf_on_rank_0_spatial, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using Box = ArborX::Box<3>;
  using Sphere = ArborX::Sphere<3>;

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  // tree has one unique leaf that lives on rank 0
  auto const tree =
      (comm_rank == 0
           ? makeDistributedTree<DeviceType, Box>(
                 comm, ExecutionSpace{},
                 {
                     {{{0., 0., 0.}}, {{1., 1., 1.}}},
                 })
           : makeDistributedTree<DeviceType, Box>(comm, ExecutionSpace{}, {}));

  BOOST_TEST(!tree.empty());
  BOOST_TEST(tree.size() == 1);

  BOOST_TEST(
      ArborX::Details::equals(tree.bounds(), {{{0., 0., 0.}}, {{1., 1., 1.}}}));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         (makeIntersectsQueries<DeviceType, Box>({})),
                         make_reference_solution<PairIndexRank>({}, {0}));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         (makeIntersectsQueries<DeviceType, Sphere>({})),
                         make_reference_solution<PairIndexRank>({}, {0}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(one_leaf_per_rank_spatial, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using Box = ArborX::Box<3>;

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  // tree has one leaf per rank
  auto const tree = makeDistributedTree<DeviceType, Box>(
      comm, ExecutionSpace{},
      {
          {{{(float)comm_rank, 0., 0.}}, {{(float)comm_rank + 1, 1., 1.}}},
      });

  BOOST_TEST(!tree.empty());
  BOOST_TEST((int)tree.size() == comm_size);

  BOOST_TEST(ArborX::Details::equals(
      tree.bounds(), {{{0., 0., 0.}}, {{(float)comm_size, 1., 1.}}}));

  ARBORX_TEST_QUERY_TREE(
      ExecutionSpace{}, tree,
      (makeIntersectsQueries<DeviceType, Box>({
          {{{(float)comm_size - (float)comm_rank - .5f, .5, .5}},
           {{(float)comm_size - (float)comm_rank - .5f, .5, .5}}},
          {{{(float)comm_rank + .5f, .5, .5}},
           {{(float)comm_rank + .5f, .5, .5}}},
      })),
      make_reference_solution<PairIndexRank>(
          {{0, comm_size - 1 - comm_rank}, {0, comm_rank}}, {0, 1, 2}));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         (makeIntersectsQueries<DeviceType, Box>({})),
                         make_reference_solution<PairIndexRank>({}, {0}));
}

template <typename DeviceType>
struct CustomInlineCallbackWithAttachment
{
  using Point = ArborX::Point<3>;

  Kokkos::View<Point *, DeviceType> points;
  Point const origin = {{0., 0., 0.}};

  template <typename Query, typename Insert>
  KOKKOS_FUNCTION void operator()(Query const &query, PairIndexRank pair,
                                  Insert const &insert) const
  {
    auto data = ArborX::getData(query);
    float const distance_to_origin =
        ArborX::Details::distance(points(pair.index), origin);

    insert(distance_to_origin + data);
  }
};

template <typename DeviceType>
struct CustomPostCallbackWithAttachment
{
  using tag = ArborX::Details::PostCallbackTag;
  using Point = ArborX::Point<3>;

  Kokkos::View<Point *, DeviceType> points;
  Point const origin = {{0., 0., 0.}};

  template <typename Predicates, typename InOutView, typename InView,
            typename OutView>
  void operator()(Predicates const &queries, InOutView &offset, InView in,
                  OutView &out) const
  {
    using ExecutionSpace = typename DeviceType::execution_space;
    using ArborX::Details::distance;
    auto const n = offset.extent(0) - 1;
    Kokkos::realloc(out, in.extent(0));
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_CLASS_LAMBDA(int i) {
          auto data = ArborX::getData(queries(i));
          for (int j = offset(i); j < offset(i + 1); ++j)
            out(j) = distance(points(in(j).index), origin) + data;
        });
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(callback_with_attachment, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using Point = ArborX::Point<3>;
  using Box = ArborX::Box<3>;

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  //  +----------0----------1----------2----------3
  //  |          |          |          |          |
  //  |          |          |          |          |
  //  |          |          |          |          |
  //  |          |          |          |          |
  //  0----------1----------2----------3----------+
  //  [  rank 0  ]
  //             [  rank 1  ]
  //                        [  rank 2  ]
  //                                   [  rank 3  ]
  auto const tree = makeDistributedTree<DeviceType, ArborX::Box<3>>(
      comm, ExecutionSpace{},
      {{{{(float)comm_rank, 0., 0.}}, {{(float)comm_rank + 1, 1., 1.}}}});

  //  +--------0---------1----------2---------3
  //  |        |         |          |         |
  //  |        |         |          |         |
  //  |        |         |          |         |
  //  |        |         |          |         |
  //  0--------1----x----2-----x----3----x----+
  //                ^          ^         ^
  //                0          1         2

  int const n_queries = 1;
  using ExecutionSpace = typename DeviceType::execution_space;
  Kokkos::View<Point *, DeviceType> points("Testing::points", n_queries);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n_queries), KOKKOS_LAMBDA(int i) {
        points(i) = {(float)(comm_rank) + 1.5f, 0.f, 0.f};
      });
  auto points_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, points);

  // This is a bit tricky. One would assume that it should be
  // distance(points(i), origin) + comm_rank, matching code inside the
  // callback. However, the callbacks are initialized and called on the rank
  // that produces results, not on the rank that sets up the queries. In this
  // example, for point 0 the callback will be called on rank 1, which is
  // initialized with point 1. So, within the callback, points(0) corresponds
  // to point 1 physically, and not point 0, even though the query() call is
  // called on rank 0.
  int const n_results = (comm_rank < comm_size - 1) ? 1 : 0;
  Point const origin = {{0., 0., 0.}};
  Kokkos::View<float *, DeviceType> ref("Testing::ref", n_results);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n_results), KOKKOS_LAMBDA(int i) {
        ref(i) =
            float(ArborX::Details::distance(points(i), origin) + 1) + comm_rank;
      });

  {
    Kokkos::View<float *, DeviceType> custom("Testing::custom", 0);
    Kokkos::View<int *, DeviceType> offset("Testing::offset", 0);
    tree.query(ExecutionSpace{},
               makeIntersectsWithAttachmentQueries<DeviceType, Box, int>(
                   {{points_host(0), points_host(0)}}, {comm_rank}),
               CustomInlineCallbackWithAttachment<DeviceType>{points}, custom,
               offset);

    auto custom_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, custom);
    auto ref_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ref);
    auto offset_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offset);
    BOOST_TEST(make_compressed_storage(offset_host, custom_host) ==
                   make_compressed_storage(offset_host, ref_host),
               tt::per_element());
  }
  {
    Kokkos::View<float *, DeviceType> custom("Testing::custom", 0);
    Kokkos::View<int *, DeviceType> offset("Testing::offset", 0);
    tree.query(ExecutionSpace{},
               makeIntersectsWithAttachmentQueries<DeviceType, Box, int>(
                   {{points_host(0), points_host(0)}}, {comm_rank}),
               CustomPostCallbackWithAttachment<DeviceType>{points}, custom,
               offset);

    auto custom_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, custom);
    auto ref_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ref);
    auto offset_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offset);
    BOOST_TEST(make_compressed_storage(offset_host, custom_host) ==
                   make_compressed_storage(offset_host, ref_host),
               tt::per_element());
  }
}

template <typename DeviceType>
struct CustomPureInlineCallback
{
  Kokkos::View<int *, DeviceType> counts;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &, PairIndexRank pair) const
  {
    Kokkos::atomic_inc(&counts(pair.index));
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(pure_spatial_callback, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using Point = ArborX::Point<3>;

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  //  +----------0----------1----------2----------3
  //  |          |          |          |          |
  //  |          |          |          |          |
  //  |          |          |          |          |
  //  |          |          |          |          |
  //  0----------1----------2----------3----------+
  //  [  rank 0  ]
  //             [  rank 1  ]
  //                        [  rank 2  ]
  //                                   [  rank 3  ]
  auto const tree = makeDistributedTree<DeviceType, ArborX::Box<3>>(
      comm, ExecutionSpace{},
      {{{{(float)comm_rank, 0., 0.}}, {{(float)comm_rank + 1, 1., 1.}}}});

  //  +--------0---------1----------2---------3
  //  |        |         |          |         |
  //  |        |         |          |         |
  //  |        |         |          |         |
  //  |        |         |          |         |
  //  0--------1----x----2-----x----3----x----+    x
  //                ^          ^         ^         ^
  //                0          1         2         3
  Kokkos::View<decltype(ArborX::intersects(Point{})) *, DeviceType> queries(
      "Testing::queries", 1);
  auto queries_host = Kokkos::create_mirror_view(queries);
  queries_host(0) = ArborX::intersects(Point{1.5f + comm_rank, 0, 0});
  deep_copy(queries, queries_host);

  Kokkos::View<int *, DeviceType> counts("Testing::counts", queries.size());
  tree.query(ExecutionSpace{}, queries,
             CustomPureInlineCallback<DeviceType>{counts});
  auto counts_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, counts);

  std::vector<int> counts_ref;
  counts_ref.push_back(comm_rank > 0 ? 1 : 0);

  BOOST_TEST(counts_host == counts_ref, tt::per_element());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(boost_comparison, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using Point = ArborX::Point<3>;
  using Box = ArborX::Box<3>;
  using Sphere = ArborX::Sphere<3>;

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  // Construct a random cloud of point. We use the same seed on all the
  // processors.
  double const Lx = 10.0;
  double const Ly = 10.0;
  double const Lz = 10.0;
  int const n = 100;
  auto cloud = ArborXTest::make_random_cloud<Point>(
      Kokkos::DefaultHostExecutionSpace{}, n, Lx, Ly, Lz, 0);
  auto queries = ArborXTest::make_random_cloud<Point>(
      Kokkos::DefaultHostExecutionSpace{}, n, Lx, Ly, Lz, 1234);

  // The formula is a bit complicated but it does not require n be divisible
  // by comm_size
  int const local_n = (n + comm_size - 1 - comm_rank) / comm_size;
  std::vector<Box> bounding_boxes(local_n);
  for (int i = 0; i < n; ++i)
  {
    if (i % comm_size == comm_rank)
    {
      auto const &point = cloud(i);
      bounding_boxes[i / comm_size] = {point, point};
    }
  }

  // Initialize the distributed search tree
  auto const distributed_tree =
      makeDistributedTree<DeviceType>(comm, ExecutionSpace{}, bounding_boxes);

  // make queries
  Kokkos::View<float *[3], ExecutionSpace> point_coords("Testing::point_coords",
                                                        local_n);
  auto point_coords_host = Kokkos::create_mirror_view(point_coords);
  Kokkos::View<float *, ExecutionSpace> radii("Testing::radii", local_n);
  auto radii_host = Kokkos::create_mirror_view(radii);
  std::default_random_engine generator(0);
  std::uniform_real_distribution<float> distribution_radius(
      0.0, std::sqrt(Lx * Lx + Ly * Ly + Lz * Lz));
  std::uniform_int_distribution<int> distribution_k(1, std::floor(sqrt(n * n)));
  for (int i = 0; i < n; ++i)
  {
    if (i % comm_size == comm_rank)
    {
      auto const &point = queries(i);
      int const j = i / comm_size;
      radii_host(j) = distribution_radius(generator);

      point_coords_host(j, 0) = point[0];
      point_coords_host(j, 1) = point[1];
      point_coords_host(j, 2) = point[2];
    }
  }
  Kokkos::deep_copy(point_coords, point_coords_host);
  Kokkos::deep_copy(radii, radii_host);

  Kokkos::View<decltype(ArborX::intersects(Sphere{})) *, DeviceType>
      within_queries("Testing::within_queries", local_n);
  Kokkos::parallel_for(
      "register_within_queries",
      Kokkos::RangePolicy<ExecutionSpace>(0, local_n), KOKKOS_LAMBDA(int i) {
        within_queries(i) = ArborX::intersects(Sphere{
            {{point_coords(i, 0), point_coords(i, 1), point_coords(i, 2)}},
            radii(i)});
      });

  auto within_queries_host = Kokkos::create_mirror_view(within_queries);
  Kokkos::deep_copy(within_queries_host, within_queries);

  BoostExt::ParallelRTree<Box> rtree(
      comm, ExecutionSpace{},
      Kokkos::View<Box *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
          bounding_boxes.data(), bounding_boxes.size()));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, distributed_tree, within_queries,
                         query(ExecutionSpace{}, rtree, within_queries_host));
}
