/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "ArborXTest_PairIndexRank.hpp"
#include "ArborXTest_StdVectorToKokkosView.hpp"
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include <ArborX_Box.hpp>
#include <ArborX_DistributedTree.hpp>
#include <ArborX_Ray.hpp>

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <vector>

#include "Search_UnitTestHelpers.hpp"
#include <mpi.h>

#define BOOST_TEST_MODULE DistributedTree

using ArborXTest::PairIndexRank;

struct PairRankIndex
{
  int rank;
  int index;

  friend bool operator==(PairRankIndex lhs, PairRankIndex rhs)
  {
    return lhs.index == rhs.index && lhs.rank == rhs.rank;
  }
  friend bool operator<(PairRankIndex lhs, PairRankIndex rhs)
  {
    return lhs.rank < rhs.rank ||
           (lhs.rank == rhs.rank && lhs.index < rhs.index);
  }
  friend std::ostream &operator<<(std::ostream &stream,
                                  PairRankIndex const &pair)
  {
    return stream << '[' << pair.rank << ',' << pair.index << ']';
  }
};

struct DistributedNearestCallback
{
  int rank;

  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const &, Value const &value,
                                  OutputFunctor const &out) const
  {
    out({rank, value.index});
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(hello_world_nearest, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  using Point = ArborX::Point<3, double>;

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
  for (int i = 0; i < n; ++i)
    points[i] = {{(double)i / n + comm_rank, 0., 0.}};

  auto tree = makeDistributedTree<DeviceType>(comm, ExecutionSpace{}, points);

  // 0---0---0---0---1---1---1---1---2---2---2---2---3---3---3---3---
  // |               |               |               |               |
  // |               |               |           x   x   x           |
  // |               |           x   x   x        <--0-->            |
  // |           x   x   x        <--1-->            |               |
  // x   x        <--2-->            |               |               |
  // 3-->            |               |               |               |
  // |               |               |               |               |
  Kokkos::View<ArborX::Nearest<Point> *, DeviceType> nearest_queries(
      "Testing::nearest_queries", 1);
  auto nearest_queries_host = Kokkos::create_mirror_view(nearest_queries);
  nearest_queries_host(0) =
      ArborX::nearest<Point>({{0.f + comm_size - 1 - comm_rank, 0., 0.}},
                             comm_rank < comm_size - 1 ? 3 : 2);
  deep_copy(nearest_queries, nearest_queries_host);

  std::vector<PairIndexRank> values;
  values.reserve(n + 1);
  for (int i = 0; i < n; ++i)
  {
    values.push_back({n - 1 - i, comm_size - 1 - comm_rank});
  }

  BOOST_TEST(n > 2);
  if (comm_rank < comm_size - 1)
  {
    ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree, nearest_queries,
                           make_reference_solution<PairIndexRank>(
                               {{0, comm_size - 1 - comm_rank},
                                {n - 1, comm_size - 2 - comm_rank},
                                {1, comm_size - 1 - comm_rank}},
                               {0, 3}));
  }
  else
  {
    ARBORX_TEST_QUERY_TREE(
        ExecutionSpace{}, tree, nearest_queries,
        make_reference_solution<PairIndexRank>(
            {{0, comm_size - 1 - comm_rank}, {1, comm_size - 1 - comm_rank}},
            {0, 2}));
  }

  // Now do the same with callbacks
  if (comm_rank < comm_size - 1)
  {
    ARBORX_TEST_QUERY_TREE_CALLBACK(
        ExecutionSpace{}, tree, nearest_queries,
        ArborX::Experimental::declare_callback_constrained(
            DistributedNearestCallback{comm_rank}),
        make_reference_solution<PairRankIndex>(
            {{comm_size - 1 - comm_rank, 0},
             {comm_size - 2 - comm_rank, n - 1},
             {comm_size - 1 - comm_rank, 1}},
            {0, 3}));
  }
  else
  {
    ARBORX_TEST_QUERY_TREE_CALLBACK(
        ExecutionSpace{}, tree, nearest_queries,
        ArborX::Experimental::declare_callback_constrained(
            DistributedNearestCallback{comm_rank}),
        make_reference_solution<PairRankIndex>(
            {{comm_size - 1 - comm_rank, 0}, {comm_size - 1 - comm_rank, 1}},
            {0, 2}));
  }
}

// FIXME: Almost identical to hellow_world_nearest, but uses double. Testing
// needs refactoring.
BOOST_AUTO_TEST_CASE_TEMPLATE(double_tree, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  using Point = ArborX::Point<3, double>;

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
  for (int i = 0; i < n; ++i)
    points[i] = {{(double)i / n + comm_rank, 0., 0.}};

  auto tree = makeDistributedTree<DeviceType>(comm, ExecutionSpace{}, points);

  // 0---0---0---0---1---1---1---1---2---2---2---2---3---3---3---3---
  // |               |               |               |               |
  // |               |               |           x   x   x           |
  // |               |           x   x   x        <--0-->            |
  // |           x   x   x        <--1-->            |               |
  // x   x        <--2-->            |               |               |
  // 3-->            |               |               |               |
  // |               |               |               |               |
  Kokkos::View<ArborX::Nearest<Point> *, DeviceType> nearest_queries(
      "Testing::nearest_queries", 1);
  auto nearest_queries_host = Kokkos::create_mirror_view(nearest_queries);
  nearest_queries_host(0) =
      ArborX::nearest<Point>({{0.f + comm_size - 1 - comm_rank, 0., 0.}},
                             comm_rank < comm_size - 1 ? 3 : 2);
  deep_copy(nearest_queries, nearest_queries_host);

  std::vector<PairIndexRank> values;
  values.reserve(n + 1);
  for (int i = 0; i < n; ++i)
    values.push_back({n - 1 - i, comm_size - 1 - comm_rank});

  BOOST_TEST(n > 2);
  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree, nearest_queries,
                         (comm_rank < comm_size - 1
                              ? make_reference_solution<PairIndexRank>(
                                    {{0, comm_size - 1 - comm_rank},
                                     {n - 1, comm_size - 2 - comm_rank},
                                     {1, comm_size - 1 - comm_rank}},
                                    {0, 3})
                              : make_reference_solution<PairIndexRank>(
                                    {{0, comm_size - 1 - comm_rank},
                                     {1, comm_size - 1 - comm_rank}},
                                    {0, 2})));
}

#if 0
BOOST_AUTO_TEST_CASE_TEMPLATE(empty_tree_nearest, DeviceType,
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
                           makeNearestQueries<DeviceType>({}),
                           make_reference_solution<PairIndexRank>({}, {0}));

    // All ranks but rank 0 have a single query with a nearest predicate
    if (comm_rank == 0)
    {
      ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                             makeNearestQueries<DeviceType>({}),
                             make_reference_solution<PairIndexRank>({}, {0}));
    }
    else
    {
      ARBORX_TEST_QUERY_TREE(
          ExecutionSpace{}, tree,
          makeNearestQueries<DeviceType>({
              {{{0., 0., 0.}}, comm_rank},
          }),
          make_reference_solution<PairIndexRank>({}, {0, 0}));
    }
  }
}
#endif

BOOST_AUTO_TEST_CASE_TEMPLATE(unique_leaf_on_rank_0_nearest, DeviceType,
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
                         (makeNearestQueries<DeviceType, Point>({})),
                         make_reference_solution<PairIndexRank>({}, {0}));

  // Querying for more neighbors than there are leaves in the tree
  ARBORX_TEST_QUERY_TREE(
      ExecutionSpace{}, tree,
      (makeNearestQueries<DeviceType, Point>({
          {{{(float)comm_rank, (float)comm_rank, (float)comm_rank}}, comm_size},
      })),
      make_reference_solution<PairIndexRank>({{0, 0}}, {0, 1}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(one_leaf_per_rank_nearest, DeviceType,
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

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         (makeNearestQueries<DeviceType, Point>({})),
                         make_reference_solution<PairIndexRank>({}, {0}));

  if (comm_rank > 0)
  {
    ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                           (makeNearestQueries<DeviceType, Point>({
                               {{{0., 0., 0.}}, comm_rank * comm_size},
                           })),
                           make_reference_solution(
                               [comm_size]() {
                                 std::vector<PairIndexRank> values;
                                 values.reserve(comm_size);
                                 for (int i = 0; i < comm_size; ++i)
                                   values.push_back({0, i});
                                 return values;
                               }(),
                               {0, comm_size}));
  }
  else
  {
    ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                           (makeNearestQueries<DeviceType, Point>({
                               {{{0., 0., 0.}}, comm_rank * comm_size},
                           })),
                           make_reference_solution<PairIndexRank>({}, {0, 0}));
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(do_not_exceed_capacity, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  // This unit tests exposes bug that essentially assumed the number of
  // neighbors queried for would not exceed the maximum size of the default
  // underlying container for a priority queue which was 256.
  // https://github.com/arborx/ArborX/pull/126#issuecomment-538410096
  // Each rank has the exact same cloud and perform a knn query that will push
  // comm_size * 512 elements into the queue.
  using ArborX::Box;
  using ArborX::nearest;
  using Point = ArborX::Point<3>;
  using ExecutionSpace = typename DeviceType::execution_space;
  MPI_Comm comm = MPI_COMM_WORLD;
  std::vector<Point> points(512);
  for (int i = 0; i < (int)points.size(); ++i)
    points[i] = {{(float)i, (float)i, (float)i}};

  auto const tree =
      makeDistributedTree<DeviceType>(comm, ExecutionSpace{}, points);
  Kokkos::View<decltype(nearest(Point{})) *, DeviceType> queries(
      "Testing::queries", 1);
  Kokkos::deep_copy(queries, nearest(Point{0, 0, 0}, 512));
  Kokkos::View<PairIndexRank *, DeviceType> values("Testing::values", 0);
  Kokkos::View<int *, DeviceType> offset("Testing::offset", 0);
  BOOST_CHECK_NO_THROW(tree.query(ExecutionSpace{}, queries, values, offset));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(non_approximate_nearest_neighbors, DeviceType,
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
  auto const tree = makeDistributedTree<DeviceType, Box>(
      comm, ExecutionSpace{},
      {
          {{{(float)comm_rank, 0., 0.}}, {{(float)comm_rank, 0., 0.}}},
          {{{(float)comm_rank + 1, 1., 1.}}, {{(float)comm_rank + 1, 1., 1.}}},
      });

  BOOST_TEST(!tree.empty());
  BOOST_TEST((int)tree.size() == 2 * comm_size);

  //  +----------0----------1----------2----------3
  //  |          |          |          |          |
  //  |          |          |          |          |
  //  |          |          |          |          |
  //  |          |          |          |          |
  //  0-------x--1-------X--2-------X--3-------X--+
  //          ^          ^          ^          ^
  //          3          2          1          0
  ARBORX_TEST_QUERY_TREE(
      ExecutionSpace{}, tree,
      (makeNearestQueries<DeviceType, Point>({
          {{{(float)(comm_size - 1 - comm_rank) + .75f, 0., 0.}}, 1},
      })),
      make_reference_solution<PairIndexRank>(
          {{0, comm_size - comm_rank - (comm_rank == 0)}}, {0, 1}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(non_approximate_box_nearest_neighbors, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
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
  auto const tree = makeDistributedTree<DeviceType, Box>(
      comm, ExecutionSpace{},
      {
          {{{(float)comm_rank, 0., 0.}}, {{(float)comm_rank, 0., 0.}}},
          {{{(float)comm_rank + 1, 1., 1.}}, {{(float)comm_rank + 1, 1., 1.}}},
      });

  BOOST_TEST(!tree.empty());
  BOOST_TEST((int)tree.size() == 2 * comm_size);

  //  +----------0----------1----------2----------3
  //  |          |          |          |          |
  //  |          |          |          |          |
  //  |          |          |          |          |
  //  |          |          |          |          |
  //  0-------x--1-------X--2-------X--3-------X--+
  //          ^          ^          ^          ^
  //          3          2          1          0
  ARBORX_TEST_QUERY_TREE(
      ExecutionSpace{}, tree,
      (makeNearestQueries<DeviceType, Box>({
          {{{(float)(comm_size - 1 - comm_rank) + .65f, 0., 0.},
            {(float)(comm_size - 1 - comm_rank) + .85f, 0., 0.}},
           1},
      })),
      make_reference_solution<PairIndexRank>(
          {{0, comm_size - comm_rank - (comm_rank == 0)}}, {0, 1}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(non_approximate_sphere_nearest_neighbors,
                              DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using Box = ArborX::Box<3>;
  using Sphere = ArborX::Sphere<3>;

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
  auto const tree = makeDistributedTree<DeviceType, Box>(
      comm, ExecutionSpace{},
      {
          {{{(float)comm_rank, 0., 0.}}, {{(float)comm_rank, 0., 0.}}},
          {{{(float)comm_rank + 1, 1., 1.}}, {{(float)comm_rank + 1, 1., 1.}}},
      });

  BOOST_TEST(!tree.empty());
  BOOST_TEST((int)tree.size() == 2 * comm_size);

  //  +----------0----------1----------2----------3
  //  |          |          |          |          |
  //  |          |          |          |          |
  //  |          |          |          |          |
  //  |          |          |          |          |
  //  0-------x--1-------X--2-------X--3-------X--+
  //          ^          ^          ^          ^
  //          3          2          1          0
  ARBORX_TEST_QUERY_TREE(
      ExecutionSpace{}, tree,
      (makeNearestQueries<DeviceType, Sphere>({{
          {{(float)(comm_size - 1 - comm_rank) + .75f, 0., 0.}, 0.1},
          1,
      }})),
      make_reference_solution<PairIndexRank>(
          {{0, comm_size - comm_rank - (comm_rank == 0)}}, {0, 1}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(distributed_ray, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;

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
      {
          {{{(float)comm_rank, 0., 0.}}, {{(float)comm_rank + 1, 1., 1.}}},
      });

  std::vector<ArborX::Experimental::Ray<float>> rays = {
      {{comm_rank + 0.5f, -0.5f, 0.5f}, {0.f, 1.f, 0.f}},
      {{-0.5f, 0.5f, 0.5f}, {1.f, 0.f, 0.f}},
  };

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         ArborX::Experimental::make_nearest(
                             ArborXTest::toView<MemorySpace>(rays), 1),
                         make_reference_solution<PairIndexRank>(
                             {{0, comm_rank}, {0, 0}}, {0, 1, 2}));
}
