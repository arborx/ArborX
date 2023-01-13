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

#include "ArborXTest_StdVectorToKokkosView.hpp"
#include "ArborX_BoostRTreeHelpers.hpp"
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include <ArborX_DistributedTree.hpp>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <iostream>
#include <random>
#include <tuple>

#include "Search_UnitTestHelpers.hpp"
#include <mpi.h>

#define BOOST_TEST_MODULE DistributedTree

namespace tt = boost::test_tools;

using ArborX::PairIndexRank;
using ArborX::Details::TupleIndexRankDistance;

BOOST_AUTO_TEST_CASE_TEMPLATE(hello_world, DeviceType, ARBORX_DEVICE_TYPES)
{
  using Tree = ArborX::DistributedTree<typename DeviceType::memory_space>;
  using ExecutionSpace = typename DeviceType::execution_space;

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  int const n = 4;
  Kokkos::View<ArborX::Point *, DeviceType> points("Testing::points", n);
  // [  rank 0       [  rank 1       [  rank 2       [  rank 3       [
  // x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---
  // ^   ^   ^   ^
  // 0   1   2   3   ^   ^   ^   ^
  //                 0   1   2   3   ^   ^   ^   ^
  //                                 0   1   2   3   ^   ^   ^   ^
  //                                                 0   1   2   3
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
        points(i) = {{(double)i / n + comm_rank, 0., 0.}};
      });

  Tree tree(comm, ExecutionSpace{}, points);

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
  Kokkos::View<decltype(ArborX::intersects(ArborX::Sphere{})) *, DeviceType>
      queries("Testing::queries", 1);
  auto queries_host = Kokkos::create_mirror_view(queries);
  queries_host(0) = ArborX::intersects(
      ArborX::Sphere{{{0.5 + comm_size - 1 - comm_rank, 0., 0.}}, 0.5});
  deep_copy(queries, queries_host);

  // 0---0---0---0---1---1---1---1---2---2---2---2---3---3---3---3---
  // |               |               |               |               |
  // |               |               |           x   x   x           |
  // |               |           x   x   x        <--0-->            |
  // |           x   x   x        <--1-->            |               |
  // x   x        <--2-->            |               |               |
  // 3-->            |               |               |               |
  // |               |               |               |               |
  Kokkos::View<ArborX::Nearest<ArborX::Point> *, DeviceType> nearest_queries(
      "Testing::nearest_queries", 1);
  auto nearest_queries_host = Kokkos::create_mirror_view(nearest_queries);
  nearest_queries_host(0) = ArborX::nearest<ArborX::Point>(
      {{0.0 + comm_size - 1 - comm_rank, 0., 0.}},
      comm_rank < comm_size - 1 ? 3 : 2);
  deep_copy(nearest_queries, nearest_queries_host);

  std::vector<PairIndexRank> values;
  values.reserve(n + 1);
  for (int i = 0; i < n; ++i)
  {
    values.push_back({n - 1 - i, comm_size - 1 - comm_rank});
  }
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
}

BOOST_AUTO_TEST_CASE_TEMPLATE(empty_tree, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  auto const tree = makeDistributedTree<DeviceType>(comm, {});

  BOOST_TEST(tree.empty());
  BOOST_TEST(tree.size() == 0);

  BOOST_TEST(ArborX::Details::equals(tree.bounds(), {}));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeIntersectsBoxQueries<DeviceType>({}),
                         make_reference_solution<PairIndexRank>({}, {0}));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeIntersectsSphereQueries<DeviceType>({}),
                         make_reference_solution<PairIndexRank>({}, {0}));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeNearestQueries<DeviceType>({}),
                         make_reference_solution<PairIndexRank>({}, {0}));

  ARBORX_TEST_QUERY_TREE_WITH_DISTANCE(
      ExecutionSpace{}, tree, makeNearestQueries<DeviceType>({}),
      make_reference_solution<TupleIndexRankDistance>({}, {0}));

  // Only rank 0 has a couple spatial queries with a spatial predicate
  if (comm_rank == 0)
  {
    ARBORX_TEST_QUERY_TREE(
        ExecutionSpace{}, tree, makeIntersectsBoxQueries<DeviceType>({{}, {}}),
        make_reference_solution<PairIndexRank>({}, {0, 0, 0}));
  }
  else
  {
    ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                           makeIntersectsBoxQueries<DeviceType>({}),
                           make_reference_solution<PairIndexRank>({}, {0}));
  }

  // All ranks but rank 0 have a single query with a spatial predicate
  if (comm_rank == 0)
  {
    ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                           makeIntersectsSphereQueries<DeviceType>({}),
                           make_reference_solution<PairIndexRank>({}, {0}));
  }
  else
  {
    ARBORX_TEST_QUERY_TREE(
        ExecutionSpace{}, tree,
        makeIntersectsSphereQueries<DeviceType>({
            {{{(float)comm_rank, 0.f, 0.f}}, (float)comm_size},
        }),
        make_reference_solution<PairIndexRank>({}, {0, 0}));
  }

  // All ranks but rank 0 have a single query with a nearest predicate
  if (comm_rank == 0)
  {
    ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                           makeNearestQueries<DeviceType>({}),
                           make_reference_solution<PairIndexRank>({}, {0}));
  }
  else
  {
    ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                           makeNearestQueries<DeviceType>({
                               {{{0., 0., 0.}}, comm_rank},
                           }),
                           make_reference_solution<PairIndexRank>({}, {0, 0}));
  }

  // All ranks have a single query with a nearest predicate (this version
  // returns distances as well)
  ARBORX_TEST_QUERY_TREE_WITH_DISTANCE(
      ExecutionSpace{}, tree,
      makeNearestQueries<DeviceType>({
          {{{0., 0., 0.}}, comm_size},
      }),
      make_reference_solution<TupleIndexRankDistance>({}, {0, 0}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(unique_leaf_on_rank_0, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  // tree has one unique leaf that lives on rank 0
  auto const tree =
      (comm_rank == 0 ? makeDistributedTree<DeviceType>(
                            comm,
                            {
                                {{{0., 0., 0.}}, {{1., 1., 1.}}},
                            })
                      : makeDistributedTree<DeviceType>(comm, {}));

  BOOST_TEST(!tree.empty());
  BOOST_TEST(tree.size() == 1);

  BOOST_TEST(
      ArborX::Details::equals(tree.bounds(), {{{0., 0., 0.}}, {{1., 1., 1.}}}));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeIntersectsBoxQueries<DeviceType>({}),
                         make_reference_solution<PairIndexRank>({}, {0}));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeIntersectsSphereQueries<DeviceType>({}),
                         make_reference_solution<PairIndexRank>({}, {0}));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeNearestQueries<DeviceType>({}),
                         make_reference_solution<PairIndexRank>({}, {0}));

  ARBORX_TEST_QUERY_TREE_WITH_DISTANCE(
      ExecutionSpace{}, tree, makeNearestQueries<DeviceType>({}),
      make_reference_solution<TupleIndexRankDistance>({}, {0}));

  // Querying for more neighbors than there are leaves in the tree
  ARBORX_TEST_QUERY_TREE(
      ExecutionSpace{}, tree,
      makeNearestQueries<DeviceType>({
          {{{(double)comm_rank, (double)comm_rank, (double)comm_rank}},
           comm_size},
      }),
      make_reference_solution<PairIndexRank>({{0, 0}}, {0, 1}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(one_leaf_per_rank, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  // tree has one leaf per rank
  auto const tree = makeDistributedTree<DeviceType>(
      comm,
      {
          {{{(double)comm_rank, 0., 0.}}, {{(double)comm_rank + 1., 1., 1.}}},
      });

  BOOST_TEST(!tree.empty());
  BOOST_TEST((int)tree.size() == comm_size);

  BOOST_TEST(ArborX::Details::equals(
      tree.bounds(), {{{0., 0., 0.}}, {{(double)comm_size, 1., 1.}}}));

  ARBORX_TEST_QUERY_TREE(
      ExecutionSpace{}, tree,
      makeIntersectsBoxQueries<DeviceType>({
          {{{(double)comm_size - (double)comm_rank - .5, .5, .5}},
           {{(double)comm_size - (double)comm_rank - .5, .5, .5}}},
          {{{(double)comm_rank + .5, .5, .5}},
           {{(double)comm_rank + .5, .5, .5}}},
      }),
      make_reference_solution<PairIndexRank>(
          {{0, comm_size - 1 - comm_rank}, {0, comm_rank}}, {0, 1, 2}));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeNearestQueries<DeviceType>({}),
                         make_reference_solution<PairIndexRank>({}, {0}));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeIntersectsBoxQueries<DeviceType>({}),
                         make_reference_solution<PairIndexRank>({}, {0}));

  if (comm_rank > 0)
  {
    ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                           makeNearestQueries<DeviceType>({
                               {{{0., 0., 0.}}, comm_rank * comm_size},
                           }),
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
                           makeNearestQueries<DeviceType>({
                               {{{0., 0., 0.}}, comm_rank * comm_size},
                           }),
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
  using ArborX::Point;
  using ExecutionSpace = typename DeviceType::execution_space;
  MPI_Comm comm = MPI_COMM_WORLD;
  Kokkos::View<Point *, DeviceType> points("Testing::points", 512);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, 512), KOKKOS_LAMBDA(int i) {
        points(i) = {{(float)i, (float)i, (float)i}};
      });

  ArborX::DistributedTree<typename DeviceType::memory_space> tree{
      comm, ExecutionSpace{}, points};
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
  auto const tree = makeDistributedTree<DeviceType>(
      comm, {
                {{{(double)comm_rank, 0., 0.}}, {{(double)comm_rank, 0., 0.}}},
                {{{(double)comm_rank + 1., 1., 1.}},
                 {{(double)comm_rank + 1., 1., 1.}}},
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
  if (comm_rank > 0)
  {
    ARBORX_TEST_QUERY_TREE(
        ExecutionSpace{}, tree,
        makeNearestQueries<DeviceType>({
            {{{(double)(comm_size - 1 - comm_rank) + .75, 0., 0.}}, 1},
        }),
        make_reference_solution<PairIndexRank>({{0, comm_size - comm_rank}},
                                               {0, 1}));
  }
  else
  {
    ARBORX_TEST_QUERY_TREE(
        ExecutionSpace{}, tree,
        makeNearestQueries<DeviceType>({
            {{{(double)(comm_size - 1 - comm_rank) + .75, 0., 0.}}, 1},
        }),
        make_reference_solution<PairIndexRank>({{0, comm_size - 1}}, {0, 1}));
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(non_approximate_box_nearest_neighbors, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

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
  auto const tree = makeDistributedTree<DeviceType>(
      comm, {
                {{{(double)comm_rank, 0., 0.}}, {{(double)comm_rank, 0., 0.}}},
                {{{(double)comm_rank + 1., 1., 1.}},
                 {{(double)comm_rank + 1., 1., 1.}}},
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
  if (comm_rank > 0)
  {
    ARBORX_TEST_QUERY_TREE(
        ExecutionSpace{}, tree,
        makeBoxNearestQueries<DeviceType>({
            {{{(double)(comm_size - 1 - comm_rank) + .65, 0., 0.}},
             {{(double)(comm_size - 1 - comm_rank) + .85, 0., 0.}},
             1},
        }),
        make_reference_solution<PairIndexRank>({{0, comm_size - comm_rank}},
                                               {0, 1}));
  }
  else
  {
    ARBORX_TEST_QUERY_TREE(
        ExecutionSpace{}, tree,
        makeBoxNearestQueries<DeviceType>({
            {{{(double)(comm_size - 1 - comm_rank) + .65, 0., 0.}},
             {{(double)(comm_size - 1 - comm_rank) + .85, 0., 0.}},
             1},
        }),
        make_reference_solution<PairIndexRank>({{0, comm_size - 1}}, {0, 1}));
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(non_approximate_sphere_nearest_neighbors,
                              DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

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
  auto const tree = makeDistributedTree<DeviceType>(
      comm, {
                {{{(double)comm_rank, 0., 0.}}, {{(double)comm_rank, 0., 0.}}},
                {{{(double)comm_rank + 1., 1., 1.}},
                 {{(double)comm_rank + 1., 1., 1.}}},
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
  if (comm_rank > 0)
  {
    ARBORX_TEST_QUERY_TREE(
        ExecutionSpace{}, tree,
        makeSphereNearestQueries<DeviceType>({
            {{{(double)(comm_size - 1 - comm_rank) + .75, 0., 0.}}, 0.1, 1},
        }),
        make_reference_solution<PairIndexRank>({{0, comm_size - comm_rank}},
                                               {0, 1}));
  }
  else
  {
    ARBORX_TEST_QUERY_TREE(
        ExecutionSpace{}, tree,
        makeSphereNearestQueries<DeviceType>({
            {{{(double)(comm_size - 1 - comm_rank) + .75, 0., 0.}}, 0.1, 1},
        }),
        make_reference_solution<PairIndexRank>({{0, comm_size - 1}}, {0, 1}));
  }
}

template <typename DeviceType>
struct CustomInlineCallbackWithAttachment
{
  Kokkos::View<ArborX::Point *, DeviceType> points;
  ArborX::Point const origin = {{0., 0., 0.}};
  template <typename Query, typename Insert>
  KOKKOS_FUNCTION void operator()(Query const &query, int index,
                                  Insert const &insert) const
  {
    auto data = ArborX::getData(query);
    float const distance_to_origin =
        ArborX::Details::distance(points(index), origin);

    insert(distance_to_origin + data);
  }
};

template <typename DeviceType>
struct CustomPostCallbackWithAttachment
{
  using tag = ArborX::Details::PostCallbackTag;
  Kokkos::View<ArborX::Point *, DeviceType> points;
  ArborX::Point const origin = {{0., 0., 0.}};
  template <typename Predicates, typename InOutView, typename InView,
            typename OutView>
  void operator()(Predicates const &queries, InOutView &offset, InView in,
                  OutView &out) const
  {
    using ExecutionSpace = typename DeviceType::execution_space;
    using ArborX::Details::distance;
    auto const n = offset.extent(0) - 1;
    Kokkos::realloc(out, in.extent(0));
    // NOTE workaround to avoid implicit capture of *this
    auto const &points_ = points;
    auto const &origin_ = origin;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
          auto data = ArborX::getData(queries(i));
          for (int j = offset(i); j < offset(i + 1); ++j)
          {
            out(j) = (float)distance(points_(in(j)), origin_) + data;
          }
        });
  }
};
BOOST_AUTO_TEST_CASE_TEMPLATE(callback_with_attachment, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

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
  auto const tree = makeDistributedTree<DeviceType>(
      comm,
      {{{{(double)comm_rank, 0., 0.}}, {{(double)comm_rank + 1, 1., 1.}}}});

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
  Kokkos::View<ArborX::Point *, DeviceType> points("Testing::points",
                                                   n_queries);
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
  ArborX::Point const origin = {{0., 0., 0.}};
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
               makeIntersectsBoxWithAttachmentQueries<DeviceType, int>(
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
               makeIntersectsBoxWithAttachmentQueries<DeviceType, int>(
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

std::vector<std::array<double, 3>>
make_random_cloud(double const Lx, double const Ly, double const Lz,
                  int const n, double const seed)
{
  std::vector<std::array<double, 3>> cloud(n);
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<double> distribution_x(0.0, Lx);
  std::uniform_real_distribution<double> distribution_y(0.0, Ly);
  std::uniform_real_distribution<double> distribution_z(0.0, Lz);
  for (int i = 0; i < n; ++i)
  {
    double x = distribution_x(generator);
    double y = distribution_y(generator);
    double z = distribution_z(generator);
    cloud[i] = {{x, y, z}};
  }
  return cloud;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(boost_comparison, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

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
  auto cloud = make_random_cloud(Lx, Ly, Lz, n, 0);
  auto queries = make_random_cloud(Lx, Ly, Lz, n, 1234);

  // The formula is a bit complicated but it does not require n be divisible
  // by comm_size
  int const local_n = (n + comm_size - 1 - comm_rank) / comm_size;
  Kokkos::View<ArborX::Box *, DeviceType> bounding_boxes(
      "Testing::bounding_boxes", local_n);
  auto bounding_boxes_host = Kokkos::create_mirror_view(bounding_boxes);
  for (int i = 0; i < n; ++i)
  {
    if (i % comm_size == comm_rank)
    {
      auto const &point = cloud[i];
      double const x = std::get<0>(point);
      double const y = std::get<1>(point);
      double const z = std::get<2>(point);
      bounding_boxes_host[i / comm_size] = {{{x, y, z}}, {{x, y, z}}};
    }
  }
  Kokkos::deep_copy(bounding_boxes, bounding_boxes_host);

  // Initialize the distributed search tree
  ArborX::DistributedTree<typename DeviceType::memory_space> distributed_tree(
      comm, ExecutionSpace{}, bounding_boxes);

  // make queries
  Kokkos::View<double *[3], ExecutionSpace> point_coords(
      "Testing::point_coords", local_n);
  auto point_coords_host = Kokkos::create_mirror_view(point_coords);
  Kokkos::View<double *, ExecutionSpace> radii("Testing::radii", local_n);
  auto radii_host = Kokkos::create_mirror_view(radii);
  Kokkos::View<int *[2], ExecutionSpace> within_n_pts("Testing::within_n_pts",
                                                      local_n);
  std::default_random_engine generator(0);
  std::uniform_real_distribution<double> distribution_radius(
      0.0, std::sqrt(Lx * Lx + Ly * Ly + Lz * Lz));
  std::uniform_int_distribution<int> distribution_k(1, std::floor(sqrt(n * n)));
  for (int i = 0; i < n; ++i)
  {
    if (i % comm_size == comm_rank)
    {
      auto const &point = queries[i];
      int const j = i / comm_size;
      double const x = std::get<0>(point);
      double const y = std::get<1>(point);
      double const z = std::get<2>(point);
      radii_host(j) = distribution_radius(generator);

      point_coords_host(j, 0) = x;
      point_coords_host(j, 1) = y;
      point_coords_host(j, 2) = z;
    }
  }
  Kokkos::deep_copy(point_coords, point_coords_host);
  Kokkos::deep_copy(radii, radii_host);

  Kokkos::View<decltype(ArborX::intersects(ArborX::Sphere{})) *, DeviceType>
      within_queries("Testing::within_queries", local_n);
  Kokkos::parallel_for(
      "register_within_queries",
      Kokkos::RangePolicy<ExecutionSpace>(0, local_n), KOKKOS_LAMBDA(int i) {
        within_queries(i) = ArborX::intersects(ArborX::Sphere{
            {{point_coords(i, 0), point_coords(i, 1), point_coords(i, 2)}},
            radii(i)});
      });

  auto within_queries_host = Kokkos::create_mirror_view(within_queries);
  Kokkos::deep_copy(within_queries_host, within_queries);

  BoostExt::ParallelRTree<ArborX::Box> rtree(comm, ExecutionSpace{},
                                             bounding_boxes_host);

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, distributed_tree, within_queries,
                         query(ExecutionSpace{}, rtree, within_queries_host));
}

template <typename MemorySpace>
class RayNearestPredicate
{
public:
  RayNearestPredicate(
      Kokkos::View<ArborX::Experimental::Ray *, MemorySpace> const &rays)
      : _rays(rays)
  {}

  KOKKOS_FUNCTION std::size_t size() const { return _rays.extent(0); }

  KOKKOS_FUNCTION ArborX::Experimental::Ray const &get(unsigned int i) const
  {
    return _rays(i);
  }

private:
  Kokkos::View<ArborX::Experimental::Ray *, MemorySpace> _rays;
};

template <typename MemorySpace>
struct ArborX::AccessTraits<RayNearestPredicate<MemorySpace>,
                            ArborX::PredicatesTag>
{
  using memory_space = MemorySpace;

  static KOKKOS_FUNCTION std::size_t
  size(RayNearestPredicate<MemorySpace> const &ray_nearest)
  {
    return ray_nearest.size();
  }

  static KOKKOS_FUNCTION auto
  get(RayNearestPredicate<MemorySpace> const &ray_nearest, std::size_t i)
  {
    return nearest(ray_nearest.get(i), 1);
  }
};

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
  auto const tree = makeDistributedTree<DeviceType>(
      comm,
      {
          {{{(double)comm_rank, 0., 0.}}, {{(double)comm_rank + 1., 1., 1.}}},
      });

  std::vector<ArborX::Experimental::Ray> rays = {
      {{comm_rank + 0.5f, -0.5f, 0.5f}, {0.f, 1.f, 0.f}},
      {{-0.5f, 0.5f, 0.5f}, {1.f, 0.f, 0.f}},
  };

  ARBORX_TEST_QUERY_TREE(
      ExecutionSpace{}, tree,
      RayNearestPredicate(ArborXTest::toView<MemorySpace>(rays)),
      make_reference_solution<PairIndexRank>({{0, comm_rank}, {0, 0}},
                                             {0, 1, 2}));
}
