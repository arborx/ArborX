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

#include "ArborX_BoostRTreeHelpers.hpp"
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include <ArborX_DistributedSearchTree.hpp>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <iostream>
#include <random>
#include <tuple>

#include "Search_UnitTestHelpers.hpp"
#include <mpi.h>

#define BOOST_TEST_MODULE DistributedSearchTree

BOOST_AUTO_TEST_CASE_TEMPLATE(hello_world, DeviceType, ARBORX_DEVICE_TYPES)
{
  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  int const n = 4;
  Kokkos::View<ArborX::Point *, DeviceType> points("points", n);
  // [  rank 0       [  rank 1       [  rank 2       [  rank 3       [
  // x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---x---
  // ^   ^   ^   ^
  // 0   1   2   3   ^   ^   ^   ^
  //                 0   1   2   3   ^   ^   ^   ^
  //                                 0   1   2   3   ^   ^   ^   ^
  //                                                 0   1   2   3
  using ExecutionSpace = typename DeviceType::execution_space;
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0, n),
                       KOKKOS_LAMBDA(int i) {
                         points(i) = {{(double)i / n + comm_rank, 0., 0.}};
                       });

  ArborX::DistributedSearchTree<DeviceType> tree(comm, points);

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
      queries("queries", 1);
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
      "nearest_queries", 1);
  auto nearest_queries_host = Kokkos::create_mirror_view(nearest_queries);
  nearest_queries_host(0) = ArborX::nearest<ArborX::Point>(
      {{0.0 + comm_size - 1 - comm_rank, 0., 0.}},
      comm_rank < comm_size - 1 ? 3 : 2);
  deep_copy(nearest_queries, nearest_queries_host);

  std::vector<int> indices_ref;
  std::vector<int> ranks_ref;
  for (int i = 0; i < n; ++i)
  {
    indices_ref.push_back(n - 1 - i);
    ranks_ref.push_back(comm_size - 1 - comm_rank);
  }
  if (comm_rank > 0)
  {
    indices_ref.push_back(0);
    ranks_ref.push_back(comm_size - comm_rank);
    checkResults(tree, queries, indices_ref, {0, n + 1}, ranks_ref);
  }
  else
  {
    checkResults(tree, queries, indices_ref, {0, n}, ranks_ref);
  }

  BOOST_TEST(n > 2);
  if (comm_rank < comm_size - 1)
  {
    checkResults(tree, nearest_queries, {0, n - 1, 1}, {0, 3},
                 {comm_size - 1 - comm_rank, comm_size - 2 - comm_rank,
                  comm_size - 1 - comm_rank});
  }
  else
  {
    checkResults(tree, nearest_queries, {0, 1}, {0, 2},
                 {comm_size - 1 - comm_rank, comm_size - 1 - comm_rank});
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(empty_tree, DeviceType, ARBORX_DEVICE_TYPES)
{
  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  auto const empty_dist_tree = makeDistributedSearchTree<DeviceType>(comm, {});

  BOOST_TEST(empty_dist_tree.empty());
  BOOST_TEST(empty_dist_tree.size() == 0);

  BOOST_TEST(ArborX::Details::equals(empty_dist_tree.bounds(), {}));

  checkResults(empty_dist_tree, makeIntersectsBoxQueries<DeviceType>({}), {},
               {0}, {});

  checkResults(empty_dist_tree, makeIntersectsSphereQueries<DeviceType>({}), {},
               {0}, {});

  checkResults(empty_dist_tree, makeNearestQueries<DeviceType>({}), {}, {0},
               {});

  checkResults(empty_dist_tree, makeNearestQueries<DeviceType>({}), {}, {0}, {},
               {});

  // Only rank 0 has a couple spatial queries with a spatial predicate
  if (comm_rank == 0)
    checkResults(empty_dist_tree,
                 makeIntersectsBoxQueries<DeviceType>({
                     {},
                     {},
                 }),
                 {}, {0, 0, 0}, {});
  else
    checkResults(empty_dist_tree, makeIntersectsBoxQueries<DeviceType>({}), {},
                 {0}, {});

  // All ranks but rank 0 have a single query with a spatial predicate
  if (comm_rank == 0)
    checkResults(empty_dist_tree, makeIntersectsSphereQueries<DeviceType>({}),
                 {}, {0}, {});
  else
    checkResults(empty_dist_tree,
                 makeIntersectsSphereQueries<DeviceType>({
                     {{{(double)comm_rank, 0., 0.}}, (double)comm_size},
                 }),
                 {}, {0, 0}, {});

  // All ranks but rank 0 have a single query with a nearest predicate
  if (comm_rank == 0)
    checkResults(empty_dist_tree, makeNearestQueries<DeviceType>({}), {}, {0},
                 {});
  else
    checkResults(empty_dist_tree,
                 makeNearestQueries<DeviceType>({
                     {{{0., 0., 0.}}, comm_rank},
                 }),
                 {}, {0, 0}, {});

  // All ranks have a single query with a nearest predicate (this version
  // returns distances as well)
  checkResults(empty_dist_tree,
               makeNearestQueries<DeviceType>({
                   {{{0., 0., 0.}}, comm_size},
               }),
               {}, {0, 0}, {}, {});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(unique_leaf_on_rank_0, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  // tree has one unique leaf that lives on rank 0
  auto const tree =
      (comm_rank == 0 ? makeDistributedSearchTree<DeviceType>(
                            comm,
                            {
                                {{{0., 0., 0.}}, {{1., 1., 1.}}},
                            })
                      : makeDistributedSearchTree<DeviceType>(comm, {}));

  BOOST_TEST(!tree.empty());
  BOOST_TEST(tree.size() == 1);

  BOOST_TEST(
      ArborX::Details::equals(tree.bounds(), {{{0., 0., 0.}}, {{1., 1., 1.}}}));

  checkResults(tree, makeIntersectsBoxQueries<DeviceType>({}), {}, {0}, {});

  checkResults(tree, makeIntersectsSphereQueries<DeviceType>({}), {}, {0}, {});

  checkResults(tree, makeNearestQueries<DeviceType>({}), {}, {0}, {});

  checkResults(tree, makeNearestQueries<DeviceType>({}), {}, {0}, {}, {});

  // Querying for more neighbors than there are leaves in the tree
  checkResults(tree,
               makeNearestQueries<DeviceType>({
                   {{{(double)comm_rank, (double)comm_rank, (double)comm_rank}},
                    comm_size},
               }),
               {0}, {0, 1}, {0});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(one_leaf_per_rank, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  // tree has one leaf per rank
  auto const tree = makeDistributedSearchTree<DeviceType>(
      comm,
      {
          {{{(double)comm_rank, 0., 0.}}, {{(double)comm_rank + 1., 1., 1.}}},
      });

  BOOST_TEST(!tree.empty());
  BOOST_TEST((int)tree.size() == comm_size);

  BOOST_TEST(ArborX::Details::equals(
      tree.bounds(), {{{0., 0., 0.}}, {{(double)comm_size, 1., 1.}}}));

  checkResults(tree,
               makeIntersectsBoxQueries<DeviceType>({
                   {{{(double)comm_size - (double)comm_rank - .5, .5, .5}},
                    {{(double)comm_size - (double)comm_rank - .5, .5, .5}}},
                   {{{(double)comm_rank + .5, .5, .5}},
                    {{(double)comm_rank + .5, .5, .5}}},
               }),
               {0, 0}, {0, 1, 2}, {comm_size - 1 - comm_rank, comm_rank});

  checkResults(tree, makeNearestQueries<DeviceType>({}), {}, {0}, {});
  checkResults(tree, makeIntersectsBoxQueries<DeviceType>({}), {}, {0}, {});

  if (comm_rank > 0)
    checkResults(tree,
                 makeNearestQueries<DeviceType>({
                     {{{0., 0., 0.}}, comm_rank * comm_size},
                 }),
                 std::vector<int>(comm_size, 0), {0, comm_size}, [comm_size]() {
                   std::vector<int> r(comm_size);
                   std::iota(begin(r), end(r), 0);
                   return r;
                 }());
  else
    checkResults(tree,
                 makeNearestQueries<DeviceType>({
                     {{{0., 0., 0.}}, comm_rank * comm_size},
                 }),
                 {}, {0, 0}, {});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(do_not_exceed_capacity, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
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
  Kokkos::View<Point *, DeviceType> points("points", 512);
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0, 512),
                       KOKKOS_LAMBDA(int i) {
                         points(i) = {{(float)i, (float)i, (float)i}};
                       });
  ArborX::DistributedSearchTree<DeviceType> tree{comm, points};
  Kokkos::View<decltype(nearest(Point{})) *, DeviceType> queries("queries", 1);
  Kokkos::deep_copy(queries, nearest(Point{0, 0, 0}, 512));
  Kokkos::View<int *, DeviceType> indices("indices", 0);
  Kokkos::View<int *, DeviceType> offset("offset", 0);
  Kokkos::View<int *, DeviceType> ranks("ranks", 0);
  BOOST_CHECK_NO_THROW(tree.query(queries, indices, offset, ranks));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(non_approximate_nearest_neighbors, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
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
  auto const tree = makeDistributedSearchTree<DeviceType>(
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
    checkResults(tree,
                 makeNearestQueries<DeviceType>({
                     {{{(double)(comm_size - 1 - comm_rank) + .75, 0., 0.}}, 1},
                 }),
                 {0}, {0, 1}, {comm_size - comm_rank});
  else
    checkResults(tree,
                 makeNearestQueries<DeviceType>({
                     {{{(double)(comm_size - 1 - comm_rank) + .75, 0., 0.}}, 1},
                 }),
                 {0}, {0, 1}, {comm_size - 1});
}

template <typename DeviceType>
struct CustomInlineCallbackAttachmentSpatialPredicate
{
  using tag = ArborX::Details::InlineCallbackTag;
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
struct CustomPostCallbackAttachmentSpatialPredicate
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
    ArborX::reallocWithoutInitializing(out, in.extent(0));
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
  auto const tree = makeDistributedSearchTree<DeviceType>(
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
  Kokkos::View<ArborX::Point *, DeviceType> points("points", n_queries);
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0, n_queries),
                       KOKKOS_LAMBDA(int i) {
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
  Kokkos::View<float *, DeviceType> ref("ref", n_results);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n_results), KOKKOS_LAMBDA(int i) {
        ref(i) =
            float(ArborX::Details::distance(points(i), origin) + 1) + comm_rank;
      });

  {
    Kokkos::View<float *, DeviceType> custom("custom", 0);
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    tree.query(
        makeIntersectsBoxWithAttachmentQueries<DeviceType, int>(
            {{points_host(0), points_host(0)}}, {comm_rank}),
        CustomInlineCallbackAttachmentSpatialPredicate<DeviceType>{points},
        custom, offset);

    auto custom_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, custom);
    auto ref_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ref);
    auto offset_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offset);
    validateResults(std::make_tuple(offset_host, custom_host),
                    std::make_tuple(offset_host, ref_host));
  }
  {
    Kokkos::View<float *, DeviceType> custom("custom", 0);
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    tree.query(makeIntersectsBoxWithAttachmentQueries<DeviceType, int>(
                   {{points_host(0), points_host(0)}}, {comm_rank}),
               CustomPostCallbackAttachmentSpatialPredicate<DeviceType>{points},
               custom, offset);

    auto custom_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, custom);
    auto ref_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ref);
    auto offset_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offset);
    validateResults(std::make_tuple(offset_host, custom_host),
                    std::make_tuple(offset_host, ref_host));
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
  Kokkos::View<ArborX::Box *, DeviceType> bounding_boxes("bounding_boxes",
                                                         local_n);
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
  ArborX::DistributedSearchTree<DeviceType> distributed_tree(comm,
                                                             bounding_boxes);

  auto rtree = BoostRTreeHelpers::makeRTree(comm, bounding_boxes_host);

  // make queries
  using ExecutionSpace = typename DeviceType::execution_space;
  Kokkos::View<double * [3], ExecutionSpace> point_coords("point_coords",
                                                          local_n);
  auto point_coords_host = Kokkos::create_mirror_view(point_coords);
  Kokkos::View<double *, ExecutionSpace> radii("radii", local_n);
  auto radii_host = Kokkos::create_mirror_view(radii);
  Kokkos::View<int * [2], ExecutionSpace> within_n_pts("within_n_pts", local_n);
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
      within_queries("within_queries", local_n);
  Kokkos::parallel_for(
      "register_within_queries",
      Kokkos::RangePolicy<ExecutionSpace>(0, local_n), KOKKOS_LAMBDA(int i) {
        within_queries(i) = ArborX::intersects(ArborX::Sphere{
            {{point_coords(i, 0), point_coords(i, 1), point_coords(i, 2)}},
            radii(i)});
      });

  // Perform the search
  Kokkos::View<int *, DeviceType> indices("indices", 0);
  Kokkos::View<int *, DeviceType> offset("offset", 0);
  Kokkos::View<int *, DeviceType> ranks("ranks", 0);
  distributed_tree.query(within_queries, indices, offset, ranks);
  auto indices_host = Kokkos::create_mirror_view(indices);
  Kokkos::deep_copy(indices_host, indices);
  auto offset_host = Kokkos::create_mirror_view(offset);
  Kokkos::deep_copy(offset_host, offset);
  auto ranks_host = Kokkos::create_mirror_view(ranks);
  Kokkos::deep_copy(ranks_host, ranks);
  auto bvh_results = std::make_tuple(offset_host, indices_host, ranks_host);

  auto within_queries_host = Kokkos::create_mirror_view(within_queries);
  Kokkos::deep_copy(within_queries_host, within_queries);
  auto rtree_results =
      BoostRTreeHelpers::performQueries(rtree, within_queries_host);

  validateResults(bvh_results, rtree_results);
}

template <typename DeviceType>
struct Helper
{
  template <typename View1, typename View2>
  static void checkSendAcrossNetwork(MPI_Comm comm, View1 const &ranks,
                                     View2 const &v_exp, View2 const &v_ref)
  {
    ArborX::Details::Distributor<DeviceType> distributor(comm);
    distributor.createFromSends(typename DeviceType::execution_space{}, ranks);

    // NOTE here we assume that the reference solution is sized properly
    auto v_imp = Kokkos::create_mirror(typename View2::memory_space(), v_ref);

    ArborX::Details::DistributedSearchTreeImpl<DeviceType>::sendAcrossNetwork(
        typename DeviceType::execution_space{}, distributor, v_exp, v_imp);

    // FIXME not sure why I need that guy but I do get a bus error when it
    // is not here...
    Kokkos::fence();

    auto v_imp_host = Kokkos::create_mirror_view(v_imp);
    Kokkos::deep_copy(v_imp_host, v_imp);
    auto v_ref_host = Kokkos::create_mirror_view(v_ref);
    Kokkos::deep_copy(v_ref_host, v_ref);

    BOOST_TEST(v_imp.extent(0) == v_ref.extent(0));
    BOOST_TEST(v_imp.extent(1) == v_ref.extent(1));
    for (unsigned int i = 0; i < v_imp.extent(0); ++i)
    {
      for (unsigned int j = 0; j < v_imp.extent(1); ++j)
      {
        BOOST_TEST(v_imp_host(i, j) == v_ref_host(i, j));
      }
    }
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(send_across_network, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  MPI_Comm comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);

  int const DIM = 3;

  // send 1 packet to rank k
  // receive comm_size packets
  Kokkos::View<int **, DeviceType> u_exp("u_exp", comm_size, DIM);
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0, comm_size),
                       KOKKOS_LAMBDA(int i) {
                         for (int j = 0; j < DIM; ++j)
                           u_exp(i, j) = i + j * comm_rank;
                       });
  Kokkos::fence();

  Kokkos::View<int *, DeviceType> ranks_u("", comm_size);
  ArborX::iota(ExecutionSpace{}, ranks_u, 0);

  Kokkos::View<int **, DeviceType> u_ref("u_ref", comm_size, DIM);
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0, comm_size),
                       KOKKOS_LAMBDA(int i) {
                         for (int j = 0; j < DIM; ++j)
                           u_ref(i, j) = comm_rank + i * j;
                       });
  Kokkos::fence();

  Helper<DeviceType>::checkSendAcrossNetwork(comm, ranks_u, u_exp, u_ref);
}
