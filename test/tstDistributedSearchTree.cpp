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
  Kokkos::View<ArborX::Within *, DeviceType> queries("queries", 1);
  auto queries_host = Kokkos::create_mirror_view(queries);
  queries_host(0) =
      ArborX::within({{0.5 + comm_size - 1 - comm_rank, 0., 0.}}, 0.5);
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

  auto const emptry_dist_tree = makeDistributedSearchTree<DeviceType>(comm, {});

  BOOST_TEST(emptry_dist_tree.empty());
  BOOST_TEST(emptry_dist_tree.size() == 0);

  BOOST_TEST(ArborX::Details::equals(emptry_dist_tree.bounds(), {}));

  checkResults(emptry_dist_tree, makeOverlapQueries<DeviceType>({}), {}, {0},
               {});

  checkResults(emptry_dist_tree, makeWithinQueries<DeviceType>({}), {}, {0},
               {});

  checkResults(emptry_dist_tree, makeNearestQueries<DeviceType>({}), {}, {0},
               {});

  checkResults(emptry_dist_tree, makeNearestQueries<DeviceType>({}), {}, {0},
               {}, {});

  // Only rank 0 has a couple spatial queries with a spatial predicate
  if (comm_rank == 0)
    checkResults(emptry_dist_tree,
                 makeOverlapQueries<DeviceType>({
                     {},
                     {},
                 }),
                 {}, {0, 0, 0}, {});
  else
    checkResults(emptry_dist_tree, makeOverlapQueries<DeviceType>({}), {}, {0},
                 {});

  // All ranks but rank 0 have a single query with a spatial predicate
  if (comm_rank == 0)
    checkResults(emptry_dist_tree, makeWithinQueries<DeviceType>({}), {}, {0},
                 {});
  else
    checkResults(emptry_dist_tree,
                 makeWithinQueries<DeviceType>({
                     {{{(double)comm_rank, 0., 0.}}, (double)comm_size},
                 }),
                 {}, {0, 0}, {});

  // All ranks but rank 0 have a single query with a nearest predicate
  if (comm_rank == 0)
    checkResults(emptry_dist_tree, makeNearestQueries<DeviceType>({}), {}, {0},
                 {});
  else
    checkResults(emptry_dist_tree,
                 makeNearestQueries<DeviceType>({
                     {{{0., 0., 0.}}, comm_rank},
                 }),
                 {}, {0, 0}, {});

  // All ranks have a single query with a nearest predicate (this version
  // returns distances as well)
  checkResults(emptry_dist_tree,
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

  checkResults(tree, makeOverlapQueries<DeviceType>({}), {}, {0}, {});

  checkResults(tree, makeWithinQueries<DeviceType>({}), {}, {0}, {});

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
               makeOverlapQueries<DeviceType>({
                   {{{(double)comm_size - (double)comm_rank - .5, .5, .5}},
                    {{(double)comm_size - (double)comm_rank - .5, .5, .5}}},
                   {{{(double)comm_rank + .5, .5, .5}},
                    {{(double)comm_rank + .5, .5, .5}}},
               }),
               {0, 0}, {0, 1, 2}, {comm_size - 1 - comm_rank, comm_rank});

  checkResults(tree, makeNearestQueries<DeviceType>({}), {}, {0}, {});
  checkResults(tree, makeOverlapQueries<DeviceType>({}), {}, {0}, {});

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

  Kokkos::View<ArborX::Within *, DeviceType> within_queries("within_queries",
                                                            local_n);
  Kokkos::parallel_for(
      "register_within_queries",
      Kokkos::RangePolicy<ExecutionSpace>(0, local_n), KOKKOS_LAMBDA(int i) {
        within_queries(i) = ArborX::within(
            {{point_coords(i, 0), point_coords(i, 1), point_coords(i, 2)}},
            radii(i));
      });
  Kokkos::fence();

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
