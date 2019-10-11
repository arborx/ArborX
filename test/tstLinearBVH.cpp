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
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_LinearBVH.hpp>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <iostream>
#include <random>
#include <tuple>

#include "Search_UnitTestHelpers.hpp"

#define BOOST_TEST_MODULE LinearBVH

namespace tt = boost::test_tools;

template <typename T>
struct TreeTypeTraits;

template <typename... DeviceTypes>
struct TreeTypeTraits<std::tuple<DeviceTypes...>>
{
  using type = std::tuple<ArborX::BVH<DeviceTypes>...>;
};

using TreeTypes = typename TreeTypeTraits<ARBORX_DEVICE_TYPES>::type;

BOOST_AUTO_TEST_CASE_TEMPLATE(empty_tree, Tree, TreeTypes)
{
  using device_type = typename Tree::device_type;
  // tree is empty, it has no leaves.
  for (auto const &empty_tree : {
           Tree{},         // default constructed
           make<Tree>({}), // constructed with empty view of boxes
       })
  {
    BOOST_TEST(empty_tree.empty());
    BOOST_TEST(empty_tree.size() == 0);
    // Tree::bounds() returns an invalid box when the tree is empty.
    BOOST_TEST(ArborX::Details::equals(empty_tree.bounds(), {}));

    // Passing a view with no query does seem a bit silly but we still need
    // to support it. And since the tag dispatching yields different tree
    // traversals for nearest and spatial predicates, we do have to check
    // the results for various type of queries.
    checkResults(empty_tree, makeIntersectsBoxQueries<device_type>({}), {},
                 {0});

    // NOTE: Admittedly testing for both intersection with a box and with a
    // sphere queries might be a bit overkill but I'd rather test for all the
    // queries we plan on using.
    checkResults(empty_tree, makeIntersectsSphereQueries<device_type>({}), {},
                 {0});

    checkResults(empty_tree, makeNearestQueries<device_type>({}), {}, {0});

    // Passing an empty distance vector.
    checkResults(empty_tree, makeNearestQueries<device_type>({}), {}, {0});

    // Now passing a couple queries of various type and checking the
    // results.
    checkResults(
        empty_tree,
        makeIntersectsBoxQueries<device_type>({
            {}, // Did not bother giving a valid box here but that's fine.
            {},
        }),
        {}, {0, 0, 0});

    checkResults(empty_tree,
                 makeIntersectsSphereQueries<device_type>({
                     {{{0., 0., 0.}}, 1.},
                     {{{1., 1., 1.}}, 2.},
                 }),
                 {}, {0, 0, 0});

    checkResults(empty_tree,
                 makeNearestQueries<device_type>({
                     {{{0., 0., 0.}}, 1},
                     {{{1., 1., 1.}}, 2},
                 }),
                 {}, {0, 0, 0});

    checkResults(empty_tree,
                 makeNearestQueries<device_type>({
                     {{{0., 0., 0.}}, 1},
                     {{{1., 1., 1.}}, 2},
                 }),
                 {}, {0, 0, 0}, {});
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(single_leaf_tree, Tree, TreeTypes)
{
  using device_type = typename Tree::device_type;
  // tree has a single leaf (unit box)
  auto const single_leaf_tree = make<Tree>({
      {{{0., 0., 0.}}, {{1., 1., 1.}}},
  });

  BOOST_TEST(!single_leaf_tree.empty());
  BOOST_TEST(single_leaf_tree.size() == 1);
  BOOST_TEST(ArborX::Details::equals(single_leaf_tree.bounds(),
                                     {{{0., 0., 0.}}, {{1., 1., 1.}}}));

  checkResults(single_leaf_tree, makeIntersectsBoxQueries<device_type>({}), {},
               {0});

  checkResults(single_leaf_tree, makeIntersectsSphereQueries<device_type>({}),
               {}, {0});

  checkResults(single_leaf_tree, makeNearestQueries<device_type>({}), {}, {0});

  checkResults(single_leaf_tree, makeNearestQueries<device_type>({}), {}, {0},
               {});

  checkResults(
      single_leaf_tree,
      makeNearestQueries<device_type>({{{0., 0., 0.}, 3}, {{4., 5., 1.}, 1}}),
      {0, 0}, {0, 1, 2}, {0., 5.});

  checkResults(single_leaf_tree,
               makeIntersectsBoxQueries<device_type>({
                   {{{5., 5., 5.}}, {{5., 5., 5.}}},
                   {{{.5, .5, .5}}, {{.5, .5, .5}}},
               }),
               {0}, {0, 0, 1});

  checkResults(single_leaf_tree,
               makeIntersectsSphereQueries<device_type>({
                   {{{0., 0., 0.}}, 1.},
                   {{{1., 1., 1.}}, 3.},
                   {{{5., 5., 5.}}, 2.},
               }),
               {0, 0}, {0, 1, 2, 2});

  checkResults(single_leaf_tree,
               makeNearestQueries<device_type>({
                   {{{0., 0., 0.}}, 1},
                   {{{1., 1., 1.}}, 2},
                   {{{2., 2., 2.}}, 3},
               }),
               {0, 0, 0}, {0, 1, 2, 3});

  checkResults(single_leaf_tree,
               makeNearestQueries<device_type>({
                   {{{1., 0., 0.}}, 1},
                   {{{0., 2., 0.}}, 2},
                   {{{0., 0., 3.}}, 3},
               }),
               {0, 0, 0}, {0, 1, 2, 3}, {0., 1., 2.});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(couple_leaves_tree, Tree, TreeTypes)
{
  using device_type = typename Tree::device_type;

  auto const couple_leaves_tree = make<Tree>({
      {{{0., 0., 0.}}, {{0., 0., 0.}}},
      {{{1., 1., 1.}}, {{1., 1., 1.}}},
  });

  BOOST_TEST(!couple_leaves_tree.empty());
  BOOST_TEST(couple_leaves_tree.size() == 2);
  BOOST_TEST(ArborX::Details::equals(couple_leaves_tree.bounds(),
                                     {{{0., 0., 0.}}, {{1., 1., 1.}}}));

  // single query intersects with nothing
  checkResults(couple_leaves_tree,
               makeIntersectsBoxQueries<device_type>({
                   {},
               }),
               {}, {0, 0});

  // single query intersects with both
  checkResults(couple_leaves_tree,
               makeIntersectsBoxQueries<device_type>({
                   {{{0., 0., 0.}}, {{1., 1., 1.}}},
               }),
               {1, 0}, {0, 2});

  // single query intersects with only one
  checkResults(couple_leaves_tree,
               makeIntersectsBoxQueries<device_type>({
                   {{{0.5, 0.5, 0.5}}, {{1.5, 1.5, 1.5}}},
               }),
               {1}, {0, 1});

  // a couple queries both intersect with nothing
  checkResults(couple_leaves_tree,
               makeIntersectsBoxQueries<device_type>({
                   {},
                   {},
               }),
               {}, {0, 0, 0});

  // a couple queries first intersects with nothing second with only one
  checkResults(couple_leaves_tree,
               makeIntersectsBoxQueries<device_type>({
                   {},
                   {{{0., 0., 0.}}, {{0., 0., 0.}}},
               }),
               {0}, {0, 0, 1});

  // no query
  checkResults(couple_leaves_tree, makeIntersectsBoxQueries<device_type>({}),
               {}, {0});

  checkResults(couple_leaves_tree,
               makeNearestQueries<device_type>({
                   {{{0., 0., 0.}}, 2},
                   {{{1., 0., 0.}}, 4},
               }),
               {0, 1, 0, 1}, {0, 2, 4}, {0., sqrt(3.), 1., sqrt(2.)});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(duplicated_leaves, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  // The tree contains multiple (more than two) leaves that will be assigned
  // the same Morton code.  This was able to trigger a bug that we discovered
  // when building trees over ~10M indexable values.  The hierarchy generated
  // at construction had leaves with no parent which yielded a segfault later
  // when computing bounding boxes and walking the hierarchy toward the root.
  auto const bvh = make<ArborX::BVH<DeviceType>>({
      {{{0., 0., 0.}}, {{0., 0., 0.}}},
      {{{1., 1., 1.}}, {{1., 1., 1.}}},
      {{{1., 1., 1.}}, {{1., 1., 1.}}},
      {{{1., 1., 1.}}, {{1., 1., 1.}}},
  });

  checkResults(bvh,
               makeIntersectsSphereQueries<DeviceType>({
                   {{{0., 0., 0.}}, 1.},
                   {{{1., 1., 1.}}, 1.},
                   {{{.5, .5, .5}}, 1.},
               }),
               {0, 1, 2, 3, 0, 1, 2, 3}, {0, 1, 4, 8});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(buffer_optimization, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  auto const bvh = make<ArborX::BVH<DeviceType>>({
      {{{0., 0., 0.}}, {{0., 0., 0.}}},
      {{{1., 0., 0.}}, {{1., 0., 0.}}},
      {{{2., 0., 0.}}, {{2., 0., 0.}}},
      {{{3., 0., 0.}}, {{3., 0., 0.}}},
  });

  auto const queries = makeIntersectsBoxQueries<DeviceType>({
      {},
      {{{0., 0., 0.}}, {{3., 3., 3.}}},
      {},
  });

  using ViewType = Kokkos::View<int *, DeviceType>;
  ViewType indices("indices", 0);
  ViewType offset("offset", 0);

  std::vector<int> indices_ref = {3, 2, 1, 0};
  std::vector<int> offset_ref = {0, 0, 4, 4};
  auto checkResultsAreFine = [&indices, &offset, &indices_ref,
                              &offset_ref]() -> void {
    auto indices_host = Kokkos::create_mirror_view(indices);
    Kokkos::deep_copy(indices_host, indices);
    auto offset_host = Kokkos::create_mirror_view(offset);
    Kokkos::deep_copy(offset_host, offset);
    BOOST_TEST(indices_host == indices_ref, tt::per_element());
    BOOST_TEST(offset_host == offset_ref, tt::per_element());
  };

  BOOST_CHECK_NO_THROW(bvh.query(queries, indices, offset));
  checkResultsAreFine();

  // compute number of results per query
  auto counts = ArborX::cloneWithoutInitializingNorCopying(offset);
  ArborX::adjacentDifference(offset, counts);
  // extract optimal buffer size
  auto const max_results_per_query = ArborX::max(counts);
  BOOST_TEST(max_results_per_query == 4);

  // optimal size
  BOOST_CHECK_NO_THROW(
      bvh.query(queries, indices, offset, -max_results_per_query));
  checkResultsAreFine();

  // buffer size insufficient
  BOOST_TEST(max_results_per_query > 1);
  BOOST_CHECK_NO_THROW(bvh.query(queries, indices, offset, +1));
  checkResultsAreFine();
  BOOST_CHECK_THROW(bvh.query(queries, indices, offset, -1),
                    ArborX::SearchException);

  // adequate buffer size
  BOOST_TEST(max_results_per_query < 5);
  BOOST_CHECK_NO_THROW(bvh.query(queries, indices, offset, +5));
  checkResultsAreFine();
  BOOST_CHECK_NO_THROW(bvh.query(queries, indices, offset, -5));
  checkResultsAreFine();

  // passing null size skips the buffer optimization and never throws
  BOOST_CHECK_NO_THROW(bvh.query(queries, indices, offset, 0));
  checkResultsAreFine();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(not_exceeding_stack_capacity, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  std::vector<ArborX::Box> boxes;
  int const n = 4096; // exceed stack capacity which is 64
  boxes.reserve(n);
  for (int i = 0; i < n; ++i)
  {
    double const a = i;
    double const b = i + 1;
    boxes.push_back({{{a, a, a}}, {{b, b, b}}});
  }
  auto const bvh = make<ArborX::BVH<DeviceType>>(boxes);

  Kokkos::View<int *, DeviceType> indices("indices", 0);
  Kokkos::View<int *, DeviceType> offset("offset", 0);
  // query number of nearest neighbors that exceed capacity of the stack is
  // not a problem
  BOOST_CHECK_NO_THROW(bvh.query(makeNearestQueries<DeviceType>({
                                     {{{0., 0., 0.}}, n},
                                 }),
                                 indices, offset));
  BOOST_TEST(ArborX::lastElement(offset) == n);

  // spatial query that find all indexable in the tree is also fine
  BOOST_CHECK_NO_THROW(bvh.query(makeIntersectsBoxQueries<DeviceType>({
                                     {},
                                     {{{0., 0., 0.}}, {{n, n, n}}},
                                 }),
                                 indices, offset));
  BOOST_TEST(ArborX::lastElement(offset) == n);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(miscellaneous, DeviceType, ARBORX_DEVICE_TYPES)
{
  auto const bvh = make<ArborX::BVH<DeviceType>>({
      {{{1., 3., 5.}}, {{2., 4., 6.}}},
  });
  auto const empty_bvh = make<ArborX::BVH<DeviceType>>({});

  // Batched queries BVH::query( Kokkos::View<Query *, ...>, ... ) returns
  // early if the tree is empty.  Below we ensure that a direct call to the
  // single query TreeTraversal::query() actually handles empty trees
  // properly.
  using ExecutionSpace = typename DeviceType::execution_space;
  Kokkos::View<int *, DeviceType> zeros("zeros", 3);
  Kokkos::deep_copy(zeros, 255);
  Kokkos::View<Kokkos::pair<int, double> *, DeviceType> empty_buffer(
      "empty_buffer", 0);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, 1), KOKKOS_LAMBDA(int) {
        ArborX::Point p = {{0., 0., 0.}};
        double r = 1.0;
        // spatial query on empty tree
        zeros(0) = ArborX::Details::TreeTraversal<DeviceType>::query(
            empty_bvh, ArborX::intersects(ArborX::Sphere{p, r}), [](int) {});
        // nearest query on empty tree
        zeros(1) = ArborX::Details::TreeTraversal<DeviceType>::query(
            empty_bvh, ArborX::nearest(p), [](int, double) {}, empty_buffer);
        // nearest query for k < 1
        zeros(2) = ArborX::Details::TreeTraversal<DeviceType>::query(
            bvh, ArborX::nearest(p, 0), [](int, double) {}, empty_buffer);
      });
  ExecutionSpace().fence();
  auto zeros_host = Kokkos::create_mirror_view(zeros);
  Kokkos::deep_copy(zeros_host, zeros);
  std::vector<int> zeros_ref = {0, 0, 0};
  BOOST_TEST(zeros_host == zeros_ref, tt::per_element());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(structured_grid, DeviceType, ARBORX_DEVICE_TYPES)
{
  double Lx = 100.0;
  double Ly = 100.0;
  double Lz = 100.0;
  int nx = 11;
  int ny = 11;
  int nz = 11;
  int n = nx * ny * nz;

  using ExecutionSpace = typename DeviceType::execution_space;

  Kokkos::View<ArborX::Box *, DeviceType> bounding_boxes("bounding_boxes", n);
  Kokkos::parallel_for("fill_bounding_boxes",
                       Kokkos::RangePolicy<ExecutionSpace>(0, nx),
                       KOKKOS_LAMBDA(int i) {
                         [[gnu::unused]] double x, y, z;
                         for (int j = 0; j < ny; ++j)
                           for (int k = 0; k < nz; ++k)
                           {
                             x = i * Lx / (nx - 1);
                             y = j * Ly / (ny - 1);
                             z = k * Lz / (nz - 1);
                             bounding_boxes[i + j * nx + k * (nx * ny)] = {
                                 {{x, y, z}}, {{x, y, z}}};
                           }
                       });
  ExecutionSpace().fence();

  ArborX::BVH<DeviceType> bvh(bounding_boxes);

  // (i) use same objects for the queries than the objects we constructed the
  // BVH
  // i-2  i-1  i  i+1
  //
  //  o    o   o   o   j+1
  //          ---
  //  o    o | x | o   j
  //          ---
  //  o    o   o   o   j-1
  //
  //  o    o   o   o   j-2
  //
  Kokkos::View<decltype(ArborX::intersects(ArborX::Box{})) *, DeviceType>
      queries("queries", n);
  Kokkos::parallel_for("fill_queries",
                       Kokkos::RangePolicy<ExecutionSpace>(0, n),
                       KOKKOS_LAMBDA(int i) {
                         queries(i) = ArborX::intersects(bounding_boxes(i));
                       });
  ExecutionSpace().fence();

  Kokkos::View<int *, DeviceType> indices("indices", n);
  Kokkos::View<int *, DeviceType> offset("offset", n);

  bvh.query(queries, indices, offset);

  auto indices_host = Kokkos::create_mirror_view(indices);
  Kokkos::deep_copy(indices_host, indices);
  auto offset_host = Kokkos::create_mirror_view(offset);
  Kokkos::deep_copy(offset_host, offset);

  // we expect the collision list to be diag(0, 1, ..., nx*ny*nz-1)
  for (int i = 0; i < n; ++i)
  {
    BOOST_TEST(indices_host(i) == i);
    BOOST_TEST(offset_host(i) == i);
  }

  // (ii) use bounding boxes that intersects with first neighbors
  //
  // i-2  i-1  i  i+1
  //
  //  o    x---x---x   j+1
  //       |       |
  //  o    x   x   x   j
  //       |       |
  //  o    x---x---x   j-1
  //
  //  o    o   o   o   j-2
  //

  auto bounding_boxes_host = Kokkos::create_mirror_view(bounding_boxes);
  std::function<int(int, int, int)> ind = [nx, ny](int i, int j, int k) {
    return i + j * nx + k * (nx * ny);
  };
  std::vector<std::set<int>> ref(n);
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k)
      {
        int const index = ind(i, j, k);
        // bounding box around nodes of the structured grid will
        // intersect with neighboring nodes
        bounding_boxes_host[index] = {
            {{(i - 1) * Lx / (nx - 1), (j - 1) * Ly / (ny - 1),
              (k - 1) * Lz / (nz - 1)}},
            {{(i + 1) * Lx / (nx - 1), (j + 1) * Ly / (ny - 1),
              (k + 1) * Lz / (nz - 1)}}};
        // fill in reference solution to check againt the collision
        // list computed during the tree traversal
        if ((i > 0) && (j > 0) && (k > 0))
          ref[index].emplace(ind(i - 1, j - 1, k - 1));
        if ((i > 0) && (k > 0))
          ref[index].emplace(ind(i - 1, j, k - 1));
        if ((i > 0) && (j < ny - 1) && (k > 0))
          ref[index].emplace(ind(i - 1, j + 1, k - 1));
        if ((i > 0) && (j > 0))
          ref[index].emplace(ind(i - 1, j - 1, k));
        if (i > 0)
          ref[index].emplace(ind(i - 1, j, k));
        if ((i > 0) && (j < ny - 1))
          ref[index].emplace(ind(i - 1, j + 1, k));
        if ((i > 0) && (j > 0) && (k < nz - 1))
          ref[index].emplace(ind(i - 1, j - 1, k + 1));
        if ((i > 0) && (k < nz - 1))
          ref[index].emplace(ind(i - 1, j, k + 1));
        if ((i > 0) && (j < ny - 1) && (k < nz - 1))
          ref[index].emplace(ind(i - 1, j + 1, k + 1));

        if ((j > 0) && (k > 0))
          ref[index].emplace(ind(i, j - 1, k - 1));
        if (k > 0)
          ref[index].emplace(ind(i, j, k - 1));
        if ((j < ny - 1) && (k > 0))
          ref[index].emplace(ind(i, j + 1, k - 1));
        if (j > 0)
          ref[index].emplace(ind(i, j - 1, k));
        if (true) // NOLINT
          ref[index].emplace(ind(i, j, k));
        if (j < ny - 1)
          ref[index].emplace(ind(i, j + 1, k));
        if ((j > 0) && (k < nz - 1))
          ref[index].emplace(ind(i, j - 1, k + 1));
        if (k < nz - 1)
          ref[index].emplace(ind(i, j, k + 1));
        if ((j < ny - 1) && (k < nz - 1))
          ref[index].emplace(ind(i, j + 1, k + 1));

        if ((i < nx - 1) && (j > 0) && (k > 0))
          ref[index].emplace(ind(i + 1, j - 1, k - 1));
        if ((i < nx - 1) && (k > 0))
          ref[index].emplace(ind(i + 1, j, k - 1));
        if ((i < nx - 1) && (j < ny - 1) && (k > 0))
          ref[index].emplace(ind(i + 1, j + 1, k - 1));
        if ((i < nx - 1) && (j > 0))
          ref[index].emplace(ind(i + 1, j - 1, k));
        if (i < nx - 1)
          ref[index].emplace(ind(i + 1, j, k));
        if ((i < nx - 1) && (j < ny - 1))
          ref[index].emplace(ind(i + 1, j + 1, k));
        if ((i < nx - 1) && (j > 0) && (k < nz - 1))
          ref[index].emplace(ind(i + 1, j - 1, k + 1));
        if ((i < nx - 1) && (k < nz - 1))
          ref[index].emplace(ind(i + 1, j, k + 1));
        if ((i < nx - 1) && (j < ny - 1) && (k < nz - 1))
          ref[index].emplace(ind(i + 1, j + 1, k + 1));
      }

  Kokkos::deep_copy(bounding_boxes, bounding_boxes_host);
  Kokkos::parallel_for("fill_first_neighbors_queries",
                       Kokkos::RangePolicy<ExecutionSpace>(0, n),
                       KOKKOS_LAMBDA(int i) {
                         queries[i] = ArborX::intersects(bounding_boxes[i]);
                       });
  ExecutionSpace().fence();
  bvh.query(queries, indices, offset);
  indices_host = Kokkos::create_mirror_view(indices);
  Kokkos::deep_copy(indices_host, indices);
  offset_host = Kokkos::create_mirror_view(offset);
  Kokkos::deep_copy(offset_host, offset);

  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k)
      {
        int index = ind(i, j, k);
        for (int l = offset_host(index); l < offset_host(index + 1); ++l)
        {
          BOOST_TEST(ref[index].count(indices_host(l)) != 0);
        }
      }

  // (iii) use random points
  //
  // i-1      i      i+1
  //
  //  o       o       o   j+1
  //         -------
  //        |       |
  //        |   +   |
  //  o     | x     | o   j
  //         -------
  //
  //  o       o       o   j-1
  //
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution_x(0.0, Lz);
  std::uniform_real_distribution<double> distribution_y(0.0, Ly);
  std::uniform_real_distribution<double> distribution_z(0.0, Lz);

  for (int l = 0; l < n; ++l)
  {
    double x = distribution_x(generator);
    double y = distribution_y(generator);
    double z = distribution_z(generator);
    bounding_boxes_host(l) = {
        {{x - 0.5 * Lx / (nx - 1), y - 0.5 * Ly / (ny - 1),
          z - 0.5 * Lz / (nz - 1)}},
        {{x + 0.5 * Lx / (nx - 1), y + 0.5 * Ly / (ny - 1),
          z + 0.5 * Lz / (nz - 1)}}};

    int i = std::round(x / Lx * (nx - 1));
    int j = std::round(y / Ly * (ny - 1));
    int k = std::round(z / Lz * (nz - 1));
    // Save the indices for the check
    ref[l] = {ind(i, j, k)};
  }

  Kokkos::deep_copy(bounding_boxes, bounding_boxes_host);
  Kokkos::parallel_for("fill_first_neighbors_queries",
                       Kokkos::RangePolicy<ExecutionSpace>(0, n),
                       KOKKOS_LAMBDA(int i) {
                         queries[i] = ArborX::intersects(bounding_boxes[i]);
                       });
  ExecutionSpace().fence();
  bvh.query(queries, indices, offset);
  indices_host = Kokkos::create_mirror_view(indices);
  Kokkos::deep_copy(indices_host, indices);
  offset_host = Kokkos::create_mirror_view(offset);
  Kokkos::deep_copy(offset_host, offset);

  for (int i = 0; i < n; ++i)
  {
    BOOST_TEST(offset_host(i) == i);
    BOOST_TEST(ref[i].count(indices_host(i)) != 0);
  }
}

std::vector<std::array<double, 3>>
make_stuctured_cloud(double Lx, double Ly, double Lz, int nx, int ny, int nz)
{
  std::vector<std::array<double, 3>> cloud(nx * ny * nz);
  std::function<int(int, int, int)> ind = [nx, ny](int i, int j, int k) {
    return i + j * nx + k * (nx * ny);
  };
  double x, y, z;
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k)
      {
        x = i * Lx / (nx - 1);
        y = j * Ly / (ny - 1);
        z = k * Lz / (nz - 1);
        cloud[ind(i, j, k)] = {{x, y, z}};
      }
  return cloud;
}

std::vector<std::array<double, 3>> make_random_cloud(double Lx, double Ly,
                                                     double Lz, int n)
{
  std::vector<std::array<double, 3>> cloud(n);
  std::default_random_engine generator;
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

BOOST_AUTO_TEST_CASE_TEMPLATE(boost_rtree, DeviceType, ARBORX_DEVICE_TYPES)
{
  // contruct a cloud of points (nodes of a structured grid)
  double Lx = 10.0;
  double Ly = 10.0;
  double Lz = 10.0;
  int nx = 11;
  int ny = 11;
  int nz = 11;
  auto cloud = make_stuctured_cloud(Lx, Ly, Lz, nx, ny, nz);
  int n = cloud.size();

  Kokkos::View<ArborX::Box *, DeviceType> bounding_boxes("bounding_boxes", n);
  auto bounding_boxes_host = Kokkos::create_mirror_view(bounding_boxes);
  // build bounding volume hierarchy
  for (int i = 0; i < n; ++i)
  {
    auto const &point = cloud[i];
    double x = std::get<0>(point);
    double y = std::get<1>(point);
    double z = std::get<2>(point);
    bounding_boxes_host[i] = {{{x, y, z}}, {{x, y, z}}};
  }

  Kokkos::deep_copy(bounding_boxes, bounding_boxes_host);

  ArborX::BVH<DeviceType> bvh(bounding_boxes);

  auto rtree = BoostRTreeHelpers::makeRTree(bounding_boxes_host);

  // random points for radius search and kNN queries
  // compare our solution against Boost R-tree
  int const n_points = 100;
  auto queries = make_random_cloud(Lx, Ly, Lz, n_points);
  using ExecutionSpace = typename DeviceType::execution_space;
  Kokkos::View<double * [3], ExecutionSpace> point_coords("point_coords",
                                                          n_points);
  auto point_coords_host = Kokkos::create_mirror_view(point_coords);
  Kokkos::View<double *, ExecutionSpace> radii("radii", n_points);
  auto radii_host = Kokkos::create_mirror_view(radii);
  Kokkos::View<int * [2], ExecutionSpace> within_n_pts("within_n_pts",
                                                       n_points);
  Kokkos::View<int * [2], ExecutionSpace> nearest_n_pts("nearest_n_pts",
                                                        n_points);
  Kokkos::View<int *, ExecutionSpace> k("distribution_k", n_points);
  auto k_host = Kokkos::create_mirror_view(k);
  // use random radius for the search and random number k of for the kNN
  // search
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution_radius(
      0.0, std::sqrt(Lx * Lx + Ly * Ly + Lz * Lz));
  std::uniform_int_distribution<int> distribution_k(
      1, std::floor(sqrt(nx * nx + ny * ny + nz * nz)));
  for (unsigned int i = 0; i < n_points; ++i)
  {
    auto const &point = queries[i];
    double x = std::get<0>(point);
    double y = std::get<1>(point);
    double z = std::get<2>(point);
    radii_host[i] = distribution_radius(generator);
    k_host[i] = distribution_k(generator);
    point_coords_host(i, 0) = x;
    point_coords_host(i, 1) = y;
    point_coords_host(i, 2) = z;
  }

  Kokkos::deep_copy(point_coords, point_coords_host);
  Kokkos::deep_copy(radii, radii_host);
  Kokkos::deep_copy(k, k_host);

  Kokkos::View<ArborX::Nearest<ArborX::Point> *, DeviceType> nearest_queries(
      "nearest_queries", n_points);
  Kokkos::parallel_for(
      "register_nearest_queries",
      Kokkos::RangePolicy<ExecutionSpace>(0, n_points), KOKKOS_LAMBDA(int i) {
        nearest_queries(i) = ArborX::nearest<ArborX::Point>(
            {{point_coords(i, 0), point_coords(i, 1), point_coords(i, 2)}},
            k(i));
      });
  ExecutionSpace().fence();
  auto nearest_queries_host = Kokkos::create_mirror_view(nearest_queries);
  Kokkos::deep_copy(nearest_queries_host, nearest_queries);

  Kokkos::View<decltype(ArborX::intersects(ArborX::Sphere{})) *, DeviceType>
      within_queries("within_queries", n_points);
  Kokkos::parallel_for(
      "register_within_queries",
      Kokkos::RangePolicy<ExecutionSpace>(0, n_points), KOKKOS_LAMBDA(int i) {
        within_queries(i) = ArborX::intersects(ArborX::Sphere{
            {{point_coords(i, 0), point_coords(i, 1), point_coords(i, 2)}},
            radii(i)});
      });
  ExecutionSpace().fence();
  auto within_queries_host = Kokkos::create_mirror_view(within_queries);
  Kokkos::deep_copy(within_queries_host, within_queries);

  auto rtree_results =
      BoostRTreeHelpers::performQueries(rtree, nearest_queries_host);

  Kokkos::View<int *, DeviceType> offset_nearest("offset_nearest", 0);
  Kokkos::View<int *, DeviceType> indices_nearest("indices_nearest", 0);
  bvh.query(nearest_queries, indices_nearest, offset_nearest);
  auto offset_nearest_host = Kokkos::create_mirror_view(offset_nearest);
  Kokkos::deep_copy(offset_nearest_host, offset_nearest);
  auto indices_nearest_host = Kokkos::create_mirror_view(indices_nearest);
  Kokkos::deep_copy(indices_nearest_host, indices_nearest);
  auto bvh_results = std::make_tuple(offset_nearest_host, indices_nearest_host);

  validateResults(rtree_results, bvh_results);

  auto const alternate_tree_traversal_algorithm =
      ArborX::Details::NearestQueryAlgorithm::PriorityQueueBased_Deprecated;
  bvh.query(nearest_queries, indices_nearest, offset_nearest,
            alternate_tree_traversal_algorithm);
  Kokkos::deep_copy(offset_nearest_host, offset_nearest);
  Kokkos::deep_copy(indices_nearest_host, indices_nearest);
  bvh_results = std::make_tuple(offset_nearest_host, indices_nearest_host);
  validateResults(rtree_results, bvh_results);

  Kokkos::View<int *, DeviceType> offset_within("offset_within", 0);
  Kokkos::View<int *, DeviceType> indices_within("indices_within", 0);
  bvh.query(within_queries, indices_within, offset_within);
  auto offset_within_host = Kokkos::create_mirror_view(offset_within);
  Kokkos::deep_copy(offset_within_host, offset_within);
  auto indices_within_host = Kokkos::create_mirror_view(indices_within);
  Kokkos::deep_copy(indices_within_host, indices_within);
  bvh_results = std::make_tuple(offset_within_host, indices_within_host);

  rtree_results = BoostRTreeHelpers::performQueries(rtree, within_queries_host);

  validateResults(rtree_results, bvh_results);
}
