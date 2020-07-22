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

BOOST_AUTO_TEST_CASE_TEMPLATE(test_empty_tree, Tree, TreeTypes)
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
    ARBORX_TEST_QUERY_TREE(empty_tree,
                           makeIntersectsBoxQueries<device_type>({}),
                           make_reference_solution<int>({}, {0}));

    // NOTE: Admittedly testing for both intersection with a box and with a
    // sphere queries might be a bit overkill but I'd rather test for all the
    // queries we plan on using.
    ARBORX_TEST_QUERY_TREE(empty_tree,
                           makeIntersectsSphereQueries<device_type>({}),
                           make_reference_solution<int>({}, {0}));

    ARBORX_TEST_QUERY_TREE(empty_tree, makeNearestQueries<device_type>({}),
                           make_reference_solution<int>({}, {0}));

    // Passing an empty distance vector.
    ARBORX_TEST_QUERY_TREE_WITH_DISTANCE(
        empty_tree, makeNearestQueries<device_type>({}),
        (make_reference_solution<Kokkos::pair<int, float>>)({}, {0}));

    // Now passing a couple queries of various type and checking the
    // results.
    ARBORX_TEST_QUERY_TREE(
        empty_tree,
        makeIntersectsBoxQueries<device_type>({
            {}, // Did not bother giving a valid box here but that's fine.
            {},
        }),
        make_reference_solution<int>({}, {0, 0, 0}));

    ARBORX_TEST_QUERY_TREE(empty_tree,
                           makeIntersectsSphereQueries<device_type>({
                               {{{0., 0., 0.}}, 1.},
                               {{{1., 1., 1.}}, 2.},
                           }),
                           make_reference_solution<int>({}, {0, 0, 0}));

    ARBORX_TEST_QUERY_TREE(empty_tree,
                           makeNearestQueries<device_type>({
                               {{{0., 0., 0.}}, 1},
                               {{{1., 1., 1.}}, 2},
                           }),
                           make_reference_solution<int>({}, {0, 0, 0}));

    ARBORX_TEST_QUERY_TREE_WITH_DISTANCE(
        empty_tree,
        makeNearestQueries<device_type>({
            {{{0., 0., 0.}}, 1},
            {{{1., 1., 1.}}, 2},
        }),
        (make_reference_solution<Kokkos::pair<int, float>>)({}, {0, 0, 0}));
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_single_leaf_tree, Tree, TreeTypes)
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

  ARBORX_TEST_QUERY_TREE(single_leaf_tree,
                         makeIntersectsBoxQueries<device_type>({}),
                         make_reference_solution<int>({}, {0}));

  ARBORX_TEST_QUERY_TREE(single_leaf_tree,
                         makeIntersectsSphereQueries<device_type>({}),
                         make_reference_solution<int>({}, {0}));

  ARBORX_TEST_QUERY_TREE(single_leaf_tree, makeNearestQueries<device_type>({}),
                         make_reference_solution<int>({}, {0}));

  ARBORX_TEST_QUERY_TREE_WITH_DISTANCE(
      single_leaf_tree, makeNearestQueries<device_type>({}),
      (make_reference_solution<Kokkos::pair<int, float>>)({}, {0}));

  ARBORX_TEST_QUERY_TREE_WITH_DISTANCE(
      single_leaf_tree,
      makeNearestQueries<device_type>({{{0., 0., 0.}, 3}, {{4., 5., 1.}, 1}}),
      (make_reference_solution<Kokkos::pair<int, float>>)({{0, 0.}, {0, 5.}},
                                                          {0, 1, 2}));

  ARBORX_TEST_QUERY_TREE(single_leaf_tree,
                         makeIntersectsBoxQueries<device_type>({
                             {{{5., 5., 5.}}, {{5., 5., 5.}}},
                             {{{.5, .5, .5}}, {{.5, .5, .5}}},
                         }),
                         make_reference_solution<int>({0}, {0, 0, 1}));

  ARBORX_TEST_QUERY_TREE(single_leaf_tree,
                         makeIntersectsSphereQueries<device_type>({
                             {{{0., 0., 0.}}, 1.},
                             {{{1., 1., 1.}}, 3.},
                             {{{5., 5., 5.}}, 2.},
                         }),
                         make_reference_solution<int>({0, 0}, {0, 1, 2, 2}));

  ARBORX_TEST_QUERY_TREE(single_leaf_tree,
                         makeNearestQueries<device_type>({
                             {{{0., 0., 0.}}, 1},
                             {{{1., 1., 1.}}, 2},
                             {{{2., 2., 2.}}, 3},
                         }),
                         make_reference_solution<int>({0, 0, 0}, {0, 1, 2, 3}));

  ARBORX_TEST_QUERY_TREE_WITH_DISTANCE(
      single_leaf_tree,
      makeNearestQueries<device_type>({
          {{{1., 0., 0.}}, 1},
          {{{0., 2., 0.}}, 2},
          {{{0., 0., 3.}}, 3},
      }),
      (make_reference_solution<Kokkos::pair<int, float>>)({{0, 0.f},
                                                           {0, 1.f},
                                                           {0, 2.f}},
                                                          {0, 1, 2, 3}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_couple_leaves_tree, Tree, TreeTypes)
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
  ARBORX_TEST_QUERY_TREE(couple_leaves_tree,
                         makeIntersectsBoxQueries<device_type>({
                             {},
                         }),
                         make_reference_solution<int>({}, {0, 0}));

  // single query intersects with both
  ARBORX_TEST_QUERY_TREE(couple_leaves_tree,
                         makeIntersectsBoxQueries<device_type>({
                             {{{0., 0., 0.}}, {{1., 1., 1.}}},
                         }),
                         make_reference_solution<int>({1, 0}, {0, 2}));

  // single query intersects with only one
  ARBORX_TEST_QUERY_TREE(couple_leaves_tree,
                         makeIntersectsBoxQueries<device_type>({
                             {{{0.5, 0.5, 0.5}}, {{1.5, 1.5, 1.5}}},
                         }),
                         make_reference_solution<int>({1}, {0, 1}));

  // a couple queries both intersect with nothing
  ARBORX_TEST_QUERY_TREE(couple_leaves_tree,
                         makeIntersectsBoxQueries<device_type>({
                             {},
                             {},
                         }),
                         make_reference_solution<int>({}, {0, 0, 0}));

  // a couple queries first intersects with nothing second with only one
  ARBORX_TEST_QUERY_TREE(couple_leaves_tree,
                         makeIntersectsBoxQueries<device_type>({
                             {},
                             {{{0., 0., 0.}}, {{0., 0., 0.}}},
                         }),
                         make_reference_solution<int>({0}, {0, 0, 1}));

  // no query
  ARBORX_TEST_QUERY_TREE(couple_leaves_tree,
                         makeIntersectsBoxQueries<device_type>({}),
                         make_reference_solution<int>({}, {0}));

  ARBORX_TEST_QUERY_TREE_WITH_DISTANCE(
      couple_leaves_tree,
      makeNearestQueries<device_type>({
          {{{0., 0., 0.}}, 2},
          {{{1., 0., 0.}}, 4},
      }),
      (make_reference_solution<Kokkos::pair<int, float>>)({{0, 0.f},
                                                           {1, sqrt(3.f)},
                                                           {0, 1.f},
                                                           {1, sqrt(2.f)}},
                                                          {0, 2, 4}));
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

  ARBORX_TEST_QUERY_TREE(
      bvh,
      makeIntersectsSphereQueries<DeviceType>({
          {{{0., 0., 0.}}, 1.},
          {{{1., 1., 1.}}, 1.},
          {{{.5, .5, .5}}, 1.},
      }),
      make_reference_solution<int>({0, 1, 2, 3, 0, 1, 2, 3}, {0, 1, 4, 8}));
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

  std::vector<int> indices_ref = {0, 1, 2, 3};
  std::vector<int> offset_ref = {0, 0, 4, 4};
  auto checkResultsAreFine = [&indices, &offset, &indices_ref,
                              &offset_ref]() -> void {
    auto indices_host = Kokkos::create_mirror_view(indices);
    Kokkos::deep_copy(indices_host, indices);
    auto offset_host = Kokkos::create_mirror_view(offset);
    Kokkos::deep_copy(offset_host, offset);
    BOOST_TEST(make_compressed_storage(offset_host, indices_host) ==
                   make_compressed_storage(offset_ref, indices_ref),
               tt::per_element());
  };

  BOOST_CHECK_NO_THROW(bvh.query(queries, indices, offset));
  checkResultsAreFine();

  using ExecutionSpace = typename DeviceType::execution_space;
  // compute number of results per query
  auto counts = ArborX::cloneWithoutInitializingNorCopying(offset);
  ArborX::adjacentDifference(ExecutionSpace{}, offset, counts);
  // extract optimal buffer size
  auto const max_results_per_query = ArborX::max(ExecutionSpace{}, counts);
  BOOST_TEST(max_results_per_query == 4);

  // optimal size
  BOOST_CHECK_NO_THROW(
      bvh.query(queries, indices, offset,
                ArborX::Experimental::TraversalPolicy().setBufferSize(
                    -max_results_per_query)));
  checkResultsAreFine();

  // buffer size insufficient
  BOOST_TEST(max_results_per_query > 1);
  BOOST_CHECK_NO_THROW(
      bvh.query(queries, indices, offset,
                ArborX::Experimental::TraversalPolicy().setBufferSize(+1)));
  checkResultsAreFine();
  BOOST_CHECK_THROW(
      bvh.query(queries, indices, offset,
                ArborX::Experimental::TraversalPolicy().setBufferSize(-1)),
      ArborX::SearchException);

  // adequate buffer size
  BOOST_TEST(max_results_per_query < 5);
  BOOST_CHECK_NO_THROW(
      bvh.query(queries, indices, offset,
                ArborX::Experimental::TraversalPolicy().setBufferSize(+5)));
  checkResultsAreFine();
  BOOST_CHECK_NO_THROW(
      bvh.query(queries, indices, offset,
                ArborX::Experimental::TraversalPolicy().setBufferSize(-5)));
  checkResultsAreFine();

  // passing null size skips the buffer optimization and never throws
  BOOST_CHECK_NO_THROW(
      bvh.query(queries, indices, offset,
                ArborX::Experimental::TraversalPolicy().setBufferSize(0)));
  checkResultsAreFine();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(unsorted_predicates, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  auto const bvh = make<ArborX::BVH<DeviceType>>({
      {{{0., 0., 0.}}, {{0., 0., 0.}}},
      {{{1., 1., 1.}}, {{1., 1., 1.}}},
      {{{2., 2., 2.}}, {{2., 2., 2.}}},
      {{{3., 3., 3.}}, {{3., 3., 3.}}},
  });

  using ViewType = Kokkos::View<int *, DeviceType>;
  ViewType indices("indices", 0);
  ViewType offset("offset", 0);

  std::vector<int> indices_ref = {2, 3, 0, 1};
  std::vector<int> offset_ref = {0, 2, 4};
  auto checkResultsAreFine = [&indices, &offset, &indices_ref,
                              &offset_ref]() -> void {
    auto indices_host = Kokkos::create_mirror_view(indices);
    Kokkos::deep_copy(indices_host, indices);
    auto offset_host = Kokkos::create_mirror_view(offset);
    Kokkos::deep_copy(offset_host, offset);
    BOOST_TEST(make_compressed_storage(offset_host, indices_host) ==
                   make_compressed_storage(offset_ref, indices_ref),
               tt::per_element());
  };

  {
    auto const queries = makeIntersectsBoxQueries<DeviceType>({
        {{{2., 2., 2.}}, {{3., 3., 3.}}},
        {{{0., 0., 0.}}, {{1., 1., 1.}}},
    });

    BOOST_CHECK_NO_THROW(bvh.query(
        queries, indices, offset,
        ArborX::Experimental::TraversalPolicy().setPredicateSorting(true)));
    checkResultsAreFine();

    BOOST_CHECK_NO_THROW(bvh.query(
        queries, indices, offset,
        ArborX::Experimental::TraversalPolicy().setPredicateSorting(false)));
    checkResultsAreFine();
  }

  indices_ref = {2, 3, 0, 1};
  {
    auto queries = makeNearestQueries<DeviceType>({
        {{{2.5, 2.5, 2.5}}, 2},
        {{{0.5, 0.5, 0.5}}, 2},
    });

    BOOST_CHECK_NO_THROW(bvh.query(
        queries, indices, offset,
        ArborX::Experimental::TraversalPolicy().setPredicateSorting(true)));
    checkResultsAreFine();

    BOOST_CHECK_NO_THROW(bvh.query(
        queries, indices, offset,
        ArborX::Experimental::TraversalPolicy().setPredicateSorting(false)));
    checkResultsAreFine();
  }
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

template <typename DeviceType>
struct CustomInlineCallbackSpatialPredicate
{
  using tag = ArborX::Details::InlineCallbackTag;
  Kokkos::View<ArborX::Point *, DeviceType> points;
  ArborX::Point const origin = {{0., 0., 0.}};
  template <typename Query, typename Insert>
  KOKKOS_FUNCTION void operator()(Query const &, int index,
                                  Insert const &insert) const
  {
    float const distance_to_origin =
        ArborX::Details::distance(points(index), origin);
    insert({index, distance_to_origin});
  }
};

template <typename DeviceType>
struct CustomPostCallbackSpatialPredicate
{
  using tag = ArborX::Details::PostCallbackTag;
  Kokkos::View<ArborX::Point *, DeviceType> points;
  ArborX::Point const origin = {{0., 0., 0.}};
  template <typename Predicates, typename InOutView, typename InView,
            typename OutView>
  void operator()(Predicates const &, InOutView &offset, InView in,
                  OutView &out) const
  {
    using ExecutionSpace = typename DeviceType::execution_space;
    using ArborX::Details::distance;
    auto const n = offset.extent(0) - 1;
    ArborX::reallocWithoutInitializing(out, in.extent(0));
    // NOTE woraround to avoid implicit capture of *this
    auto const &points_ = points;
    auto const &origin_ = origin;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
          for (int j = offset(i); j < offset(i + 1); ++j)
          {
            out(j) = {in(j), (float)distance(points_(in(j)), origin_)};
          }
        });
  }
};

template <typename DeviceType>
struct CustomInlineCallbackNearestPredicate
{
  using tag = ArborX::Details::InlineCallbackTag;
  template <typename Query, typename Insert>
  KOKKOS_FUNCTION void operator()(Query const &, int index, float distance,
                                  Insert const &insert) const
  {
    insert({index, (float)distance});
  }
};

template <typename DeviceType>
struct CustomPostCallbackNearestPredicate
{
  using tag = ArborX::Details::PostCallbackTag;
  template <typename Predicates, typename InOutView, typename InView,
            typename OutView>
  void operator()(Predicates const &, InOutView &offset, InView in,
                  OutView &out) const
  {
    using ExecutionSpace = typename DeviceType::execution_space;
    auto const n = offset.extent(0) - 1;
    ArborX::reallocWithoutInitializing(out, in.extent(0));
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0, n),
                         KOKKOS_LAMBDA(int i) {
                           for (int j = offset(i); j < offset(i + 1); ++j)
                           {
                             out(j) = {in(j).first, (float)in(j).second};
                           }
                         });
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(callback, DeviceType, ARBORX_DEVICE_TYPES)
{
  int const n = 10;
  Kokkos::View<ArborX::Point *, DeviceType> points("points", n);
  Kokkos::View<Kokkos::pair<int, float> *, DeviceType> ref("ref", n);
  ArborX::Point const origin = {{0., 0., 0.}};
  using ExecutionSpace = typename DeviceType::execution_space;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
        points(i) = {{(double)i, (double)i, (double)i}};
        ref(i) = {i, (float)ArborX::Details::distance(points(i), origin)};
      });
  ArborX::BVH<DeviceType> const bvh{points};
  {
    Kokkos::View<Kokkos::pair<int, float> *, DeviceType> custom("custom", 0);
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    bvh.query(makeIntersectsBoxQueries<DeviceType>({
                  bvh.bounds(),
              }),
              CustomInlineCallbackSpatialPredicate<DeviceType>{points}, custom,
              offset);

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
    Kokkos::View<Kokkos::pair<int, float> *, DeviceType> custom("custom", 0);
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    bvh.query(makeIntersectsBoxQueries<DeviceType>({
                  bvh.bounds(),
              }),
              CustomPostCallbackSpatialPredicate<DeviceType>{points}, custom,
              offset);

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
    Kokkos::View<Kokkos::pair<int, float> *, DeviceType> custom("custom", 0);
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    bvh.query(makeNearestQueries<DeviceType>({
                  {origin, n},
              }),
              CustomInlineCallbackNearestPredicate<DeviceType>{}, custom,
              offset);

    auto custom_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, custom);
    auto ref_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ref);
    BOOST_TEST(custom_host == ref_host, tt::per_element());
  }
  {
    Kokkos::View<Kokkos::pair<int, float> *, DeviceType> custom("custom", 0);
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    bvh.query(makeNearestQueries<DeviceType>({
                  {origin, n},
              }),
              CustomPostCallbackNearestPredicate<DeviceType>{}, custom, offset);

    auto custom_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, custom);
    auto ref_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ref);
    BOOST_TEST(custom_host == ref_host, tt::per_element());
  }
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
    float const distance_to_origin =
        ArborX::Details::distance(points(index), origin);

    auto data = ArborX::getData(query);
    insert({index, data + distance_to_origin});
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
          auto data_2 = ArborX::getData(queries(i));
          auto data = data_2[1];
          for (int j = offset(i); j < offset(i + 1); ++j)
          {
            out(j) = {in(j), data + (float)distance(points_(in(j)), origin_)};
          }
        });
  }
};

template <typename DeviceType>
struct CustomInlineCallbackAttachmentNearestPredicate
{
  using tag = ArborX::Details::InlineCallbackTag;
  template <typename Query, typename Insert>
  KOKKOS_FUNCTION void operator()(Query const &query, int index, float distance,
                                  Insert const &insert) const
  {
    auto data = ArborX::getData(query);
    insert({index, data + (float)distance});
  }
};

template <typename DeviceType>
struct CustomPostCallbackAttachmentNearestPredicate
{
  using tag = ArborX::Details::PostCallbackTag;
  template <typename Predicates, typename InOutView, typename InView,
            typename OutView>
  void operator()(Predicates const &queries, InOutView &offset, InView in,
                  OutView &out) const
  {
    using ExecutionSpace = typename DeviceType::execution_space;
    auto const n = offset.extent(0) - 1;
    ArborX::reallocWithoutInitializing(out, in.extent(0));
    Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0, n),
                         KOKKOS_LAMBDA(int i) {
                           auto data_2 = ArborX::getData(queries(i));
                           auto data = data_2[1];
                           for (int j = offset(i); j < offset(i + 1); ++j)
                           {
                             out(j) = {in(j).first, data + (float)in(j).second};
                           }
                         });
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(callback_with_attachment, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  int const n = 10;
  Kokkos::View<ArborX::Point *, DeviceType> points("points", n);
  Kokkos::View<Kokkos::pair<int, float> *, DeviceType> ref("ref", n);
  ArborX::Point const origin = {{0., 0., 0.}};
  float const delta = 5.0;
  using ExecutionSpace = typename DeviceType::execution_space;
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(
                                                                      int i) {
    points(i) = {{(double)i, (double)i, (double)i}};
    ref(i) = {i, delta + (float)ArborX::Details::distance(points(i), origin)};
  });
  ArborX::BVH<DeviceType> const bvh{points};
  {
    Kokkos::View<Kokkos::pair<int, float> *, DeviceType> custom("custom", 0);
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    bvh.query(
        makeIntersectsBoxWithAttachmentQueries<DeviceType, float>(
            {bvh.bounds()}, {delta}),
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
    Kokkos::View<Kokkos::pair<int, float> *, DeviceType> custom("custom", 0);
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    bvh.query(makeIntersectsBoxWithAttachmentQueries<DeviceType,
                                                     Kokkos::Array<float, 2>>(
                  {bvh.bounds()}, {{0., delta}}),
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
  {
    Kokkos::View<Kokkos::pair<int, float> *, DeviceType> custom("custom", 0);
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    bvh.query(makeNearestWithAttachmentQueries<DeviceType, float>({{origin, n}},
                                                                  {delta}),
              CustomInlineCallbackAttachmentNearestPredicate<DeviceType>{},
              custom, offset);

    auto custom_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, custom);
    auto ref_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ref);
    BOOST_TEST(custom_host == ref_host, tt::per_element());
  }
  {
    Kokkos::View<Kokkos::pair<int, float> *, DeviceType> custom("custom", 0);
    Kokkos::View<int *, DeviceType> offset("offset", 0);
    bvh.query(
        makeNearestWithAttachmentQueries<DeviceType, Kokkos::Array<float, 2>>(
            {{origin, n}}, {{0, delta}}),
        CustomPostCallbackAttachmentNearestPredicate<DeviceType>{}, custom,
        offset);

    auto custom_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, custom);
    auto ref_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ref);
    BOOST_TEST(custom_host == ref_host, tt::per_element());
  }
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
  Kokkos::parallel_for(
      "fill_bounding_boxes", Kokkos::RangePolicy<ExecutionSpace>(0, nx),
      KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < ny; ++j)
          for (int k = 0; k < nz; ++k)
          {
            ArborX::Point p{
                {i * Lx / (nx - 1), j * Ly / (ny - 1), k * Lz / (nz - 1)}};
            bounding_boxes[i + j * nx + k * (nx * ny)] = {p, p};
          }
      });

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
        // fill in reference solution to check against the collision
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

    auto const i = static_cast<int>(std::round(x / Lx * (nx - 1)));
    auto const j = static_cast<int>(std::round(y / Ly * (ny - 1)));
    auto const k = static_cast<int>(std::round(z / Lz * (nz - 1)));
    // Save the indices for the check
    ref[l] = {ind(i, j, k)};
  }

  Kokkos::deep_copy(bounding_boxes, bounding_boxes_host);
  Kokkos::parallel_for("fill_first_neighbors_queries",
                       Kokkos::RangePolicy<ExecutionSpace>(0, n),
                       KOKKOS_LAMBDA(int i) {
                         queries[i] = ArborX::intersects(bounding_boxes[i]);
                       });
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
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k)
      {
        cloud[ind(i, j, k)] = {
            {i * Lx / (nx - 1), j * Ly / (ny - 1), k * Lz / (nz - 1)}};
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
  // construct a cloud of points (nodes of a structured grid)
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
  auto within_queries_host = Kokkos::create_mirror_view(within_queries);
  Kokkos::deep_copy(within_queries_host, within_queries);

  ArborX::BVH<DeviceType> bvh(bounding_boxes);

  BoostExt::RTree<ArborX::Box> rtree(bounding_boxes_host);

  ARBORX_TEST_QUERY_TREE(bvh, nearest_queries,
                         query(rtree, nearest_queries_host));

  ARBORX_TEST_QUERY_TREE(bvh, within_queries,
                         query(rtree, within_queries_host));
}
