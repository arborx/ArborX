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

#include <boost/test/unit_test.hpp>

#include <vector>

#include "Search_UnitTestHelpers.hpp"
// clang-format off
#include "ArborXTest_TreeTypeTraits.hpp"
// clang-format on

BOOST_AUTO_TEST_SUITE(Degenerate)

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE_TEMPLATE(empty_tree_spatial_predicate, TreeTypeTraits,
                              TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;

  // tree is empty, it has no leaves.
  Tree default_initialized;
  Tree value_initialized{};
  for (auto const &tree : {
           default_initialized, value_initialized,
           make<Tree>(ExecutionSpace{},
                      {}), // constructed with empty view of boxes
       })
  {
    BOOST_TEST(tree.empty());
    BOOST_TEST(tree.size() == 0);
    // Tree::bounds() returns an invalid box when the tree is empty.
    using ArborX::Details::equals;
    BOOST_TEST(equals(static_cast<ArborX::Box>(tree.bounds()), {}));

    // Passing a view with no query does seem a bit silly but we still need
    // to support it. And since the tag dispatching yields different tree
    // traversals for nearest and spatial predicates, we do have to check
    // the results for various type of queries.
    ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                           makeIntersectsBoxQueries<DeviceType>({}),
                           make_reference_solution<int>({}, {0}));

    // Now passing a couple queries of various type and checking the
    // results.
    ARBORX_TEST_QUERY_TREE(
        ExecutionSpace{}, tree,
        makeIntersectsBoxQueries<DeviceType>({
            {}, // Did not bother giving a valid box here but that's fine.
            {},
        }),
        make_reference_solution<int>({}, {0, 0, 0}));

#ifndef ARBORX_TEST_DISABLE_SPATIAL_QUERY_INTERSECTS_SPHERE
    // NOTE: Admittedly testing for both intersection with a box and with a
    // sphere queries might be a bit overkill but I'd rather test for all the
    // queries we plan on using.
    ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                           makeIntersectsSphereQueries<DeviceType>({}),
                           make_reference_solution<int>({}, {0}));

    ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                           makeIntersectsSphereQueries<DeviceType>({
                               {{{0., 0., 0.}}, 1.},
                               {{{1., 1., 1.}}, 2.},
                           }),
                           make_reference_solution<int>({}, {0, 0, 0}));
#endif
  }
}

#ifndef ARBORX_TEST_DISABLE_NEAREST_QUERY
BOOST_AUTO_TEST_CASE_TEMPLATE(empty_tree_nearest_predicate, TreeTypeTraits,
                              TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;

  // tree is empty, it has no leaves.
  for (auto const &tree : {
           Tree{}, // default constructed
           make<Tree>(ExecutionSpace{},
                      {}), // constructed with empty view of boxes
       })
  {
    BOOST_TEST(tree.empty());
    BOOST_TEST(tree.size() == 0);
    // Tree::bounds() returns an invalid box when the tree is empty.
    using ArborX::Details::equals;
    BOOST_TEST(equals(static_cast<ArborX::Box>(tree.bounds()), {}));

    // Passing a view with no query does seem a bit silly but we still need
    // to support it. And since the tag dispatching yields different tree
    // traversals for nearest and spatial predicates, we do have to check
    // the results for various type of queries.
    ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                           makeNearestQueries<DeviceType>({}),
                           make_reference_solution<int>({}, {0}));

    // Now passing a couple queries of various type and checking the
    // results.
    ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                           makeNearestQueries<DeviceType>({
                               {{{0., 0., 0.}}, 1},
                               {{{1., 1., 1.}}, 2},
                           }),
                           make_reference_solution<int>({}, {0, 0, 0}));
  }
}
#endif

BOOST_AUTO_TEST_CASE_TEMPLATE(single_leaf_tree_spatial_predicate,
                              TreeTypeTraits, TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;

  // tree has a single leaf (unit box)
  auto const tree =
      make<Tree>(ExecutionSpace{}, {
                                       {{{0., 0., 0.}}, {{1., 1., 1.}}},
                                   });

  BOOST_TEST(!tree.empty());
  BOOST_TEST(tree.size() == 1);
  using ArborX::Details::equals;
  BOOST_TEST(equals(static_cast<ArborX::Box>(tree.bounds()),
                    {{{0., 0., 0.}}, {{1., 1., 1.}}}));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeIntersectsBoxQueries<DeviceType>({}),
                         make_reference_solution<int>({}, {0}));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeIntersectsBoxQueries<DeviceType>({
                             {{{5., 5., 5.}}, {{5., 5., 5.}}},
                             {{{.5, .5, .5}}, {{.5, .5, .5}}},
                         }),
                         make_reference_solution<int>({0}, {0, 0, 1}));

#ifndef ARBORX_TEST_DISABLE_SPATIAL_QUERY_INTERSECTS_SPHERE
  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeIntersectsSphereQueries<DeviceType>({}),
                         make_reference_solution<int>({}, {0}));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeIntersectsSphereQueries<DeviceType>({
                             {{{0., 0., 0.}}, 1.},
                             {{{1., 1., 1.}}, 3.},
                             {{{5., 5., 5.}}, 2.},
                         }),
                         make_reference_solution<int>({0, 0}, {0, 1, 2, 2}));
#endif
}

#ifndef ARBORX_TEST_DISABLE_NEAREST_QUERY
BOOST_AUTO_TEST_CASE_TEMPLATE(single_leaf_tree_nearest_predicate,
                              TreeTypeTraits, TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;

  // tree has a single leaf (unit box)
  auto const tree =
      make<Tree>(ExecutionSpace{}, {
                                       {{{0., 0., 0.}}, {{1., 1., 1.}}},
                                   });

  BOOST_TEST(!tree.empty());
  BOOST_TEST(tree.size() == 1);
  using ArborX::Details::equals;
  BOOST_TEST(equals(static_cast<ArborX::Box>(tree.bounds()),
                    {{{0., 0., 0.}}, {{1., 1., 1.}}}));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeNearestQueries<DeviceType>({}),
                         make_reference_solution<int>({}, {0}));
  ARBORX_TEST_QUERY_TREE(
      ExecutionSpace{}, tree,
      makeNearestQueries<DeviceType>({{{0., 0., 0.}, 3}, {{4., 5., 1.}, 1}}),
      make_reference_solution<int>({0, 0}, {0, 1, 2}));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeNearestQueries<DeviceType>({
                             {{{0., 0., 0.}}, 1},
                             {{{1., 1., 1.}}, 2},
                             {{{2., 2., 2.}}, 3},
                         }),
                         make_reference_solution<int>({0, 0, 0}, {0, 1, 2, 3}));
}
#endif

// FIXME Tree with two leaves is not "degenerated".  Find a better place for it.
BOOST_AUTO_TEST_CASE_TEMPLATE(couple_leaves_tree_spatial_predicate,
                              TreeTypeTraits, TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;

  auto const tree =
      make<Tree>(ExecutionSpace{}, {
                                       {{{0., 0., 0.}}, {{0., 0., 0.}}},
                                       {{{1., 1., 1.}}, {{1., 1., 1.}}},
                                   });

  BOOST_TEST(!tree.empty());
  BOOST_TEST(tree.size() == 2);
  using ArborX::Details::equals;
  BOOST_TEST(equals(static_cast<ArborX::Box>(tree.bounds()),
                    {{{0., 0., 0.}}, {{1., 1., 1.}}}));

  // single query intersects with nothing
  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeIntersectsBoxQueries<DeviceType>({
                             {},
                         }),
                         make_reference_solution<int>({}, {0, 0}));

  // single query intersects with both
  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeIntersectsBoxQueries<DeviceType>({
                             {{{0., 0., 0.}}, {{1., 1., 1.}}},
                         }),
                         make_reference_solution<int>({1, 0}, {0, 2}));

  // single query intersects with only one
  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeIntersectsBoxQueries<DeviceType>({
                             {{{0.5, 0.5, 0.5}}, {{1.5, 1.5, 1.5}}},
                         }),
                         make_reference_solution<int>({1}, {0, 1}));

  // a couple queries both intersect with nothing
  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeIntersectsBoxQueries<DeviceType>({
                             {},
                             {},
                         }),
                         make_reference_solution<int>({}, {0, 0, 0}));

  // a couple queries first intersects with nothing second with only one
  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeIntersectsBoxQueries<DeviceType>({
                             {},
                             {{{0., 0., 0.}}, {{0., 0., 0.}}},
                         }),
                         make_reference_solution<int>({0}, {0, 0, 1}));

  // no query
  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeIntersectsBoxQueries<DeviceType>({}),
                         make_reference_solution<int>({}, {0}));
}

#ifndef ARBORX_TEST_DISABLE_NEAREST_QUERY
BOOST_AUTO_TEST_CASE_TEMPLATE(couple_leaves_tree_nearest_predicate,
                              TreeTypeTraits, TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;

  auto const tree =
      make<Tree>(ExecutionSpace{}, {
                                       {{{0., 0., 0.}}, {{0., 0., 0.}}},
                                       {{{1., 1., 1.}}, {{1., 1., 1.}}},
                                   });

  BOOST_TEST(!tree.empty());
  BOOST_TEST(tree.size() == 2);
  using ArborX::Details::equals;
  BOOST_TEST(equals(static_cast<ArborX::Box>(tree.bounds()),
                    {{{0., 0., 0.}}, {{1., 1., 1.}}}));

  // no query
  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeNearestQueries<DeviceType>({}),
                         make_reference_solution<int>({}, {0}));

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree,
                         makeNearestQueries<DeviceType>({
                             {{{0., 0., 0.}}, 2},
                             {{{1., 0., 0.}}, 4},
                         }),
                         make_reference_solution<int>({0, 1, 0, 1}, {0, 2, 4}));
}
#endif

#ifndef ARBORX_TEST_DISABLE_SPATIAL_QUERY_INTERSECTS_SPHERE
BOOST_AUTO_TEST_CASE_TEMPLATE(duplicated_leaves_spatial_predicate,
                              TreeTypeTraits, TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;

  // The tree contains multiple (more than two) leaves that will be assigned
  // the same Morton code.  This was able to trigger a bug that we discovered
  // when building trees over ~10M indexable values.  The hierarchy generated
  // at construction had leaves with no parent which yielded a segfault later
  // when computing bounding boxes and walking the hierarchy toward the root.
  auto const tree =
      make<Tree>(ExecutionSpace{}, {
                                       {{{0., 0., 0.}}, {{0., 0., 0.}}},
                                       {{{1., 1., 1.}}, {{1., 1., 1.}}},
                                       {{{1., 1., 1.}}, {{1., 1., 1.}}},
                                       {{{1., 1., 1.}}, {{1., 1., 1.}}},
                                   });

  ARBORX_TEST_QUERY_TREE(
      ExecutionSpace{}, tree,
      makeIntersectsSphereQueries<DeviceType>({
          {{{0., 0., 0.}}, 1.},
          {{{1., 1., 1.}}, 1.},
          {{{.5, .5, .5}}, 1.},
      }),
      make_reference_solution<int>({0, 1, 2, 3, 0, 1, 2, 3}, {0, 1, 4, 8}));
}
#endif

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(Miscellaneous)

BOOST_AUTO_TEST_CASE_TEMPLATE(not_exceeding_stack_capacity_spatial_predicate,
                              TreeTypeTraits, TreeTypeTraitsList)
{
  // FIXME This unit test might make little sense for other trees than BVH
  // using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;

  std::vector<ArborX::Box> boxes;
  int const n = 4096; // exceeds stack capacity which is 64
  boxes.reserve(n);
  for (int i = 0; i < n; ++i)
  {
    float const a = i;
    float const b = i + 1;
    boxes.push_back({{{a, a, a}}, {{b, b, b}}});
  }
  ExecutionSpace space;
  auto const bvh =
      make<ArborX::BVH<typename DeviceType::memory_space>>(space, boxes);

  Kokkos::View<int *, DeviceType> indices("indices", 0);
  Kokkos::View<int *, DeviceType> offset("offset", 0);
  // spatial query that is satisfied by all leaves in the tree
  BOOST_CHECK_NO_THROW(ArborX::query(bvh, space,
                                     makeIntersectsBoxQueries<DeviceType>({
                                         {},
                                         {{{0., 0., 0.}}, {{n, n, n}}},
                                     }),
                                     indices, offset));
  BOOST_TEST(KokkosExt::lastElement(space, offset) == n);
}

#ifndef ARBORX_TEST_DISABLE_NEAREST_QUERY
BOOST_AUTO_TEST_CASE_TEMPLATE(not_exceeding_stack_capacity_nearest_predicate,
                              TreeTypeTraits, TreeTypeTraitsList)
{
  // FIXME This unit test might make little sense for other trees than BVH
  // using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;

  std::vector<ArborX::Box> boxes;
  int const n = 4096; // exceed stack capacity which is 64
  boxes.reserve(n);
  for (int i = 0; i < n; ++i)
  {
    float const a = i;
    float const b = i + 1;
    boxes.push_back({{{a, a, a}}, {{b, b, b}}});
  }
  ExecutionSpace space;
  auto const bvh =
      make<ArborX::BVH<typename DeviceType::memory_space>>(space, boxes);

  Kokkos::View<int *, DeviceType> indices("indices", 0);
  Kokkos::View<int *, DeviceType> offset("offset", 0);
  // nearest query asking for as many neighbors as they are leaves in the tree
  BOOST_CHECK_NO_THROW(ArborX::query(bvh, space,
                                     makeNearestQueries<DeviceType>({
                                         {{{0., 0., 0.}}, n},
                                     }),
                                     indices, offset));
  BOOST_TEST(KokkosExt::lastElement(space, offset) == n);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
