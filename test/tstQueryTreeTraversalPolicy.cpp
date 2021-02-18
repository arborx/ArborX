/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include <ArborX_LinearBVH.hpp>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <iostream>
#include <random>
#include <tuple>

#include "Search_UnitTestHelpers.hpp"

BOOST_AUTO_TEST_SUITE(TraversalPolicy)

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE_TEMPLATE(buffer_optimization, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  auto const bvh = make<ArborX::BVH<typename DeviceType::memory_space>>(
      ExecutionSpace{}, {
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

  BOOST_CHECK_NO_THROW(
      ArborX::query(bvh, ExecutionSpace{}, queries, indices, offset));
  checkResultsAreFine();

  // compute number of results per query
  auto counts = ArborX::cloneWithoutInitializingNorCopying(offset);
  ArborX::adjacentDifference(ExecutionSpace{}, offset, counts);
  // extract optimal buffer size
  auto const max_results_per_query = ArborX::max(ExecutionSpace{}, counts);
  BOOST_TEST(max_results_per_query == 4);

  // optimal size
  BOOST_CHECK_NO_THROW(
      ArborX::query(bvh, ExecutionSpace{}, queries, indices, offset,
                    ArborX::Experimental::TraversalPolicy().setBufferSize(
                        -max_results_per_query)));
  checkResultsAreFine();

  // buffer size insufficient
  BOOST_TEST(max_results_per_query > 1);
  BOOST_CHECK_NO_THROW(
      ArborX::query(bvh, ExecutionSpace{}, queries, indices, offset,
                    ArborX::Experimental::TraversalPolicy().setBufferSize(+1)));
  checkResultsAreFine();
  BOOST_CHECK_THROW(
      ArborX::query(bvh, ExecutionSpace{}, queries, indices, offset,
                    ArborX::Experimental::TraversalPolicy().setBufferSize(-1)),
      ArborX::SearchException);

  // adequate buffer size
  BOOST_TEST(max_results_per_query < 5);
  BOOST_CHECK_NO_THROW(
      ArborX::query(bvh, ExecutionSpace{}, queries, indices, offset,
                    ArborX::Experimental::TraversalPolicy().setBufferSize(+5)));
  checkResultsAreFine();
  BOOST_CHECK_NO_THROW(
      ArborX::query(bvh, ExecutionSpace{}, queries, indices, offset,
                    ArborX::Experimental::TraversalPolicy().setBufferSize(-5)));
  checkResultsAreFine();

  // passing null size skips the buffer optimization and never throws
  BOOST_CHECK_NO_THROW(
      ArborX::query(bvh, ExecutionSpace{}, queries, indices, offset,
                    ArborX::Experimental::TraversalPolicy().setBufferSize(0)));
  checkResultsAreFine();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(unsorted_predicates, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  auto const bvh = make<ArborX::BVH<typename DeviceType::memory_space>>(
      ExecutionSpace{}, {
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

    BOOST_CHECK_NO_THROW(ArborX::query(
        bvh, ExecutionSpace{}, queries, indices, offset,
        ArborX::Experimental::TraversalPolicy().setPredicateSorting(true)));
    checkResultsAreFine();

    BOOST_CHECK_NO_THROW(ArborX::query(
        bvh, ExecutionSpace{}, queries, indices, offset,
        ArborX::Experimental::TraversalPolicy().setPredicateSorting(false)));
    checkResultsAreFine();
  }

  indices_ref = {2, 3, 0, 1};
  {
    auto queries = makeNearestQueries<DeviceType>({
        {{{2.5, 2.5, 2.5}}, 2},
        {{{0.5, 0.5, 0.5}}, 2},
    });

    BOOST_CHECK_NO_THROW(ArborX::query(
        bvh, ExecutionSpace{}, queries, indices, offset,
        ArborX::Experimental::TraversalPolicy().setPredicateSorting(true)));
    checkResultsAreFine();

    BOOST_CHECK_NO_THROW(ArborX::query(
        bvh, ExecutionSpace{}, queries, indices, offset,
        ArborX::Experimental::TraversalPolicy().setPredicateSorting(false)));
    checkResultsAreFine();
  }
}

BOOST_AUTO_TEST_SUITE_END()
