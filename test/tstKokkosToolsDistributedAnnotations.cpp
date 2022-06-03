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

#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include <ArborX_DistributedTree.hpp>

#include <boost/test/unit_test.hpp>

#include <string>

#include "Search_UnitTestHelpers.hpp"

BOOST_AUTO_TEST_SUITE(KokkosToolsDistributedAnnotations)

namespace tt = boost::test_tools;

bool isPrefixedWith(std::string const &s, std::string const &prefix)
{
  return s.find(prefix) == 0;
}

BOOST_AUTO_TEST_CASE(is_prefixed_with)
{
  BOOST_TEST(isPrefixedWith("ArborX::Whatever", "ArborX"));
  BOOST_TEST(!isPrefixedWith("Nope", "ArborX"));
  BOOST_TEST(!isPrefixedWith("Nope::ArborX", "ArborX"));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(
    distributed_search_tree_distributed_search_tree_allocations_prefixed,
    DeviceType, ARBORX_DEVICE_TYPES)
{
  auto tree = makeDistributedTree<DeviceType>(MPI_COMM_WORLD,
                                              {
                                                  {{{0, 0, 0}}, {{1, 1, 1}}},
                                                  {{{0, 0, 0}}, {{1, 1, 1}}},
                                              });

  Kokkos::Tools::Experimental::set_allocate_data_callback(
      [](Kokkos::Profiling::SpaceHandle /*handle*/, char const *label,
         void const * /*ptr*/, uint64_t /*size*/) {
        BOOST_TEST_MESSAGE(label);
        BOOST_TEST((isPrefixedWith(label, "ArborX::DistributedTree::") ||
                    isPrefixedWith(label, "ArborX::BVH::") ||
                    isPrefixedWith(label, "ArborX::Sorting::") ||
                    isPrefixedWith(label, "Testing::")));
      });

  { // one leaf per process
    auto tree = makeDistributedTree<DeviceType>(MPI_COMM_WORLD,
                                                {
                                                    {{{0, 0, 0}}, {{1, 1, 1}}},
                                                });
  }

  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(
    distributed_search_tree_query_allocations_prefixed, DeviceType,
    ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  auto tree = makeDistributedTree<DeviceType>(MPI_COMM_WORLD,
                                              {
                                                  {{{0, 0, 0}}, {{1, 1, 1}}},
                                                  {{{0, 0, 0}}, {{1, 1, 1}}},
                                              });

  Kokkos::Tools::Experimental::set_allocate_data_callback(
      [](Kokkos::Profiling::SpaceHandle /*handle*/, char const *label,
         void const * /*ptr*/, uint64_t /*size*/) {
        BOOST_TEST_MESSAGE(label);
        BOOST_TEST((isPrefixedWith(label, "ArborX::DistributedTree::query::") ||
                    isPrefixedWith(label, "ArborX::Distributor::") ||
                    isPrefixedWith(label, "ArborX::BVH::query::") ||
                    isPrefixedWith(label, "ArborX::TreeTraversal::spatial::") ||
                    isPrefixedWith(label, "ArborX::TreeTraversal::nearest::") ||
                    isPrefixedWith(label, "ArborX::CrsGraphWrapper::") ||
                    isPrefixedWith(label, "ArborX::Sorting::") ||
                    isPrefixedWith(label, "Kokkos::SortImpl::") ||
                    isPrefixedWith(label, "Testing::")));
      });

  // spatial predicates
  query(ExecutionSpace{}, tree,
        makeIntersectsBoxQueries<DeviceType>({
            {{{0, 0, 0}}, {{1, 1, 1}}},
            {{{0, 0, 0}}, {{1, 1, 1}}},
        }));

  // nearest predicates
  query(ExecutionSpace{}, tree,
        makeNearestQueries<DeviceType>({
            {{{0, 0, 0}}, 1},
            {{{0, 0, 0}}, 2},
        }));

  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(kernels_prefixed, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  auto const callback = [](char const *label, uint32_t, uint64_t *) {
    BOOST_TEST_MESSAGE(label);
    BOOST_TEST((isPrefixedWith(label, "ArborX::") ||
                isPrefixedWith(label, "Kokkos::")));
  };
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(callback);
  Kokkos::Tools::Experimental::set_begin_parallel_scan_callback(callback);
  Kokkos::Tools::Experimental::set_begin_parallel_reduce_callback(callback);

  // DistributedTree::query

  auto tree = makeDistributedTree<DeviceType>(MPI_COMM_WORLD,
                                              {
                                                  {{{0, 0, 0}}, {{1, 1, 1}}},
                                                  {{{0, 0, 0}}, {{1, 1, 1}}},
                                              });

  // spatial predicates
  query(ExecutionSpace{}, tree,
        makeIntersectsBoxQueries<DeviceType>({
            {{{0, 0, 0}}, {{1, 1, 1}}},
            {{{0, 0, 0}}, {{1, 1, 1}}},
        }));

  // nearest predicates
  query(ExecutionSpace{}, tree,
        makeNearestQueries<DeviceType>({
            {{{0, 0, 0}}, 1},
            {{{0, 0, 0}}, 2},
        }));

  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(nullptr);
  Kokkos::Tools::Experimental::set_begin_parallel_scan_callback(nullptr);
  Kokkos::Tools::Experimental::set_begin_parallel_reduce_callback(nullptr);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(regions_prefixed, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  Kokkos::Tools::Experimental::set_push_region_callback([](char const *label) {
    BOOST_TEST_MESSAGE(label);
    BOOST_TEST((isPrefixedWith(label, "ArborX::") ||
                isPrefixedWith(label, "Kokkos::")));
  });

  // DistributedTree::DistriibutedSearchTree

  { // empty
    auto tree = makeDistributedTree<DeviceType>(MPI_COMM_WORLD, {});
  }

  // DistributedTree::query

  auto tree = makeDistributedTree<DeviceType>(MPI_COMM_WORLD,
                                              {
                                                  {{{0, 0, 0}}, {{1, 1, 1}}},
                                                  {{{0, 0, 0}}, {{1, 1, 1}}},
                                              });

  // spatial predicates
  query(ExecutionSpace{}, tree,
        makeIntersectsBoxQueries<DeviceType>({
            {{{0, 0, 0}}, {{1, 1, 1}}},
            {{{0, 0, 0}}, {{1, 1, 1}}},
        }));

  // nearest predicates
  query(ExecutionSpace{}, tree,
        makeNearestQueries<DeviceType>({
            {{{0, 0, 0}}, 1},
            {{{0, 0, 0}}, 2},
        }));

  Kokkos::Tools::Experimental::set_push_region_callback(nullptr);
}

BOOST_AUTO_TEST_SUITE_END()
