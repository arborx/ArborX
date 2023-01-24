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

#include <regex>

#include "Search_UnitTestHelpers.hpp"

BOOST_AUTO_TEST_SUITE(KokkosToolsDistributedAnnotations)

namespace tt = boost::test_tools;

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
        std::regex re("^(Testing::"
                      "|ArborX::DistributedTree::"
                      "|ArborX::BVH::"
                      "|ArborX::Sorting::"
                      ").*");
        BOOST_TEST(std::regex_match(label, re),
                   "\"" << label << "\" does not match the regular expression");
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
        std::regex re("^(Testing::"
                      "|ArborX::DistributedTree::query::"
                      "|ArborX::Distributor::"
                      "|ArborX::BVH::query::"
                      "|ArborX::TreeTraversal::spatial::"
                      "|ArborX::TreeTraversal::nearest::"
                      "|ArborX::CrsGraphWrapper::"
                      "|ArborX::Sorting::"
                      "|Kokkos::"
                      ").*");
        BOOST_TEST(std::regex_match(label, re),
                   "\"" << label << "\" does not match the regular expression");
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
    std::regex re("^(ArborX::|Kokkos::).*");
    BOOST_TEST(std::regex_match(label, re),
               "\"" << label << "\" does not match the regular expression");
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
    std::regex re("^(ArborX::|Kokkos::).*");
    BOOST_TEST(std::regex_match(label, re),
               "\"" << label << "\" does not match the regular expression");
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
