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

#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include <ArborX_DistributedSearchTree.hpp>

#include <boost/test/unit_test.hpp>

#include <string>

#include "Search_UnitTestHelpers.hpp"

#if (KOKKOS_VERSION >= 30200) // callback registriation from within the program
                              // was added in Kokkkos v3.2

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
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      [](Kokkos::Profiling::SpaceHandle handle, const char *label,
         void const *ptr, uint64_t size) {
        std::cout << label << '\n';
        BOOST_TEST((isPrefixedWith(label, "ArborX::DistributedSearchTree::") ||
                    isPrefixedWith(label, "ArborX::BVH::") ||
                    isPrefixedWith(label, "ArborX::Sorting::") ||
                    isPrefixedWith(label, "Testing::")));
      });

  { // one leaf per process
    auto tree = makeDistributedSearchTree<DeviceType>(
        MPI_COMM_WORLD, {
                            {{{0, 0, 0}}, {{1, 1, 1}}},
                        });
  }

  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(
    distributed_search_tree_query_allocations_prefixed, DeviceType,
    ARBORX_DEVICE_TYPES)
{
  auto tree = makeDistributedSearchTree<DeviceType>(
      MPI_COMM_WORLD, {
                          {{{0, 0, 0}}, {{1, 1, 1}}},
                          {{{0, 0, 0}}, {{1, 1, 1}}},
                      });

  Kokkos::Tools::Experimental::set_allocate_data_callback(
      [](Kokkos::Profiling::SpaceHandle handle, const char *label,
         void const *ptr, uint64_t size) {
        std::cout << label << '\n';
        BOOST_TEST(
            (isPrefixedWith(label, "ArborX::DistributedSearchTree::query::") ||
             isPrefixedWith(label, "ArborX::Distributor::") ||
             isPrefixedWith(label, "ArborX::BVH::query::") ||
             isPrefixedWith(label, "ArborX::BufferOptimization::") ||
             isPrefixedWith(label, "ArborX::Sorting::") ||
             isPrefixedWith(label, "Kokkos::SortImpl::") ||
             isPrefixedWith(label, "Testing::")));
      });

  // spatial predicates
  query(tree, makeIntersectsBoxQueries<DeviceType>({
                  {{{0, 0, 0}}, {{1, 1, 1}}},
                  {{{0, 0, 0}}, {{1, 1, 1}}},
              }));

  // nearest predicates
  query(tree, makeNearestQueries<DeviceType>({
                  {{{0, 0, 0}}, 1},
                  {{{0, 0, 0}}, 2},
              }));

  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(regions_prefixed, DeviceType, ARBORX_DEVICE_TYPES)
{
  Kokkos::Tools::Experimental::set_push_region_callback([](char const *label) {
    std::cout << label << '\n';
    BOOST_TEST((isPrefixedWith(label, "ArborX::") ||
                isPrefixedWith(label, "Kokkos::")));
  });

  // DistributedSearchTree::DistriibutedSearchTree

  { // empty
    auto tree = makeDistributedSearchTree<DeviceType>(MPI_COMM_WORLD, {});
  }

  // DistributedSearchTree::query

  auto tree = makeDistributedSearchTree<DeviceType>(
      MPI_COMM_WORLD, {
                          {{{0, 0, 0}}, {{1, 1, 1}}},
                          {{{0, 0, 0}}, {{1, 1, 1}}},
                      });

  // spatial predicates
  query(tree, makeIntersectsBoxQueries<DeviceType>({
                  {{{0, 0, 0}}, {{1, 1, 1}}},
                  {{{0, 0, 0}}, {{1, 1, 1}}},
              }));

  // nearest predicates
  query(tree, makeNearestQueries<DeviceType>({
                  {{{0, 0, 0}}, 1},
                  {{{0, 0, 0}}, 2},
              }));

  Kokkos::Tools::Experimental::set_push_region_callback(nullptr);
}

BOOST_AUTO_TEST_SUITE_END()

#endif
