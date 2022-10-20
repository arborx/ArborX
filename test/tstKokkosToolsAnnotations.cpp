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
#include <ArborX_LinearBVH.hpp>

#include <boost/test/unit_test.hpp>

#include <regex>

#include "Search_UnitTestHelpers.hpp"

BOOST_AUTO_TEST_SUITE(KokkosToolsAnnotations)

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE(is_prefixed_with)
{
  std::regex re("^ArborX::.*");
  BOOST_TEST(std::regex_match("ArborX::Whatever", re));
  BOOST_TEST(!std::regex_match("Nope", re));
  BOOST_TEST(!std::regex_match("Nope::ArborX", re));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(bvh_bvh_allocations_prefixed, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using Tree = ArborX::BVH<typename DeviceType::memory_space>;
  using ExecutionSpace = typename DeviceType::execution_space;

  Kokkos::Tools::Experimental::set_allocate_data_callback(
      [](Kokkos::Profiling::SpaceHandle /*handle*/, char const *label,
         void const * /*ptr*/, uint64_t /*size*/) {
        std::regex re("^(Testing::"
                      "|ArborX::BVH::"
                      "|ArborX::Sorting::"
                      "|Kokkos::SortImpl::BinSortFunctor::"
                      "|Kokkos::Serial::" // unsure what's going on
                      ").*");
        BOOST_TEST(std::regex_match(label, re),
                   "\"" << label << "\" does not match the regular expression");
      });

  { // default constructed
    Tree tree;
  }

  { // empty
    auto tree = make<Tree>(ExecutionSpace{}, {});
  }

  { // one leaf
    auto tree = make<Tree>(ExecutionSpace{}, {
                                                 {{{0, 0, 0}}, {{1, 1, 1}}},
                                             });
  }

  { // two leaves
    auto tree = make<Tree>(ExecutionSpace{}, {
                                                 {{{0, 0, 0}}, {{1, 1, 1}}},
                                                 {{{0, 0, 0}}, {{1, 1, 1}}},
                                             });
  }

  Kokkos::Tools::Experimental::set_allocate_data_callback(nullptr);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(bvh_query_allocations_prefixed, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  auto tree = make<ArborX::BVH<typename DeviceType::memory_space>>(
      ExecutionSpace{}, {
                            {{{0, 0, 0}}, {{1, 1, 1}}},
                            {{{0, 0, 0}}, {{1, 1, 1}}},
                        });

  Kokkos::Tools::Experimental::set_allocate_data_callback(
      [](Kokkos::Profiling::SpaceHandle /*handle*/, char const *label,
         void const * /*ptr*/, uint64_t /*size*/) {
        std::regex re("^(Testing::"
                      "|ArborX::BVH::query::"
                      "|ArborX::TreeTraversal::spatial::"
                      "|ArborX::TreeTraversal::nearest::"
                      "|ArborX::CrsGraphWrapper::"
                      "|ArborX::Sorting::"
                      "|Kokkos::SortImpl::BinSortFunctor::"
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
  using Tree = ArborX::BVH<typename DeviceType::memory_space>;
  using ExecutionSpace = typename DeviceType::execution_space;

  auto const callback = [](char const *label, uint32_t, uint64_t *) {
    std::regex re("^(ArborX::|Kokkos::).*");
    BOOST_TEST(std::regex_match(label, re),
               "\"" << label << "\" does not match the regular expression");
  };
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(callback);
  Kokkos::Tools::Experimental::set_begin_parallel_scan_callback(callback);
  Kokkos::Tools::Experimental::set_begin_parallel_reduce_callback(callback);

  // BVH::BVH

  { // default constructed
    Tree tree;
  }

  { // empty
    auto tree = make<Tree>(ExecutionSpace{}, {});
  }

  { // one leaf
    auto tree = make<Tree>(ExecutionSpace{}, {
                                                 {{{0, 0, 0}}, {{1, 1, 1}}},
                                             });
  }

  { // two leaves
    auto tree = make<Tree>(ExecutionSpace{}, {
                                                 {{{0, 0, 0}}, {{1, 1, 1}}},
                                                 {{{0, 0, 0}}, {{1, 1, 1}}},
                                             });
  }

  // BVH::query

  auto tree = make<Tree>(ExecutionSpace{}, {
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
  using Tree = ArborX::BVH<typename DeviceType::memory_space>;
  using ExecutionSpace = typename DeviceType::execution_space;

  Kokkos::Tools::Experimental::set_push_region_callback([](char const *label) {
    std::regex re("^(ArborX::|Kokkos::).*");
    BOOST_TEST(std::regex_match(label, re),
               "\"" << label << "\" does not match the regular expression");
  });

  // BVH::BVH

  { // default constructed
    Tree tree;
  }

  { // empty
    auto tree = make<Tree>(ExecutionSpace{}, {});
  }

  { // one leaf
    auto tree = make<Tree>(ExecutionSpace{}, {
                                                 {{{0, 0, 0}}, {{1, 1, 1}}},
                                             });
  }

  { // two leaves
    auto tree = make<Tree>(ExecutionSpace{}, {
                                                 {{{0, 0, 0}}, {{1, 1, 1}}},
                                                 {{{0, 0, 0}}, {{1, 1, 1}}},
                                             });
  }

  // BVH::query

  auto tree = make<Tree>(ExecutionSpace{}, {
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
