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

#include <string>

#include "Search_UnitTestHelpers.hpp"

#if (KOKKOS_VERSION >= 30200) // callback registriation from within the program
                              // was added in Kokkkos v3.2

BOOST_AUTO_TEST_SUITE(KokkosToolsAnnotations)

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

BOOST_AUTO_TEST_CASE_TEMPLATE(bvh_bvh_allocations_prefixed, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using Tree = ArborX::BVH<typename DeviceType::memory_space>;
  using ExecutionSpace = typename DeviceType::execution_space;

  Kokkos::Tools::Experimental::set_allocate_data_callback(
      [](Kokkos::Profiling::SpaceHandle /*handle*/, const char *label,
         void const * /*ptr*/, uint64_t /*size*/) {
        BOOST_TEST_MESSAGE(label);
        BOOST_TEST(
            (isPrefixedWith(label, "ArborX::BVH::") || // data member
             isPrefixedWith(label, "ArborX::BVH::BVH::") ||
             isPrefixedWith(label, "ArborX::Sorting::") ||
             isPrefixedWith(label, "Kokkos::SortImpl::BinSortFunctor::") ||
             isPrefixedWith(label,
                            "Kokkos::Serial::") || // unsure what's going on
             isPrefixedWith(label, "Testing::")));
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
      [](Kokkos::Profiling::SpaceHandle /*handle*/, const char *label,
         void const * /*ptr*/, uint64_t /*size*/) {
        BOOST_TEST_MESSAGE(label);
        BOOST_TEST(
            (isPrefixedWith(label, "ArborX::BVH::query::") ||
             isPrefixedWith(label, "ArborX::TreeTraversal::spatial::") ||
             isPrefixedWith(label, "ArborX::TreeTraversal::nearest::") ||
             isPrefixedWith(label, "ArborX::CrsGraphWrapper::") ||
             isPrefixedWith(label, "ArborX::Sorting::") ||
             isPrefixedWith(label, "Kokkos::SortImpl::BinSortFunctor::") ||
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
  using Tree = ArborX::BVH<typename DeviceType::memory_space>;
  using ExecutionSpace = typename DeviceType::execution_space;

  auto const callback = [](char const *label, uint32_t, uint64_t *) {
    BOOST_TEST_MESSAGE(label);
    BOOST_TEST((isPrefixedWith(label, "ArborX::") ||
                isPrefixedWith(label, "Kokkos::")));
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
    BOOST_TEST_MESSAGE(label);
    BOOST_TEST((isPrefixedWith(label, "ArborX::") ||
                isPrefixedWith(label, "Kokkos::")));
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

#endif
