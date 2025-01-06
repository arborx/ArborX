/****************************************************************************
 * Copyright (c) 2017-2023 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include <ArborXTest_LegacyTree.hpp>
#include <ArborX_LinearBVH.hpp>

#include <boost/test/unit_test.hpp>

#include <limits>
#include <string>

#include "Search_UnitTestHelpers.hpp"

BOOST_AUTO_TEST_SUITE(KokkosToolsExecutionSpaceInstances)

namespace tt = boost::test_tools;

namespace
{
// Lambdas can only be converted to function pointers if they do not capture.
// Using a global non-static variable in an unnamed namespace to "capture" the
// device id.
uint32_t INVALID_DEVICE_ID = std::numeric_limits<uint32_t>::max();
uint32_t arborx_test_device_id = INVALID_DEVICE_ID;
uint32_t arborx_test_root_device_id = INVALID_DEVICE_ID;

void arborx_test_parallel_x_callback(char const *label, uint32_t device_id,
                                     uint64_t * /*kernel_id*/)
{
  std::string label_str(label);

  for (std::string s : {"Kokkos::View::destruction []"})
    if (label_str.find(s) != std::string::npos)
      return;

  BOOST_TEST(device_id == arborx_test_device_id,
             "\"" << label
                  << "\" kernel not on the right execution space instance: "
                  << device_id << " != " << arborx_test_device_id);
}

template <class ExecutionSpace>
void arborx_test_set_tools_callbacks(ExecutionSpace exec)
{
  arborx_test_device_id = Kokkos::Tools::Experimental::device_id(exec);
  arborx_test_root_device_id =
      Kokkos::Tools::Experimental::device_id_root<ExecutionSpace>();

  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(
      arborx_test_parallel_x_callback);
  Kokkos::Tools::Experimental::set_begin_parallel_reduce_callback(
      arborx_test_parallel_x_callback);
  Kokkos::Tools::Experimental::set_begin_parallel_scan_callback(
      arborx_test_parallel_x_callback);
}

void arborx_test_unset_tools_callbacks()
{
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(nullptr);
  Kokkos::Tools::Experimental::set_begin_parallel_reduce_callback(nullptr);
  Kokkos::Tools::Experimental::set_begin_parallel_scan_callback(nullptr);
  arborx_test_device_id = INVALID_DEVICE_ID;
  arborx_test_root_device_id = INVALID_DEVICE_ID;
}

} // namespace

BOOST_AUTO_TEST_CASE_TEMPLATE(bvh_bvh_execution_space_instance, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  using Box = ArborX::Box<3>;
  using Tree =
      LegacyTree<ArborX::BoundingVolumeHierarchy<MemorySpace,
                                                 ArborX::PairValueIndex<Box>>>;

  auto exec = Kokkos::Experimental::partition_space(ExecutionSpace{}, 1)[0];
  arborx_test_set_tools_callbacks(exec);

  { // default constructed
    Tree tree;
  }

  { // empty
    auto tree = make<Tree, Box>(exec, {});
  }

  { // one leaf
    auto tree = make<Tree, Box>(exec, {
                                          {{{0, 0, 0}}, {{1, 1, 1}}},
                                      });
  }

  { // two leaves
    auto tree = make<Tree, Box>(exec, {
                                          {{{0, 0, 0}}, {{1, 1, 1}}},
                                          {{{0, 0, 0}}, {{1, 1, 1}}},
                                      });
  }

  arborx_test_unset_tools_callbacks();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(bvh_query_execution_space_instance, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  using Box = ArborX::Box<3>;
  using Tree =
      LegacyTree<ArborX::BoundingVolumeHierarchy<MemorySpace,
                                                 ArborX::PairValueIndex<Box>>>;

  auto tree = make<Tree, Box>(ExecutionSpace{}, {
                                                    {{{0, 0, 0}}, {{1, 1, 1}}},
                                                    {{{0, 0, 0}}, {{1, 1, 1}}},
                                                });

  auto exec = Kokkos::Experimental::partition_space(ExecutionSpace{}, 1)[0];
  arborx_test_set_tools_callbacks(exec);

  // spatial predicates
  query(exec, tree,
        makeIntersectsBoxQueries<DeviceType>({
            {{{0, 0, 0}}, {{1, 1, 1}}},
            {{{0, 0, 0}}, {{1, 1, 1}}},
        }));

  // nearest predicates
  query(exec, tree,
        makeNearestQueries<DeviceType>({
            {{{0, 0, 0}}, 1},
            {{{0, 0, 0}}, 2},
        }));

  arborx_test_unset_tools_callbacks();
}

BOOST_AUTO_TEST_SUITE_END()
