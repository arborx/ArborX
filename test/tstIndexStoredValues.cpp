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

#include <ArborX_BruteForce.hpp>
#include <ArborX_LinearBVH.hpp>

#include <boost/test/unit_test.hpp>

// clang-format off
#include "ArborXTest_TreeTypeTraits.hpp"
#include "ArborXTest_StdVectorToKokkosView.hpp"
#include "ArborX_EnableViewComparison.hpp"
// clang-format on

#include <algorithm>
#include <numeric>
#include <random>

BOOST_AUTO_TEST_SUITE(AccessValuesStoredIntoIndex)

namespace tt = boost::test_tools;

struct MyIndexableGetter
{
  KOKKOS_FUNCTION ArborX::Point operator()(int i) const
  {
    return {(float)i, (float)i, (float)i};
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(empty_tree_spatial_predicate, TreeTypeTraits,
                              TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using MemorySpace = typename Tree::memory_space;

  ExecutionSpace exec;
  using Index =
      // ArborX::BoundingVolumeHierarchy<MemorySpace, int, MyIndexableGetter>;
      ArborX::BruteForce<MemorySpace, int, MyIndexableGetter>;
  Index index;
  BOOST_TEST(index.empty());

  int const n = 10;
  {
    // Let the values be some some random permutation of {0, 1, 2, ..., n-1}
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    std::shuffle(v.begin(), v.end(), std::default_random_engine());
    auto values = ArborXTest::toView<MemorySpace>(v, "ArborXTest::values");

    index = Index(exec, values);
    BOOST_TEST((int)index.size() == n);
  }

  Kokkos::View<int *, MemorySpace> stored_values("ArborXTest::stored", n);
  Kokkos::parallel_for(
      "ArborXTest::retrieve_stored_values",
      Kokkos::RangePolicy<ExecutionSpace>(exec, 0, n),
      KOKKOS_LAMBDA(int i) { stored_values[i] = index[i]; });

  auto h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), stored_values);
  // std::sort(h.data(), h.data() + n);
  std::vector<int> v(n);
  std::iota(v.begin(), v.end(), 0);
  BOOST_TEST(h == v, tt::per_element());
}

BOOST_AUTO_TEST_SUITE_END()
