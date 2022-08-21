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

#include <ArborX_DetailsKokkosExtScopedProfileRegion.hpp>

#include <boost/test/unit_test.hpp>

#include <stack>
#include <string>
#include <type_traits>

namespace
{
std::stack<std::string> arborx_test_region_stack;

// NOTE: cannot use lambdas because they can only be converted to function
// pointers if they don't capture anything
void arborx_test_push_region(char const *label)
{
  BOOST_TEST_MESSAGE(std::string("push ") + label);
  arborx_test_region_stack.push(label);
}

void arborx_test_pop_region()
{
  auto const &label = arborx_test_region_stack.top();
  BOOST_TEST_MESSAGE(std::string("pop ") + label);
  arborx_test_region_stack.pop();
}

} // namespace

BOOST_AUTO_TEST_SUITE(KokkosToolsAnnotations)

BOOST_AUTO_TEST_CASE(scoped_profile_region)
{
  Kokkos::Tools::Experimental::set_push_region_callback(
      arborx_test_push_region);
  Kokkos::Tools::Experimental::set_pop_region_callback(arborx_test_pop_region);

  BOOST_TEST(arborx_test_region_stack.empty());

  // Unnamed guard!  Profile region is popped at the end of the statement.
  KokkosExt::ScopedProfileRegion("bug");

  BOOST_TEST(arborx_test_region_stack.empty());

  {
    std::string outer_identifier = "outer";
    KokkosExt::ScopedProfileRegion guard_outer(outer_identifier);

    BOOST_TEST(arborx_test_region_stack.size() == 1);
    BOOST_TEST(arborx_test_region_stack.top() == outer_identifier);

    {
      std::string inner_identifier = "inner";
      KokkosExt::ScopedProfileRegion guard_inner(inner_identifier);
      BOOST_TEST(arborx_test_region_stack.size() == 2);
      BOOST_TEST(arborx_test_region_stack.top() == inner_identifier);
    }

    BOOST_TEST(arborx_test_region_stack.size() == 1);
    BOOST_TEST(arborx_test_region_stack.top() == outer_identifier);
  }

  BOOST_TEST(arborx_test_region_stack.empty());

  // Unset callbacks
  Kokkos::Tools::Experimental::set_push_region_callback(nullptr);
  Kokkos::Tools::Experimental::set_pop_region_callback(nullptr);
}

static_assert(
    !std::is_default_constructible<KokkosExt::ScopedProfileRegion>::value);
static_assert(
    !std::is_copy_constructible<KokkosExt::ScopedProfileRegion>::value);
static_assert(
    !std::is_move_constructible<KokkosExt::ScopedProfileRegion>::value);
static_assert(!std::is_copy_assignable<KokkosExt::ScopedProfileRegion>::value);
static_assert(!std::is_move_assignable<KokkosExt::ScopedProfileRegion>::value);

BOOST_AUTO_TEST_SUITE_END()
