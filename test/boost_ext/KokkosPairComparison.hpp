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

#ifndef ARBORX_BOOST_TEST_KOKKOS_PAIR_COMPARISON_HPP
#define ARBORX_BOOST_TEST_KOKKOS_PAIR_COMPARISON_HPP

#include <Kokkos_Pair.hpp>

#include <boost/test/tools/detail/print_helper.hpp>

#include <iostream>

// Enable comparison of Kokkos pairs
namespace boost
{
namespace test_tools
{
namespace tt_detail
{

// FIXME needed for TuplePrinter
template <typename T1, typename T2>
std::ostream &operator<<(std::ostream &os, Kokkos::pair<T1, T2> const &p)
{
  os << '(' << p.first << ',' << p.second << ')';
  return os;
}

template <typename T1, typename T2>
struct print_log_value<Kokkos::pair<T1, T2>>
{
  void operator()(std::ostream &os, Kokkos::pair<T1, T2> const &p) { os << p; }
};
} // namespace tt_detail
} // namespace test_tools
} // namespace boost

#endif
