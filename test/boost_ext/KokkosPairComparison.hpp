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

#ifndef ARBORX_BOOST_TEST_KOKKOS_PAIR_COMPARISON_HPP
#define ARBORX_BOOST_TEST_KOKKOS_PAIR_COMPARISON_HPP

#include <Kokkos_Pair.hpp>

#include <boost/test/tools/detail/print_helper.hpp>

#include <iostream>

// Enable comparison of Kokkos pairs
namespace boost::test_tools::tt_detail
{

template <typename T1, typename T2>
struct print_log_value<Kokkos::pair<T1, T2>>
{
  void operator()(std::ostream &os, Kokkos::pair<T1, T2> const &p)
  {
    os << '(';
    print_log_value<T1>()(os, p.first);
    os << ',';
    print_log_value<T2>()(os, p.second);
    os << ')';
  }
};

} // namespace boost::test_tools::tt_detail

#endif
