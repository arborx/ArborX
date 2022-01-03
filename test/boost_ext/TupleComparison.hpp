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

#ifndef ARBORX_BOOST_TEST_TUPLE_COMPARISON_HPP
#define ARBORX_BOOST_TEST_TUPLE_COMPARISON_HPP

#include <boost/test/tools/detail/print_helper.hpp>

#include <iostream>
#include <tuple>

// Enable comparison of tuples
namespace boost
{
namespace test_tools
{
namespace tt_detail
{
namespace cppreference
{
// helper function to print a tuple of any size
// adapted from https://en.cppreference.com/w/cpp/utility/tuple/tuple_cat
template <class Tuple, std::size_t N>
struct TuplePrinter
{
  static void print(std::ostream &os, Tuple const &t)
  {
    TuplePrinter<Tuple, N - 1>::print(os, t);
    os << ", " << std::get<N - 1>(t);
  }
};

template <class Tuple>
struct TuplePrinter<Tuple, 1>
{
  static void print(std::ostream &os, Tuple const &t) { os << std::get<0>(t); }
};
} // namespace cppreference

template <typename... Args>
struct print_log_value<std::tuple<Args...>>
{
  void operator()(std::ostream &os, std::tuple<Args...> const &t)
  {
    os << '(';
    cppreference::TuplePrinter<decltype(t), sizeof...(Args)>::print(os, t);
    os << ')';
  }
};
} // namespace tt_detail
} // namespace test_tools
} // namespace boost

#endif
