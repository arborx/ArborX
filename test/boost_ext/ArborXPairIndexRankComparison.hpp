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

#ifndef ARBORX_BOOST_TEST_ARBORX_PAIR_INDEX_RANK_COMPARISON_HPP
#define ARBORX_BOOST_TEST_ARBORX_PAIR_INDEX_RANK_COMPARISON_HPP

#include <ArborX_PairIndexRank.hpp>

#include <boost/test/tools/detail/print_helper.hpp>

#include <iostream>

namespace boost::test_tools::tt_detail
{

template <>
struct print_log_value<ArborX::PairIndexRank>
{
  void operator()(std::ostream &os, ArborX::PairIndexRank const &p)
  {
    os << '(' << p.index << ',' << p.rank << ')';
  }
};

} // namespace boost::test_tools::tt_detail

#endif
