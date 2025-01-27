/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_TEST_PAIR_INDEX_RANK_HPP
#define ARBORX_TEST_PAIR_INDEX_RANK_HPP

#include <Kokkos_Macros.hpp>

#include <boost/test/tools/detail/print_helper.hpp>

#include <iostream>

namespace ArborXTest
{

struct PairIndexRank
{
  int index;
  int rank;

private:
  friend KOKKOS_FUNCTION constexpr bool operator==(PairIndexRank lhs,
                                                   PairIndexRank rhs)
  {
    return lhs.index == rhs.index && lhs.rank == rhs.rank;
  }
  friend KOKKOS_FUNCTION constexpr bool operator<(PairIndexRank lhs,
                                                  PairIndexRank rhs)
  {
    return lhs.rank < rhs.rank ||
           (lhs.rank == rhs.rank && lhs.index < rhs.index);
  }
};

} // namespace ArborXTest

namespace boost::test_tools::tt_detail
{

template <>
struct print_log_value<ArborXTest::PairIndexRank>
{
  void operator()(std::ostream &os, ArborXTest::PairIndexRank const &p)
  {
    os << '(' << p.index << ',' << p.rank << ')';
  }
};

} // namespace boost::test_tools::tt_detail

#endif
