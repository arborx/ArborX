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

#ifndef ARBORX_PAIR_INDEX_RANK_HPP
#define ARBORX_PAIR_INDEX_RANK_HPP

#include <Kokkos_Macros.hpp>

namespace ArborX
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

} // namespace ArborX

#endif
