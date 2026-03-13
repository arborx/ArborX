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
#ifndef ARBORX_DISTRIBUTED_UTILS_HPP
#define ARBORX_DISTRIBUTED_UTILS_HPP

#include <algorithm> // std::sort, std::reverse
#include <array>
#include <cassert>
#include <vector>

namespace ArborX::Details
{

// Find closest DIM factors for a number. The factors are
// sorted in the descending order.
template <int DIM>
std::array<int, DIM> closestFactors(int const n)
{
  static_assert(DIM > 0);
  assert(n > 0);

  std::array<int, DIM> result;
  result.fill(1);
  if constexpr (DIM == 1)
  {
    result[0] = n;
    return result;
  }

  // Find all prime factors in increasing order
  std::vector<int> factors;
  unsigned i = 2;
  auto nn = n;
  while (nn > 1)
  {
    if (nn % i != 0)
    {
      ++i;
      continue;
    }

    factors.push_back(i);
    nn /= i;
  }
  std::reverse(factors.begin(), factors.end());

  // Approach from https://stackoverflow.com/a/5903453
  // Loop over factors in decreasing order, multiply them
  // by the currently smallest.
  for (auto &f : factors)
  {
    result[0] *= f;
    std::sort(result.begin(), result.end());
  }

  std::reverse(result.begin(), result.end());
  return result;
}

} // namespace ArborX::Details

#endif
