/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_NODE_HPP
#define ARBORX_NODE_HPP

#include <ArborX_Box.hpp>

#include <Kokkos_Pair.hpp>

#include <cassert>

namespace ArborX
{
namespace Details
{
struct Node
{
  KOKKOS_DEFAULTED_FUNCTION
  constexpr Node() = default;

  KOKKOS_INLINE_FUNCTION constexpr bool isLeaf() const noexcept
  {
    return left_child == -1;
  }

  KOKKOS_INLINE_FUNCTION constexpr std::size_t getLeafPermutationIndex() const
      noexcept
  {
    assert(isLeaf());
    return right_child;
  }

  int left_child = -1;
  int right_child = -1;
  Box bounding_box;
};

KOKKOS_INLINE_FUNCTION constexpr Node
makeLeafNode(std::size_t permutation_index, Box box) noexcept
{
  return {-1, static_cast<int>(permutation_index), std::move(box)};
}
} // namespace Details
} // namespace ArborX

#endif
