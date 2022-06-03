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

#include <ArborX_MinimumSpanningTree.hpp>

namespace
{

// NOTE: message is not required anymore with C++17
#define STATIC_ASSERT(cond) static_assert(cond, "")

KOKKOS_FUNCTION constexpr bool test_directed_edges()
{
  using ArborX::Details::DirectedEdge;

  STATIC_ASSERT((DirectedEdge{0, 1, 2.f}.weight == 2.f));
  STATIC_ASSERT((DirectedEdge{0, 1, 2.f}.source() == 0));
  STATIC_ASSERT((DirectedEdge{0, 1, 2.f}.target() == 1));

  STATIC_ASSERT((DirectedEdge{6, 5, 4.f}.weight == 4.f));
  STATIC_ASSERT((DirectedEdge{6, 5, 4.f}.source() == 6));
  STATIC_ASSERT((DirectedEdge{6, 5, 4.f}.target() == 5));

  constexpr auto inf = KokkosExt::ArithmeticTraits::infinity<float>::value;
  STATIC_ASSERT((DirectedEdge{}.weight == inf));
  STATIC_ASSERT((DirectedEdge{}.source() == INT_MAX));
  STATIC_ASSERT((DirectedEdge{}.target() == INT_MAX));

  STATIC_ASSERT((DirectedEdge{0, 1, 2.f} < DirectedEdge{0, 1, 3.f}));
  STATIC_ASSERT((DirectedEdge{0, 1, 2.f} < DirectedEdge{1, 2, 2.f}));
  STATIC_ASSERT((DirectedEdge{0, 1, 2.f} < DirectedEdge{2, 1, 2.f}));

  constexpr DirectedEdge e12{1, 2, 3};
  constexpr DirectedEdge e21{2, 1, 3};
  STATIC_ASSERT(e12.weight == e21.weight);
  STATIC_ASSERT((e12 < e21) ^ (e21 < e12));

  return true;
}
STATIC_ASSERT(test_directed_edges()); // avoid warning unused function

KOKKOS_FUNCTION constexpr bool test_weighted_edges()
{
  using ArborX::Details::WeightedEdge;

  STATIC_ASSERT((WeightedEdge{1, 2, 3} < WeightedEdge{1, 2, 4}));

  constexpr WeightedEdge e12{1, 2, 3};
  constexpr WeightedEdge e21{2, 1, 3};
  STATIC_ASSERT(e12.weight == e21.weight);
  STATIC_ASSERT(!(e12 < e21));
  STATIC_ASSERT(!(e21 < e12));

  constexpr WeightedEdge e14{1, 4, 3};
  STATIC_ASSERT(e12.weight == e14.weight);
  STATIC_ASSERT(e12 < e14);

  return true;
}
STATIC_ASSERT(test_weighted_edges()); // avoid warning unused function

} // namespace
