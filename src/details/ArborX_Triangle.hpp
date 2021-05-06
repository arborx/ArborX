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

#ifndef ARBORX_TRIANGLE_HPP
#define ARBORX_TRIANGLE_HPP

#include <ArborX_Box.hpp>
#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_Point.hpp>

namespace ArborX
{
namespace Experimental
{
struct Triangle
{
  Point a;
  Point b;
  Point c;
};

KOKKOS_INLINE_FUNCTION void expand(Box &box, Triangle const &triangle)
{
  using Details::expand;
  expand(box, triangle.a);
  expand(box, triangle.b);
  expand(box, triangle.c);
}

} // namespace Experimental
} // namespace ArborX

#endif
