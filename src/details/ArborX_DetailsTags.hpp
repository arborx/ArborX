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

#ifndef ARBORX_DETAILS_TAGS_HPP
#define ARBORX_DETAILS_TAGS_HPP

#include <ArborX_Box.hpp>
#include <ArborX_Point.hpp>

namespace ArborX
{
namespace Details
{

struct PointTag
{
};

struct BoxTag
{
};

template <typename T, typename Enable = void>
struct Tag
{
};

template <>
struct Tag<Point>
{
  using type = PointTag;
};

template <>
struct Tag<Box>
{
  using type = BoxTag;
};

template <typename T>
struct Tag<T>
{
  using type = typename T::Tag;
};

} // namespace Details
} // namespace ArborX

#endif
