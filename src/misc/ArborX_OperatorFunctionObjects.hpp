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

#ifndef ARBORX_DETAILS_OPERATOR_FUNCTION_OBJECTS_HPP
#define ARBORX_DETAILS_OPERATOR_FUNCTION_OBJECTS_HPP

#include <Kokkos_Macros.hpp>

namespace ArborX
{
namespace Details
{

template <typename T>
struct Less
{
public:
  KOKKOS_FUNCTION constexpr bool operator()(T const &lhs, T const &rhs) const
  {
    return lhs < rhs;
  }
};

template <typename T>
struct Greater
{
public:
  KOKKOS_FUNCTION constexpr bool operator()(T const &lhs, T const &rhs) const
  {
    return lhs > rhs;
  }
};

} // namespace Details
} // namespace ArborX

#endif
