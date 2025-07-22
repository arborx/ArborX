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

#ifndef ARBORX_PAIR_VALUE_INDEX_HPP
#define ARBORX_PAIR_VALUE_INDEX_HPP

#include <Kokkos_Macros.hpp>

#include <type_traits>

namespace ArborX
{

template <typename Value, typename Index = unsigned>
struct PairValueIndex
{
  static_assert(std::is_integral_v<Index>);

  using value_type = Value;
  using index_type = Index;

  Value value;
  Index index;
};

namespace Experimental
{
// Callback to store the result indices
struct ExtractIndex
{
  template <typename Query, typename Value, typename Index, typename Output>
  KOKKOS_FUNCTION void operator()(Query const &,
                                  PairValueIndex<Value, Index> const &pair,
                                  Output const &out) const
  {
    out(pair.index);
  }
};

} // namespace Experimental

namespace Details
{
template <typename T>
struct is_pair_value_index : public std::false_type
{};

template <typename Value, typename Index>
struct is_pair_value_index<PairValueIndex<Value, Index>> : public std::true_type
{};

template <typename T>
inline constexpr bool is_pair_value_index_v = is_pair_value_index<T>::value;

} // namespace Details

} // namespace ArborX

#endif
