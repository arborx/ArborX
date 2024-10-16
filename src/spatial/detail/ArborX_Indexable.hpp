/****************************************************************************
 * Copyright (c) 2023 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_INDEXABLE_HPP
#define ARBORX_INDEXABLE_HPP

#include <ArborX_GeometryTraits.hpp>
#include <detail/ArborX_AccessTraits.hpp>
#include <detail/ArborX_PairValueIndex.hpp>

namespace ArborX
{

namespace Experimental
{

template <typename Value>
struct Indexable
{
  KOKKOS_DEFAULTED_FUNCTION
  Indexable() = default;

  KOKKOS_FUNCTION Value const &operator()(Value const &value) const
  {
    return value;
  }

  KOKKOS_FUNCTION Value operator()(Value &&value) const { return value; }
};

template <typename Value, typename Index>
struct Indexable<PairValueIndex<Value, Index>>
{
  KOKKOS_DEFAULTED_FUNCTION
  Indexable() = default;

  KOKKOS_FUNCTION Value const &
  operator()(PairValueIndex<Value, Index> const &pair) const
  {
    return pair.value;
  }

  KOKKOS_FUNCTION Value operator()(PairValueIndex<Value, Index> &&pair) const
  {
    return pair.value;
  }
};

} // namespace Experimental

namespace Details
{

template <typename Values, typename IndexableGetter>
struct Indexables
{
  Values _values;
  IndexableGetter _indexable_getter;

  using memory_space = typename Values::memory_space;

  KOKKOS_FUNCTION decltype(auto) operator()(int i) const
  {
    return _indexable_getter(_values(i));
  }

  KOKKOS_FUNCTION auto size() const { return _values.size(); }
};

template <typename Values, typename IndexableGetter>
#if KOKKOS_VERSION >= 40400
KOKKOS_DEDUCTION_GUIDE
#else
KOKKOS_FUNCTION
#endif
    Indexables(Values, IndexableGetter) -> Indexables<Values, IndexableGetter>;

} // namespace Details

} // namespace ArborX

#endif
