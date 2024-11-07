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

#ifndef ARBORX_INDEXABLE_GETTER_HPP
#define ARBORX_INDEXABLE_GETTER_HPP

#include <ArborX_GeometryTraits.hpp>
#include <detail/ArborX_AccessTraits.hpp>
#include <detail/ArborX_PairValueIndex.hpp>

namespace ArborX
{

namespace Experimental
{

struct DefaultIndexableGetter
{
  KOKKOS_DEFAULTED_FUNCTION
  DefaultIndexableGetter() = default;

  template <typename Geometry, typename Enable = std::enable_if_t<
                                   GeometryTraits::is_valid_geometry<Geometry>>>
  KOKKOS_FUNCTION auto const &operator()(Geometry const &geometry) const
  {
    return geometry;
  }

  template <typename Geometry, typename Enable = std::enable_if_t<
                                   GeometryTraits::is_valid_geometry<Geometry>>>
  KOKKOS_FUNCTION auto operator()(Geometry &&geometry) const
  {
    return geometry;
  }

  template <typename Value, typename Index>
  KOKKOS_FUNCTION Value const &
  operator()(PairValueIndex<Value, Index> const &pair) const
  {
    return pair.value;
  }

  template <typename Value, typename Index>
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

#ifdef KOKKOS_ENABLE_CXX17
template <typename Values, typename IndexableGetter>
KOKKOS_DEDUCTION_GUIDE Indexables(Values, IndexableGetter)
    -> Indexables<Values, IndexableGetter>;
#endif

} // namespace Details

} // namespace ArborX

#endif
