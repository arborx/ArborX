/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DETAILS_TRAITS_HPP
#define ARBORX_DETAILS_TRAITS_HPP

#include <ArborX_Box.hpp>
#include <ArborX_DetailsTags.hpp>
#include <ArborX_Point.hpp>

#include <Kokkos_View.hpp>

namespace ArborX
{
namespace Traits
{

struct PrimitivesTag
{
};

struct PredicatesTag
{
};

template <typename T, typename Tag, typename Enable = void>
struct Access
{
};

template <typename View, typename TTag>
struct Access<View, TTag,
              typename std::enable_if<Kokkos::is_view<View>::value &&
                                      View::rank == 1>::type>
{
  // Returns a const reference
  KOKKOS_FUNCTION static typename View::const_value_type &get(View const &v,
                                                              int i)
  {
    return v(i);
  }

  static typename View::size_type size(View const &v) { return v.extent(0); }

  using Tag = typename Details::Tag<typename View::value_type>::type;
  using MemorySpace = typename View::memory_space;
};

template <typename View, typename TTag>
struct Access<View, TTag,
              typename std::enable_if<Kokkos::is_view<View>::value &&
                                      View::rank == 2>::type>
{
  // Returns by value
  KOKKOS_FUNCTION static Point get(View const &v, int i)
  {
    return {v(i, 0), v(i, 1), v(i, 2)};
  }

  static typename View::size_type size(View const &v) { return v.extent(0); }

  using Tag = Details::PointTag;
  using MemorySpace = typename View::memory_space;
};

} // namespace Traits
} // namespace ArborX

#endif
