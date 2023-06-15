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

#ifndef ARBORX_DETAILS_KOKKOS_EXT_BIT_MANIPULATION_HPP
#define ARBORX_DETAILS_KOKKOS_EXT_BIT_MANIPULATION_HPP

#include <Kokkos_Macros.hpp>

#include <type_traits>

namespace KokkosExt
{

#if KOKKOS_VERSION >= 40100
using Kokkos::bit_cast;
#else
template <class To, class From>
KOKKOS_FUNCTION std::enable_if_t<sizeof(To) == sizeof(From) &&
                                     std::is_trivially_copyable_v<To> &&
                                     std::is_trivially_copyable_v<From>,
                                 To>
bit_cast(From const &from) noexcept
{
  To to;
  memcpy(&to, &from, sizeof(To));
  return to;
}
#endif

} // namespace KokkosExt

#endif
