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

#ifndef ARBORX_IOTA_HPP
#define ARBORX_IOTA_HPP

#include "ArborX_AccessTraits.hpp"

#include <Kokkos_Core.hpp>

namespace ArborX
{
namespace Details
{

template <typename MemorySpace>
struct Iota
{
  static_assert(Kokkos::is_memory_space_v<MemorySpace>);
  using memory_space = MemorySpace;
  int _n;
};

} // namespace Details

template <typename MemorySpace>
struct AccessTraits<Details::Iota<MemorySpace>>
{
  using Self = Details::Iota<MemorySpace>;

  using memory_space = typename Self::memory_space;
  static KOKKOS_FUNCTION size_t size(Self const &self) { return self._n; }
  static KOKKOS_FUNCTION auto get(Self const &, int i) { return i; }
};

} // namespace ArborX

#endif
