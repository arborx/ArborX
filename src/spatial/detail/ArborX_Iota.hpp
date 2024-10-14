/****************************************************************************
 * Copyright (c) 2024 by the ArborX authors                                 *
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

#include <detail/ArborX_AccessTraits.hpp>

#include <Kokkos_Macros.hpp>

#include <type_traits>

namespace ArborX
{

template <typename MemorySpace, typename Index = int>
struct Iota
{
  using memory_space = MemorySpace;
  using index_type = Index;

  size_t _n;

  template <typename T,
            typename Enable = std::enable_if_t<std::is_integral_v<T>>>
  Iota(T n)
      : _n(n)
  {}
};

template <typename MemorySpace>
struct ArborX::AccessTraits<Iota<MemorySpace>, ArborX::PrimitivesTag>
{
  using Self = Iota<MemorySpace>;

  using memory_space = typename Self::memory_space;
  static KOKKOS_FUNCTION size_t size(Self const &self) { return self._n; }
  static KOKKOS_FUNCTION auto get(Self const &, size_t i)
  {
    return (typename Self::index_type)i;
  }
};

} // namespace ArborX

#endif
