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

#ifndef ARBORX_KOKKOS_EXT_SORT_HPP
#define ARBORX_KOKKOS_EXT_SORT_HPP

#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <Kokkos_Sort.hpp>

namespace ArborX::Details::KokkosExt
{

template <typename ExecutionSpace, typename Keys, typename Values>
void sortByKey(ExecutionSpace const &space, Keys &keys, Values &values)
{
  Kokkos::Profiling::ScopedRegion guard("ArborX::KokkosExt::sortByKey::Kokkos");
  Kokkos::Experimental::sort_by_key(space, keys, values);
}

} // namespace ArborX::Details::KokkosExt

#endif
