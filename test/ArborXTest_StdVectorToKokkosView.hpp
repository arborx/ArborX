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

#ifndef ARBORX_TEST_STD_VECTOR_TO_KOKKOS_VIEW_HPP
#define ARBORX_TEST_STD_VECTOR_TO_KOKKOS_VIEW_HPP

#include <Kokkos_Core.hpp>

#include <string>
#include <vector>

namespace ArborXTest
{

template <typename DeviceType, typename T>
auto toView(std::vector<T> const &v, std::string const &lbl = "Test::Unnamed")
{
  Kokkos::View<T *, DeviceType> view(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, lbl), v.size());
  Kokkos::deep_copy(view, Kokkos::View<T const *, Kokkos::HostSpace,
                                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                              v.data(), v.size()));
  return view;
}

} // namespace ArborXTest

#endif
