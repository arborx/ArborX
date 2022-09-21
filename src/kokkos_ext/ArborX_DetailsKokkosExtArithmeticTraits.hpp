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

#ifndef ARBORX_DETAILS_KOKKOS_EXT_ARITHMETIC_TRAITS_HPP
#define ARBORX_DETAILS_KOKKOS_EXT_ARITHMETIC_TRAITS_HPP

#include <Kokkos_NumericTraits.hpp>

namespace KokkosExt::ArithmeticTraits
{

template <class T>
using infinity = Kokkos::Experimental::infinity<T>;

template <class T>
using finite_max = Kokkos::Experimental::finite_max<T>;

template <class T>
using finite_min = Kokkos::Experimental::finite_min<T>;

template <class T>
using epsilon = Kokkos::Experimental::epsilon<T>;

} // namespace KokkosExt::ArithmeticTraits

#endif
