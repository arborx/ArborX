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

#ifndef ARBORX_KOKKOS_EXT_SORT_HPP
#define ARBORX_KOKKOS_EXT_SORT_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_Sort.hpp>

namespace ArborX::Details::KokkosExt
{
using Kokkos::Experimental::sort_by_key;
}

#endif
