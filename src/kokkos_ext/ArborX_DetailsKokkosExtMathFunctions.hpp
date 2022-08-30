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

#ifndef ARBORX_DETAILS_KOKKOS_EXT_MATH_FUNCTIONS_HPP
#define ARBORX_DETAILS_KOKKOS_EXT_MATH_FUNCTIONS_HPP

#include <Kokkos_MathematicalFunctions.hpp>

namespace KokkosExt
{

#if KOKKOS_VERSION >= 30700
using Kokkos::acos;
using Kokkos::cos;
using Kokkos::expm1;
using Kokkos::fabs;
using Kokkos::isfinite;
using Kokkos::sin;
#else
using Kokkos::Experimenatl::fabs;
using Kokkos::Experimental::acos;
using Kokkos::Experimental::cos;
using Kokkos::Experimental::expm1;
using Kokkos::Experimental::isfinite;
using Kokkos::Experimental::sin;
#endif

} // namespace KokkosExt

#endif
