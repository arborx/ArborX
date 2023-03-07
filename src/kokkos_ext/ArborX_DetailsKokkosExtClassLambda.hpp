/****************************************************************************
 * Copyright (c) 2017-2023 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DETAILS_KOKKOS_EXT_CLASS_LAMBDA
#define ARBORX_DETAILS_KOKKOS_EXT_CLASS_LAMBDA

#include <Kokkos_Macros.hpp>

// Remove when we require Kokkos 4.0.00
#ifdef KOKKOS_CLASS_LAMBDA
#define ARBORX_CLASS_LAMBDA KOKKOS_CLASS_LAMBDA
#elif defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#define ARBORX_CLASS_LAMBDA [ =, *this ] __host__ __device__
#else
#define ARBORX_CLASS_LAMBDA [ =, *this ]
#endif

#endif
