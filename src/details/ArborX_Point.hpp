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

#ifndef ARBORX_DETAILS_POINT_HPP
#define ARBORX_DETAILS_POINT_HPP

#include <Kokkos_Macros.hpp>

namespace ArborX
{
class Point
{
  public:
    KOKKOS_INLINE_FUNCTION
    double &operator[]( unsigned int i ) { return _coords[i]; }

    KOKKOS_INLINE_FUNCTION
    double const &operator[]( unsigned int i ) const { return _coords[i]; }

    KOKKOS_INLINE_FUNCTION
    double volatile &operator[]( unsigned int i ) volatile
    {
        return _coords[i];
    }

    KOKKOS_INLINE_FUNCTION
    double const volatile &operator[]( unsigned int i ) const volatile
    {
        return _coords[i];
    }

    // This should be private but if we make public we can use the list
    // initializer constructor.
    double _coords[3];
};
} // namespace ArborX

#endif
