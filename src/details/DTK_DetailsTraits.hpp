/****************************************************************************
 * Copyright (c) 2012-2018 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef DTK_DETAILS_TRAITS_HPP
#define DTK_DETAILS_TRAITS_HPP

#include <DTK_Box.hpp>
#include <DTK_Point.hpp>

#include <Kokkos_View.hpp>

namespace DataTransferKit
{
namespace Details
{
namespace Traits
{

template <typename T, typename Enable = void>
struct Access
{
};

template <typename View>
struct Access<View, typename std::enable_if<Kokkos::is_view<View>::value &&
                                            View::rank == 1>::type>
{
    // Returns a const reference
    KOKKOS_FUNCTION static typename View::const_value_type &get( View const &v,
                                                                 int i )
    {
        return v( i );
    }

    static typename View::size_type size( View const &v )
    {
        return v.extent( 0 );
    }

    using Tag = typename Tag<typename View::value_type>::type;
};

template <typename View>
struct Access<View, typename std::enable_if<Kokkos::is_view<View>::value &&
                                            View::rank == 2>::type>
{
    // Returns by value
    KOKKOS_FUNCTION static Point get( View const &v, int i )
    {
        return {v( i, 0 ), v( i, 1 ), v( i, 2 )};
    }

    static typename View::size_type size( View const &v )
    {
        return v.extent( 0 );
    }

    using Tag = PointTag;
};

} // namespace Traits
} // namespace Details
} // namespace DataTransferKit

#endif
