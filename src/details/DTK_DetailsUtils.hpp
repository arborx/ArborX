/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#ifndef DTK_DETAILS_UTILS_HPP
#define DTK_DETAILS_UTILS_HPP

#include <DTK_DBC.hpp>

#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

namespace DataTransferKit
{

/** \brief Assigns the given value to all elements in the view.
 */
template <typename T, typename DeviceType>
void fill( Kokkos::View<T *, DeviceType> out, T const &value )
{
    using ExecutionSpace = typename DeviceType::execution_space;
    Kokkos::parallel_for(
        "fill", Kokkos::RangePolicy<ExecutionSpace>( 0, out.extent( 0 ) ),
        KOKKOS_LAMBDA( int i ) { out( i ) = value; } );
    Kokkos::fence();
}

/** \brief Computes an exclusive scan.
 *
 *  When \c out is not provided or if \c in and \c out are the same view, the
 *  scan is performed in-place.
 *
 *  \pre \c in and \c out must have the same size.
 */
template <typename T, typename DeviceType>
void exclusivePrefixSum(
    Kokkos::View<T *, DeviceType> in,
    Kokkos::View<T *, DeviceType> out = Kokkos::View<T *, DeviceType>() )
{
    using ExecutionSpace = typename DeviceType::execution_space;
    if ( out.size() == 0 )
        out = in;
    DTK_INSIST( in.extent( 0 ) == out.extent( 0 ) );
    Kokkos::parallel_scan(
        "exclusive_scan",
        Kokkos::RangePolicy<ExecutionSpace>( 0, in.extent( 0 ) ),
        KOKKOS_LAMBDA( int i, int &update, bool final_pass ) {
            int const in_i = in( i );
            if ( final_pass )
                out( i ) = update;
            update += in_i;
        } );
    Kokkos::fence();
}

/** \brief Get a copy of the last element on the host.
 *
 *  Returns a copy of the last element in the view on the host.  Note that it
 *  may require communication between host and device (e.g. if the view passed
 *  as an argument lives on the device).
 *
 *  \pre \c in is not empty.
 */
template <typename T, typename DeviceType>
T lastElement( Kokkos::View<T *, DeviceType> in )
{
    DTK_INSIST( in.extent( 0 ) > 0 );
    auto in_subview = Kokkos::subview( in, in.extent( 0 ) - 1 );
    auto in_host = Kokkos::create_mirror_view( in_subview );
    Kokkos::deep_copy( in_host, in_subview );
    return in_host( 0 );
}

} // end namespace DataTransferKit

#endif
