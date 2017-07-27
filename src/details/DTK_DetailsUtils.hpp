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
 *  When \c in and \c out are the same view, the scan is performed in-place.
 *
 *  \pre \c in and \c out must have the same size.
 */
template <typename T, typename DeviceType>
void exclusive_prefix_sum( Kokkos::View<T *, DeviceType> in,
                           Kokkos::View<T *, DeviceType> out )
{
    using ExecutionSpace = typename DeviceType::execution_space;
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

} // end namespace DataTransferKit

#endif
