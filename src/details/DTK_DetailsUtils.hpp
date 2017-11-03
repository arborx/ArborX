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

template <typename T, typename DeviceType>
class ExclusiveScanFunctor
{
  public:
    ExclusiveScanFunctor( Kokkos::View<T *, DeviceType> const &in,
                          Kokkos::View<T *, DeviceType> const &out )
        : _in( in )
        , _out( out )
    {
    }
    KOKKOS_INLINE_FUNCTION void operator()( int i, T &update,
                                            bool final_pass ) const
    {
        T const in_i = _in( i );
        if ( final_pass )
            _out( i ) = update;
        update += in_i;
    }

  private:
    Kokkos::View<T *, DeviceType> _in;
    Kokkos::View<T *, DeviceType> _out;
};

/** \brief Computes an exclusive scan.
 *
 *  When \c dst is not provided or if \c src and \c dst are the same view, the
 *  scan is performed in-place.
 *
 *  \pre \c src and \c dst must be of rank 1 and have the same size.
 */
template <typename ST, typename... SP, typename DT, typename... DP>
void exclusivePrefixSum( Kokkos::View<ST, SP...> const &src,
                         Kokkos::View<DT, DP...> const &dst )
{
    static_assert(
        std::is_same<typename Kokkos::ViewTraits<DT, DP...>::value_type,
                     typename Kokkos::ViewTraits<
                         DT, DP...>::non_const_value_type>::value,
        "exclusivePrefixSum requires non-const destination type" );

    static_assert( ( unsigned( Kokkos::ViewTraits<DT, DP...>::rank ) ==
                     unsigned( Kokkos::ViewTraits<ST, SP...>::rank ) ) &&
                       ( unsigned( Kokkos::ViewTraits<DT, DP...>::rank ) ==
                         unsigned( 1 ) ),
                   "exclusivePrefixSum requires Views of rank 1" );

    using ExecutionSpace =
        typename Kokkos::ViewTraits<DT, DP...>::execution_space;
    using ValueType = typename Kokkos::ViewTraits<DT, DP...>::value_type;

    auto const n = src.span();
    DTK_REQUIRE( n == dst.span() );
    Kokkos::parallel_scan(
        "exclusive_scan", Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
        ExclusiveScanFunctor<ValueType, ExecutionSpace>( src, dst ) );
    Kokkos::fence();
}

template <typename T, typename... P>
void exclusivePrefixSum( Kokkos::View<T, P...> const &v )
{
    exclusivePrefixSum( v, v );
}

/** \brief Get a copy of the last element.
 *
 *  Returns a copy of the last element in the view on the host.  Note that it
 *  may require communication between host and device (e.g. if the view passed
 *  as an argument lives on the device).
 *
 *  \pre \c v is of rank 1 and not empty.
 */
template <typename T, typename... P>
typename Kokkos::ViewTraits<T, P...>::value_type
lastElement( Kokkos::View<T, P...> const &v )
{
    static_assert(
        ( unsigned( Kokkos::ViewTraits<T, P...>::rank ) == unsigned( 1 ) ),
        "lastElement requires Views of rank 1" );
    auto const n = v.span();
    DTK_REQUIRE( n > 0 );
    auto v_subview = Kokkos::subview( v, n - 1 );
    auto v_host = Kokkos::create_mirror_view( v_subview );
    Kokkos::deep_copy( v_host, v_subview );
    return v_host( 0 );
}

} // end namespace DataTransferKit

#endif
