/****************************************************************************
 * Copyright (c) 2012-2019 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <DTK_DetailsUtils.hpp>

#include <Teuchos_UnitTestHarness.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DetailsUtils, iota, DeviceType )
{
    int const n = 10;
    double const val = 3.;
    Kokkos::View<double *, DeviceType> v( "v", n );
    DataTransferKit::iota( v, val );
    std::vector<double> v_ref( n );
    std::iota( v_ref.begin(), v_ref.end(), val );
    auto v_host = Kokkos::create_mirror_view( v );
    Kokkos::deep_copy( v_host, v );
    TEST_COMPARE_ARRAYS( v_ref, v_host );

    Kokkos::View<int[3], DeviceType> w( "w" );
    DataTransferKit::iota( w );
    std::vector<int> w_ref = {0, 1, 2};
    auto w_host = Kokkos::create_mirror_view( w );
    Kokkos::deep_copy( w_host, w );
    TEST_COMPARE_ARRAYS( w_ref, w_host );
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DetailsUtils, prefix_sum, DeviceType )
{
    int const n = 10;
    Kokkos::View<int *, DeviceType> x( "x", n );
    std::vector<int> x_ref( n, 1 );
    x_ref.back() = 0;
    auto x_host = Kokkos::create_mirror_view( x );
    for ( int i = 0; i < n; ++i )
        x_host( i ) = x_ref[i];
    Kokkos::deep_copy( x, x_host );
    Kokkos::View<int *, DeviceType> y( "y", n );
    DataTransferKit::exclusivePrefixSum( x, y );
    std::vector<int> y_ref( n );
    std::iota( y_ref.begin(), y_ref.end(), 0 );
    auto y_host = Kokkos::create_mirror_view( y );
    Kokkos::deep_copy( y_host, y );
    Kokkos::deep_copy( x_host, x );
    TEST_COMPARE_ARRAYS( y_host, y_ref );
    TEST_COMPARE_ARRAYS( x_host, x_ref );
    // in-place
    DataTransferKit::exclusivePrefixSum( x, x );
    Kokkos::deep_copy( x_host, x );
    TEST_COMPARE_ARRAYS( x_host, y_ref );
    int const m = 11;
    TEST_INEQUALITY( n, m );
    Kokkos::View<int *, DeviceType> z( "z", m );
    TEST_THROW( DataTransferKit::exclusivePrefixSum( x, z ),
                DataTransferKit::SearchException );
    Kokkos::View<double[3], DeviceType> v( "v" );
    auto v_host = Kokkos::create_mirror_view( v );
    v_host( 0 ) = 1.;
    v_host( 1 ) = 1.;
    v_host( 2 ) = 0.;
    Kokkos::deep_copy( v, v_host );
    DataTransferKit::exclusivePrefixSum( v );
    Kokkos::deep_copy( v_host, v );
    std::vector<double> v_ref = {0., 1., 2.};
    TEST_COMPARE_FLOATING_ARRAYS( v_host, v_ref, 1e-14 );
    Kokkos::View<double *, DeviceType> w( "w", 4 );
    TEST_THROW( DataTransferKit::exclusivePrefixSum( v, w ),
                DataTransferKit::SearchException );
    v_host( 0 ) = 1.;
    v_host( 1 ) = 0.;
    v_host( 2 ) = 0.;
    Kokkos::deep_copy( v, v_host );
    Kokkos::resize( w, 3 );
    DataTransferKit::exclusivePrefixSum( v, w );
    auto w_host = Kokkos::create_mirror_view( w );
    Kokkos::deep_copy( w_host, w );
    std::vector<double> w_ref = {0., 1., 1.};
    TEST_COMPARE_FLOATING_ARRAYS( w_host, w_ref, 1e-14 );
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DetailsUtils, last_element, DeviceType )
{
    Kokkos::View<int *, DeviceType> v( "v", 2 );
    auto v_host = Kokkos::create_mirror_view( v );
    v_host( 0 ) = 33;
    v_host( 1 ) = 24;
    Kokkos::deep_copy( v, v_host );
    TEST_EQUALITY( DataTransferKit::lastElement( v ), 24 );
    Kokkos::View<int *, DeviceType> w( "w", 0 );
    TEST_THROW( DataTransferKit::lastElement( w ),
                DataTransferKit::SearchException );
    Kokkos::View<double[1], DeviceType> u( "u", 1 );
    Kokkos::deep_copy( u, 3.14 );
    TEST_FLOATING_EQUALITY( DataTransferKit::lastElement( u ), 3.14, 1e-14 );
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DetailsUtils, minmax, DeviceType )
{
    Kokkos::View<double[4], DeviceType> v( "v" );
    auto v_host = Kokkos::create_mirror_view( v );
    v_host( 0 ) = 3.14;
    v_host( 1 ) = 1.41;
    v_host( 2 ) = 2.71;
    v_host( 3 ) = 1.62;
    Kokkos::deep_copy( v, v_host );
    auto const result_float = DataTransferKit::minMax( v );
    TEST_FLOATING_EQUALITY( std::get<0>( result_float ), 1.41, 1e-14 );
    TEST_FLOATING_EQUALITY( std::get<1>( result_float ), 3.14, 1e-14 );
    Kokkos::View<int *, DeviceType> w( "w" );
    TEST_THROW( DataTransferKit::minMax( w ),
                DataTransferKit::SearchException );
    Kokkos::resize( w, 1 );
    Kokkos::deep_copy( w, 255 );
    auto const result_int = DataTransferKit::minMax( w );
    TEST_EQUALITY( std::get<0>( result_int ), 255 );
    TEST_EQUALITY( std::get<1>( result_int ), 255 );

    // testing use case in #336
    Kokkos::View<int[2][3], DeviceType> u( "u" );
    auto u_host = Kokkos::create_mirror_view( u );
    u_host( 0, 0 ) = 1; // x
    u_host( 0, 1 ) = 2; // y
    u_host( 0, 2 ) = 3; // z
    u_host( 1, 0 ) = 4; // x
    u_host( 1, 1 ) = 5; // y
    u_host( 1, 2 ) = 6; // Z
    Kokkos::deep_copy( u, u_host );
    auto const minmax_x =
        DataTransferKit::minMax( Kokkos::subview( u, Kokkos::ALL, 0 ) );
    TEST_EQUALITY( std::get<0>( minmax_x ), 1 );
    TEST_EQUALITY( std::get<1>( minmax_x ), 4 );
    auto const minmax_y =
        DataTransferKit::minMax( Kokkos::subview( u, Kokkos::ALL, 1 ) );
    TEST_EQUALITY( std::get<0>( minmax_y ), 2 );
    TEST_EQUALITY( std::get<1>( minmax_y ), 5 );
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DetailsUtils, accumulate, DeviceType )
{
    Kokkos::View<int[6], DeviceType> v( "v" );
    Kokkos::deep_copy( v, 5 );
    TEST_EQUALITY( DataTransferKit::accumulate( v, 3 ), 33 );

    Kokkos::View<int *, DeviceType> w( "w", 5 );
    DataTransferKit::iota( w, 2 );
    TEST_EQUALITY( DataTransferKit::accumulate( w, 4 ), 24 );
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DetailsUtils, adjacent_difference,
                                   DeviceType )
{
    Kokkos::View<int[5], DeviceType> v( "v" );
    auto v_host = Kokkos::create_mirror_view( v );
    v_host( 0 ) = 2;
    v_host( 1 ) = 4;
    v_host( 2 ) = 6;
    v_host( 3 ) = 8;
    v_host( 4 ) = 10;
    Kokkos::deep_copy( v, v_host );
    // In-place operation is not allowed
    TEST_THROW( DataTransferKit::adjacentDifference( v, v ),
                DataTransferKit::SearchException );
    auto w = Kokkos::create_mirror( DeviceType(), v );
    TEST_NOTHROW( DataTransferKit::adjacentDifference( v, w ) );
    auto w_host = Kokkos::create_mirror_view( w );
    Kokkos::deep_copy( w_host, w );
    std::vector<int> w_ref( 5, 2 );
    TEST_COMPARE_ARRAYS( w_host, w_ref );

    Kokkos::View<float *, DeviceType> x( "x", 10 );
    Kokkos::deep_copy( x, 3.14 );
    TEST_THROW( DataTransferKit::adjacentDifference( x, x ),
                DataTransferKit::SearchException );
    Kokkos::View<float[10], DeviceType> y( "y" );
    TEST_NOTHROW( DataTransferKit::adjacentDifference( x, y ) );
    std::vector<float> y_ref( 10 );
    y_ref[0] = 3.14;
    auto y_host = Kokkos::create_mirror_view( y );
    Kokkos::deep_copy( y_host, y );
    TEST_COMPARE_ARRAYS( y_host, y_ref );

    Kokkos::resize( x, 5 );
    TEST_THROW( DataTransferKit::adjacentDifference( y, x ),
                DataTransferKit::SearchException );
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DetailsUtils, min_and_max, DeviceType )
{
    Kokkos::View<int[4], DeviceType> v( "v" );
    DataTransferKit::iota( v );
    TEST_EQUALITY( 0, DataTransferKit::min( v ) );
    TEST_EQUALITY( 3, DataTransferKit::max( v ) );

    Kokkos::View<int *, DeviceType> w( "w", 7 );
    DataTransferKit::iota( w, 2 );
    TEST_EQUALITY( 2, DataTransferKit::min( w ) );
    TEST_EQUALITY( 8, DataTransferKit::max( w ) );
}

// Include the test macros.
#include "DataTransferKit_ETIHelperMacros.h"

// Create the test group
#define UNIT_TEST_GROUP( NODE )                                                \
    using DeviceType##NODE = typename NODE::device_type;                       \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DetailsUtils, iota,                  \
                                          DeviceType##NODE )                   \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DetailsUtils, prefix_sum,            \
                                          DeviceType##NODE )                   \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DetailsUtils, last_element,          \
                                          DeviceType##NODE )                   \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DetailsUtils, minmax,                \
                                          DeviceType##NODE )                   \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DetailsUtils, accumulate,            \
                                          DeviceType##NODE )                   \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DetailsUtils, adjacent_difference,   \
                                          DeviceType##NODE )                   \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DetailsUtils, min_and_max,           \
                                          DeviceType##NODE )

// Demangle the types
DTK_ETI_MANGLING_TYPEDEFS()

// Instantiate the tests
DTK_INSTANTIATE_N( UNIT_TEST_GROUP )
