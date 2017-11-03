/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#include <DTK_DetailsUtils.hpp>

#include <Teuchos_UnitTestHarness.hpp>

#include <algorithm>
#include <vector>

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
                DataTransferKit::DataTransferKitException );
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
                DataTransferKit::DataTransferKitException );
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
                DataTransferKit::DataTransferKitException );
    Kokkos::View<double[1], DeviceType> u( "u", 1 );
    Kokkos::deep_copy( u, 3.14 );
    TEST_FLOATING_EQUALITY( DataTransferKit::lastElement( u ), 3.14, 1e-14 );
}

// Include the test macros.
#include "DataTransferKitSearch_ETIHelperMacros.h"

// Create the test group
#define UNIT_TEST_GROUP( NODE )                                                \
    using DeviceType##NODE = typename NODE::device_type;                       \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DetailsUtils, prefix_sum,            \
                                          DeviceType##NODE )                   \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DetailsUtils, last_element,          \
                                          DeviceType##NODE )

// Demangle the types
DTK_ETI_MANGLING_TYPEDEFS()

// Instantiate the tests
DTK_INSTANTIATE_N( UNIT_TEST_GROUP )
