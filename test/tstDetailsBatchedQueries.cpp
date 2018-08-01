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

#include <DTK_DetailsBatchedQueries.hpp>

#include <Teuchos_UnitTestHarness.hpp>

template <typename DeviceType, typename ValueType>
Kokkos::View<ValueType *, DeviceType> toView( std::vector<ValueType> const &v )
{
    Kokkos::View<ValueType *, DeviceType> w( "whocares", v.size() );
    auto w_host = Kokkos::create_mirror_view( w );
    for ( int i = 0; i < w.extent_int( 0 ); ++i )
        w_host( i ) = v[i];
    Kokkos::deep_copy( w, w_host );
    return w;
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( BatchedQueries, permute_offset_and_indices,
                                   DeviceType )
{
    Kokkos::View<int *, DeviceType> offset( "offset" );
    Kokkos::View<int *, DeviceType> indices( "indices" );

    Kokkos::View<size_t *, DeviceType> permute( "permute" );

    TEST_THROW( DataTransferKit::Details::BatchedQueries<
                    DeviceType>::reversePermutation( permute, offset, indices ),
                DataTransferKit::DataTransferKitException );

    Kokkos::resize( offset, 1 );
    TEST_NOTHROW( DataTransferKit::Details::BatchedQueries<
                  DeviceType>::reversePermutation( permute, offset, indices ) );

    std::vector<int> offset_ = {0, 0, 1, 3, 6, 10};
    std::vector<int> indices_ = {1, 2, 2, 3, 3, 3, 4, 4, 4, 4};
    std::vector<size_t> permute_ = {4, 3, 2, 1, 0};
    std::vector<int> offset_ref = {0, 4, 7, 9, 10, 10};
    std::vector<int> indices_ref = {4, 4, 4, 4, 3, 3, 3, 2, 2, 1};

    std::tie( offset, indices ) = DataTransferKit::Details::BatchedQueries<
        DeviceType>::reversePermutation( toView<DeviceType>( permute_ ),
                                         toView<DeviceType>( offset_ ),
                                         toView<DeviceType>( indices_ ) );
    TEST_COMPARE_ARRAYS( offset, offset_ref );
    TEST_COMPARE_ARRAYS( indices, indices_ref );
}

// Include the test macros.
#include "DataTransferKitSearch_ETIHelperMacros.h"

// Create the test group
#define UNIT_TEST_GROUP( NODE )                                                \
    using DeviceType##NODE = typename NODE::device_type;                       \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT(                                      \
        BatchedQueries, permute_offset_and_indices, DeviceType##NODE )

// Demangle the types
DTK_ETI_MANGLING_TYPEDEFS()

// Instantiate the tests
DTK_INSTANTIATE_N( UNIT_TEST_GROUP )
