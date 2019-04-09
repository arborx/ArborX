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

#include <DTK_DetailsBatchedQueries.hpp>

#include "DTK_EnableDeviceTypes.hpp" // DTK_SEARCH_DEVICE_TYPES
#include "DTK_EnableViewComparison.hpp"

#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE DetailsBatchedQueries

namespace tt = boost::test_tools;

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

BOOST_AUTO_TEST_CASE_TEMPLATE( permute_offset_and_indices, DeviceType,
                               DTK_SEARCH_DEVICE_TYPES )
{
    Kokkos::View<int *, DeviceType> offset( "offset" );
    Kokkos::View<int *, DeviceType> indices( "indices" );

    Kokkos::View<size_t *, DeviceType> permute( "permute" );

    BOOST_CHECK_THROW(
        DataTransferKit::Details::BatchedQueries<
            DeviceType>::reversePermutation( permute, offset, indices ),
        DataTransferKit::SearchException );

    Kokkos::resize( offset, 1 );
    BOOST_CHECK_NO_THROW(
        DataTransferKit::Details::BatchedQueries<
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
    BOOST_TEST( offset == offset_ref, tt::per_element() );
    BOOST_TEST( indices == indices_ref, tt::per_element() );
}
