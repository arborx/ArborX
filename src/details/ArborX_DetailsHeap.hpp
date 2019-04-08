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
#ifndef ARBORX_DETAILS_HEAP_HPP
#define ARBORX_DETAILS_HEAP_HPP

#include <Kokkos_Macros.hpp>

#include <iterator> // iterator_traits
#include <utility>  // move

namespace DataTransferKit
{
namespace Details
{

template <typename RandomIterator, typename Compare>
KOKKOS_INLINE_FUNCTION bool isHeap( RandomIterator first, RandomIterator last,
                                    Compare comp )
{
    using DistanceType =
        typename std::iterator_traits<RandomIterator>::difference_type;
    DistanceType len = last - first;
    for ( DistanceType pos = 0; pos < len; ++pos )
    {
        for ( DistanceType child : {2 * pos + 1, 2 * pos + 2} )
            if ( child < len && comp( *( first + pos ), *( first + child ) ) )
                return false;
    }
    return true;
}

template <typename RandomIterator, typename DistanceType, typename ValueType,
          typename Compare>
KOKKOS_INLINE_FUNCTION void __bubbleUp( RandomIterator first, DistanceType pos,
                                        DistanceType top, ValueType val,
                                        Compare comp )
{
    DistanceType parent = ( pos - 1 ) / 2;
    while ( pos > top && comp( *( first + parent ), val ) )
    {
        *( first + pos ) = std::move( *( first + parent ) );
        pos = parent;
        parent = ( pos - 1 ) / 2;
    }
    *( first + pos ) = std::move( val );
}

template <typename RandomIterator, typename Compare>
KOKKOS_INLINE_FUNCTION void pushHeap( RandomIterator first, RandomIterator last,
                                      Compare comp )
{
    using DistanceType =
        typename std::iterator_traits<RandomIterator>::difference_type;
    using ValueType = typename std::iterator_traits<RandomIterator>::value_type;
    if ( last - first > 1 )
    {
        ValueType value = std::move( *( last - 1 ) );
        __bubbleUp( first, DistanceType( ( last - first ) - 1 ),
                    DistanceType( 0 ), std::move( value ), comp );
    }
}

template <typename RandomIterator, typename DistanceType, typename ValueType,
          typename Compare>
KOKKOS_INLINE_FUNCTION void __bubbleDown( RandomIterator first,
                                          DistanceType pos, DistanceType len,
                                          ValueType val, Compare comp )
{
    DistanceType child = 2 * pos + 1;
    // if right child exists and compares greater than left child
    if ( child + 1 < len && comp( *( first + child ), *( first + child + 1 ) ) )
        ++child;

    while ( child < len && comp( val, *( first + child ) ) )
    {
        *( first + pos ) = std::move( *( first + child ) );
        pos = child;
        child = 2 * pos + 1;
        if ( child + 1 < len &&
             comp( *( first + child ), *( first + child + 1 ) ) )
            ++child;
    }
    *( first + pos ) = std::move( val );
}

template <typename RandomIterator, typename Compare>
KOKKOS_INLINE_FUNCTION void popHeap( RandomIterator first, RandomIterator last,
                                     Compare comp )
{
    using DistanceType =
        typename std::iterator_traits<RandomIterator>::difference_type;
    using ValueType = typename std::iterator_traits<RandomIterator>::value_type;
    if ( last - first > 1 )
    {
        ValueType value = std::move( *first );
        __bubbleDown( first, DistanceType( 0 ),
                      DistanceType( ( last - first ) - 1 ),
                      std::move( *( last - 1 ) ), comp );
        *( last - 1 ) = std::move( value );
    }
}

template <typename RandomIterator, typename Compare>
KOKKOS_INLINE_FUNCTION void sortHeap( RandomIterator first, RandomIterator last,
                                      Compare comp )
{
    while ( first != last )
        popHeap( first, last--, comp );
}

} // namespace Details
} // namespace DataTransferKit

#endif
