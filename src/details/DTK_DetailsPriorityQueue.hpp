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
#ifndef DTK_DETAILS_PRIORITY_QUEUE_HPP
#define DTK_DETAILS_PRIORITY_QUEUE_HPP

#include <Kokkos_Macros.hpp>

#include <cassert>
#include <cstdlib>
#include <utility>

namespace DataTransferKit
{
namespace Details
{

template <typename T>
struct Less
{
  public:
    KOKKOS_INLINE_FUNCTION bool operator()( T const &x, T const &y )
    {
        return x < y;
    }
};

template <typename T, typename Compare = Less<T>>
class PriorityQueue
{
  public:
    using SizeType = size_t;

    KOKKOS_FUNCTION PriorityQueue() = default;

    KOKKOS_INLINE_FUNCTION bool empty() const { return _size == 0; }

    template <typename... Args>
    KOKKOS_FUNCTION void push( Args &&... args )
    {
        // ensure the queue is not already full
        assert( _size < _max_size );

        // construct the element to compare to those in the queue
        T elem( std::forward<Args>( args )... );

        // find position of the new element in the sorted array
        // COMMENT: could consider implementing a binary search here
        SizeType pos;
        for ( pos = 0; pos < _size; ++pos )
            if ( !_compare( _queue[pos], elem ) )
                break;

        // move memory to make room for it
        for ( SizeType tmp = _size; tmp > pos; --tmp )
            _queue[tmp] = _queue[tmp - 1];

        // insert the new element
        _queue[pos] = elem;

        // update the size of the queue
        ++_size;
    }

    KOKKOS_INLINE_FUNCTION void pop()
    {
        assert( _size > 0 );
        _size--;
    }

    KOKKOS_INLINE_FUNCTION T const &top() const
    {
        assert( _size > 0 );
        return *( _queue + _size - 1 );
    }

  private:
    static SizeType constexpr _max_size = 256;
    T _queue[_max_size];
    SizeType _size = 0;
    Compare _compare;
};

} // end namespace Details
} // end namespace DataTransferKit

#endif
