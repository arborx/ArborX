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
    KOKKOS_INLINE_FUNCTION void push( Args &&... args )
    {
        // ensure the heap is not already full
        assert( _size < _max_size );

        // add the element to the bottom level of the heap
        SizeType pos = _size;
        T elem{std::forward<Args>( args )...};

        // perform up-heap operation
        while ( pos > 0 )
        {
            // compare the added element with its parent
            // if they are in correct order, stop
            // if not, swap them and continue
            SizeType const parent = ( pos - 1 ) / 2;
            if ( !_compare( _heap[parent], elem ) )
                break;
            _heap[pos] = _heap[parent];
            pos = parent;
        }

        _heap[pos] = elem;

        // update the size of the heap
        ++_size;
    }

    KOKKOS_INLINE_FUNCTION void pop()
    {
        // ensure that the heap is not empty
        assert( _size > 0 );

        // replace the root with the last element on the last level
        SizeType pos = 0;

        // perform down-heap operation
        while ( true )
        {
            // compare the new root with its children
            // if they are in the correct order, stop
            // if not, swap the element with one of its children and continue
            SizeType const left_child = 2 * pos + 1;
            SizeType const right_child = 2 * pos + 2;
            SizeType next_pos = _size - 1;
            for ( SizeType child : {left_child, right_child} )
                if ( child < _size - 1 &&
                     _compare( _heap[next_pos], _heap[child] ) )
                    next_pos = child;
            if ( next_pos == _size - 1 )
                break;
            _heap[pos] = _heap[next_pos];
            pos = next_pos;
        }

        _heap[pos] = _heap[_size - 1];

        // update the size of the heap
        _size--;
    }

    KOKKOS_INLINE_FUNCTION T const &top() const
    {
        assert( _size > 0 );
        return _heap[0];
    }

  private:
    static SizeType constexpr _max_size = 256;
    T _heap[_max_size];
    SizeType _size = 0;
    Compare _compare;
};

} // namespace Details
} // namespace DataTransferKit

#endif
