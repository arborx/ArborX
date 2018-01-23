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
#ifndef DTK_DETAILS_STACK_HPP
#define DTK_DETAILS_STACK_HPP

#include <Kokkos_Macros.hpp>

#include <cassert>
#include <cstdlib>
#include <utility>

namespace DataTransferKit
{
namespace Details
{

template <typename T>
class Stack
{
  public:
    using SizeType = size_t;

    KOKKOS_FUNCTION Stack() = default;

    KOKKOS_INLINE_FUNCTION bool empty() const { return _size == 0; }

    template <typename... Args>
    KOKKOS_INLINE_FUNCTION void push( Args &&... args )
    {
        assert( _size < _max_size );
        _stack[_size++] = T( std::forward<Args>( args )... );
    }

    KOKKOS_INLINE_FUNCTION void pop()
    {
        assert( _size > 0 );
        _size--;
    }

    KOKKOS_INLINE_FUNCTION T const &top() const
    {
        assert( _size > 0 );
        return *( _stack + _size - 1 );
    }

  private:
    static SizeType constexpr _max_size = 64;
    T _stack[_max_size];
    SizeType _size = 0;
};

} // end namespace Details
} // end namespace DataTransferKit

#endif
