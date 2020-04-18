/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_DETAILS_CONTAINERS_HPP
#define ARBORX_DETAILS_CONTAINERS_HPP

#include <Kokkos_Macros.hpp>

#include <cassert> // assert
#include <cstddef> // size_t, ptrdiff_t
#include <utility> // move, forward

namespace ArborX
{
namespace Details
{

// dynamic vector with fixed maximum size
template <typename T, std::size_t N>
class StaticVector
{
  // clang-format off
  public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type &;
    using const_reference = value_type const &;
    using pointer = value_type *;
    using const_pointer = value_type const *;
    KOKKOS_DEFAULTED_FUNCTION StaticVector() = default;
    KOKKOS_INLINE_FUNCTION bool empty() const { return _size == 0; }
    KOKKOS_INLINE_FUNCTION size_type size() const { return _size; }
    KOKKOS_INLINE_FUNCTION constexpr size_type maxSize() const { return N; }
    KOKKOS_INLINE_FUNCTION constexpr size_type capacity() const { return N; }
    KOKKOS_INLINE_FUNCTION reference operator[]( size_type pos ) { assert(pos < size()); return _data[pos]; }
    KOKKOS_INLINE_FUNCTION const_reference operator[]( size_type pos ) const { assert(pos < size()); return _data[pos]; }
    KOKKOS_INLINE_FUNCTION reference back() { assert(size() > 0 ); return _data[_size - 1]; }
    KOKKOS_INLINE_FUNCTION const_reference back() const { assert(size() > 0); return _data[_size - 1]; }
    KOKKOS_INLINE_FUNCTION void pushBack(T const &value) { assert(size() < maxSize()); _data[_size++] = value; }
    KOKKOS_INLINE_FUNCTION void pushBack(T &&value) { assert(size() < maxSize()); _data[_size++] = std::move(value); }
    template<class... Args>
    KOKKOS_INLINE_FUNCTION void emplaceBack(Args&&... args) { assert(size() < maxSize()); ::new (static_cast<void*>(_data + _size++)) T(std::forward<Args>(args)...); }
    KOKKOS_INLINE_FUNCTION void popBack() { assert(size() > 0); _size--; }
    KOKKOS_INLINE_FUNCTION reference front() { assert(size() > 0); return _data[0]; }
    KOKKOS_INLINE_FUNCTION const_reference front() const { assert(size() > 0); return _data[0]; }
    KOKKOS_INLINE_FUNCTION void clear() { _size = 0; }
    KOKKOS_INLINE_FUNCTION pointer data() { return _data; }
    KOKKOS_INLINE_FUNCTION constexpr const_pointer data() const { return _data; }

  private:
    value_type _data[N];
    size_type _size = 0;
  // clang-format on
};

template <typename T>
class UnmanagedStaticVector
{
  // clang-format off
  public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = value_type &;
    using const_reference = value_type const &;
    using pointer = value_type *;
    using const_pointer = value_type const *;
    KOKKOS_FUNCTION UnmanagedStaticVector( pointer ptr, size_type max_size ) : _ptr(ptr) , _max_size(max_size) { assert(ptr != nullptr); }
    KOKKOS_INLINE_FUNCTION bool empty() const { return _size == 0; }
    KOKKOS_INLINE_FUNCTION size_type size() const { return _size; }
    KOKKOS_INLINE_FUNCTION constexpr size_type maxSize() const { return _max_size; }
    KOKKOS_INLINE_FUNCTION constexpr size_type capacity() const { return _max_size; }
    KOKKOS_INLINE_FUNCTION reference operator[]( size_type pos ) { assert(pos < size()); return *(_ptr + pos); }
    KOKKOS_INLINE_FUNCTION const_reference operator[]( size_type pos ) const { assert(pos < size()); return *(_ptr + pos); }
    KOKKOS_INLINE_FUNCTION reference back() { assert(size() > 0); return *(_ptr + _size - 1); }
    KOKKOS_INLINE_FUNCTION const_reference back() const { assert(size() > 0); return *(_ptr + _size - 1); }
    KOKKOS_INLINE_FUNCTION void pushBack(T const &value) { assert(size() < maxSize()); *(_ptr + _size++) = value; }
    KOKKOS_INLINE_FUNCTION void pushBack(T &&value) { assert(size() < maxSize()); *(_ptr + _size++) = std::move(value); }
    template<class... Args>
    KOKKOS_INLINE_FUNCTION void emplaceBack(Args&&... args) { assert(size() < maxSize()); ::new (static_cast<void*>(_ptr + _size++)) T(std::forward<Args>(args)...); }
    KOKKOS_INLINE_FUNCTION void popBack() { assert(size() > 0); _size--; }
    KOKKOS_INLINE_FUNCTION reference front() { assert(size() > 0); return *(_ptr + 0); }
    KOKKOS_INLINE_FUNCTION const_reference front() const { assert(size() > 0); return *(_ptr + 0); }
    KOKKOS_INLINE_FUNCTION void clear() { _size = 0; }
    KOKKOS_INLINE_FUNCTION pointer data() { return _ptr; }
    KOKKOS_INLINE_FUNCTION const_pointer data() const { return _ptr; }

  private:
    pointer _ptr = nullptr;
    size_type const _max_size = 0;
    size_type _size = 0;
  // clang-format on
};

} // namespace Details
} // namespace ArborX

#endif
