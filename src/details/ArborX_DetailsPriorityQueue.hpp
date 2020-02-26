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
#ifndef ARBORX_DETAILS_PRIORITY_QUEUE_HPP
#define ARBORX_DETAILS_PRIORITY_QUEUE_HPP

#include <ArborX_DetailsContainers.hpp>
#include <ArborX_DetailsHeap.hpp>

#include <Kokkos_Macros.hpp>

#include <cstddef>     // ptrdiff_t
#include <type_traits> // is_same
#include <utility>     // move, forward

namespace ArborX
{
namespace Details
{

template <typename T>
struct Less
{
public:
  KOKKOS_INLINE_FUNCTION bool operator()(T const &lhs, T const &rhs) const
  {
    return lhs < rhs;
  }
};

template <typename T>
struct Greater
{
public:
  KOKKOS_INLINE_FUNCTION bool operator()(T const &lhs, T const &rhs) const
  {
    return lhs > rhs;
  }
};

template <typename T, typename Compare = Less<T>,
          typename Container = StaticVector<T, 256>>
class PriorityQueue
{
public:
  using container_type = Container;
  using value_compare = Compare;
  using value_type = typename Container::value_type;
  using size_type = typename Container::size_type;
  using reference = typename Container::reference;
  using const_reference = typename Container::const_reference;
  static_assert(std::is_same<value_type, T>::value,
                "Template parameter T in PriorityQueue is not the same as "
                "the type of the elements stored by the underlying "
                "container Container::value_type");

  KOKKOS_DEFAULTED_FUNCTION PriorityQueue() = default;

  KOKKOS_FUNCTION PriorityQueue(Container const &c)
      : _c(c)
  {
    assert(_c.empty() || isHeap(_c.data(), _c.data() + _c.size(), _compare));
  }

  // Capacity
  KOKKOS_INLINE_FUNCTION bool empty() const { return _c.empty(); }
  KOKKOS_INLINE_FUNCTION size_type size() const { return _c.size(); }

  // Element access
  KOKKOS_INLINE_FUNCTION reference top() { return _c.front(); }
  KOKKOS_INLINE_FUNCTION const_reference top() const { return _c.front(); }

  // Modifiers
  KOKKOS_INLINE_FUNCTION void push(value_type const &value)
  {
    _c.pushBack(value);
    pushHeap(_c.data(), _c.data() + _c.size(), _compare);
  }
  KOKKOS_INLINE_FUNCTION void push(value_type &&value)
  {
    _c.pushBack(std::move(value));
    pushHeap(_c.data(), _c.data() + _c.size(), _compare);
  }
  template <class... Args>
  KOKKOS_INLINE_FUNCTION void emplace(Args &&... args)
  {
    _c.emplaceBack(std::forward<Args>(args)...);
    pushHeap(_c.data(), _c.data() + _c.size(), _compare);
  }
  KOKKOS_INLINE_FUNCTION void pop()
  {
    popHeap(_c.data(), _c.data() + _c.size(), _compare);
    _c.popBack();
  }
  // in TreeTraversal::nearestQuery, pop() is often followed by push which is
  // an opportunity for doing a single bubble-down operation instead of paying
  // for both one bubble-down and one bubble-up
  template <typename... Args>
  KOKKOS_INLINE_FUNCTION void popPush(Args &&... args)
  {
    assert(_c.size() > 0);
    __bubbleDown(_c.data(), std::ptrdiff_t(0), std::ptrdiff_t(_c.size()),
                 T{std::forward<Args>(args)...}, _compare);
  }

  // Accessors that shouldn't be there but that are convenient in
  // TreeTraversal::nearestQuery()
  KOKKOS_INLINE_FUNCTION typename Container::pointer data()
  {
    return _c.data();
  }
  KOKKOS_INLINE_FUNCTION value_compare const &valueComp() const
  {
    return _compare;
  }

private:
  Container _c;
  Compare _compare;
};

} // namespace Details
} // namespace ArborX

#endif
