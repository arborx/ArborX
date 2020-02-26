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
#ifndef ARBORX_DETAILS_STACK_HPP
#define ARBORX_DETAILS_STACK_HPP

#include <ArborX_DetailsContainers.hpp>

#include <Kokkos_Macros.hpp>

#include <type_traits> // is_same
#include <utility>     // move, forward

namespace ArborX
{
namespace Details
{

template <typename T, typename Container = StaticVector<T, 64>>
class Stack
{
public:
  using container_type = Container;
  using value_type = typename Container::value_type;
  using size_type = typename Container::size_type;
  using reference = typename Container::reference;
  using const_reference = typename Container::const_reference;
  static_assert(std::is_same<value_type, T>::value,
                "Template parameter T in Stack is not the same as "
                "the type of the elements stored by the underlying "
                "container Container::value_type");

  KOKKOS_DEFAULTED_FUNCTION Stack() = default;

  // Capacity
  KOKKOS_INLINE_FUNCTION bool empty() const { return _c.empty(); }
  KOKKOS_INLINE_FUNCTION size_type size() const { return _c.size(); }

  // Element access
  KOKKOS_INLINE_FUNCTION reference &top() { return _c.back(); }
  KOKKOS_INLINE_FUNCTION const_reference &top() const { return _c.back(); }

  // Modifiers
  KOKKOS_INLINE_FUNCTION void push(value_type const &value)
  {
    _c.pushBack(value);
  }
  KOKKOS_INLINE_FUNCTION void push(value_type &&value)
  {
    _c.pushBack(std::move(value));
  }
  template <class... Args>
  KOKKOS_INLINE_FUNCTION void emplace(Args &&... args)
  {
    _c.emplaceBack(std::forward<Args>(args)...);
  }
  KOKKOS_INLINE_FUNCTION void pop() { _c.popBack(); }

private:
  Container _c;
};

} // namespace Details
} // namespace ArborX

#endif
