/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_ENABLE_VIEW_COMPARISON_HPP
#define ARBORX_ENABLE_VIEW_COMPARISON_HPP

#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp> // is_accessible_from_host

#include <Kokkos_Core.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/utils/is_forward_iterable.hpp>

template <typename T>
struct KokkosViewIterator;

template <typename T, typename... P>
struct KokkosViewIterator<Kokkos::View<T, P...>>
{
  using self_t = KokkosViewIterator<Kokkos::View<T, P...>>;
  using view_t = Kokkos::View<T, P...>;
  using value_t = typename view_t::value_type;

  value_t &operator*()
  {
    return view.access(index[0], index[1], index[2], index[3], index[4],
                       index[5], index[6], index[7]);
  }

  value_t *operator->() { return &operator*(); }

  self_t &operator++()
  {
    index[7]++;
    auto const layout = view.layout();

    for (std::size_t i = 7; i > 0; i--)
      if (index[i] == layout.dimension[i] ||
          layout.dimension[i] == KOKKOS_INVALID_INDEX)
      {
        index[i] = 0;
        index[i - 1]++;
      }

    return *this;
  }

  self_t operator++(int)
  {
    self_t old = *this;
    operator++();
    return old;
  }

  static self_t begin(view_t const &view)
  {
    return {view, {{0, 0, 0, 0, 0, 0, 0, 0}}};
  }

  static self_t end(view_t const &view)
  {
    auto const layout = view.layout();
    return {view, {{layout.dimension[0], 0, 0, 0, 0, 0, 0, 0}}};
  }

  view_t view;
  Kokkos::Array<std::size_t, 8> index;
};

// Enable element-wise comparison for views that are accessible from the host
namespace boost
{
namespace unit_test
{

template <typename T, typename... P>
struct is_forward_iterable<Kokkos::View<T, P...>> : public boost::mpl::true_
{
  // NOTE Prefer static assertion to SFINAE because error message about no
  // operator== for the operands is not as clear.
  static_assert(
      KokkosExt::is_accessible_from_host<Kokkos::View<T, P...>>::value,
      "Restricted to host-accessible views");
};

template <typename T, typename... P>
struct bt_iterator_traits<Kokkos::View<T, P...>, true>
{
  using view_type = Kokkos::View<T, P...>;
  using value_type = typename view_type::value_type;
  using const_iterator = KokkosViewIterator<view_type>;

  static const_iterator begin(view_type const &v)
  {
    return const_iterator::begin(v);
  }

  static const_iterator end(view_type const &v)
  {
    return const_iterator::end(v);
  }

  static std::size_t size(view_type const &v) { return v.size(); }
};

template <typename T, size_t N, typename Proxy>
struct is_forward_iterable<Kokkos::Array<T, N, Proxy>>
    : public boost::mpl::true_
{};

template <typename T, size_t N, typename Proxy>
struct bt_iterator_traits<Kokkos::Array<T, N, Proxy>, true>
{
  using array_type = Kokkos::Array<T, N, Proxy>;
  using value_type = typename array_type::value_type;
  using const_iterator = typename array_type::const_pointer;
  static const_iterator begin(array_type const &v) { return v.data(); }
  static const_iterator end(array_type const &v) { return v.data() + v.size(); }
  static std::size_t size(array_type const &v) { return v.size(); }
};

} // namespace unit_test
} // namespace boost

#endif
