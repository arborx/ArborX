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

#ifndef ARBORX_DETAILS_KOKKOS_EXT_VIEW_HELPERS_HPP
#define ARBORX_DETAILS_KOKKOS_EXT_VIEW_HELPERS_HPP

#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>

namespace KokkosExt
{

/** \brief Get a copy of the last element.
 *
 *  Returns a copy of the last element in the view on the host.  Note that it
 *  may require communication between host and device (e.g. if the view passed
 *  as an argument lives on the device).
 *
 *  \pre \c v is of rank 1 and not empty.
 */
template <class ExecutionSpace, class T, class... P>
typename Kokkos::ViewTraits<T, P...>::non_const_value_type
lastElement(ExecutionSpace const &space, Kokkos::View<T, P...> const &v)
{
  static_assert(unsigned(Kokkos::ViewTraits<T, P...>::rank) == unsigned(1),
                "lastElement requires Views of rank 1");
  auto const n = v.extent(0);
  ARBORX_ASSERT(n > 0);
  auto v_subview = Kokkos::subview(v, n - 1);
  typename Kokkos::ViewTraits<T, P...>::non_const_value_type v_host;
  Kokkos::deep_copy(space, v_host, v_subview);
  space.fence("ArborX::KokkosExt::lastElement (copy to host)");
  return v_host;
}

// FIXME it is not legal to use KOKKOS_IMPL_CTOR_DEFAULT_ARG
template <class ExecutionSpace, class View>
void reallocWithoutInitializing(ExecutionSpace const &space, View &v,
                                size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                                size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG)
{
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value);
  Kokkos::realloc(Kokkos::view_alloc(space, Kokkos::WithoutInitializing), v, n0,
                  n1, n2, n3, n4, n5, n6, n7);
}

template <class ExecutionSpace, class View>
void reallocWithoutInitializing(ExecutionSpace const &space, View &v,
                                const typename View::array_layout &layout)
{
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value);
  Kokkos::realloc(Kokkos::view_alloc(space, Kokkos::WithoutInitializing), v,
                  layout);
}

template <class ExecutionSpace, class View>
typename View::non_const_type clone(ExecutionSpace const &space, View const &v,
                                    std::string const &label)
{
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value);
  static_assert(Kokkos::is_view<View>::value);
  typename View::non_const_type w(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing, label),
      v.layout());
  Kokkos::deep_copy(space, w, v);
  return w;
}

template <class ExecutionSpace, class View>
typename View::non_const_type clone(ExecutionSpace const &space, View const &v)
{
  return clone(space, v, v.label());
}

template <class ExecutionSpace, class View>
typename View::non_const_type
cloneWithoutInitializingNorCopying(ExecutionSpace const &space, View const &v)
{
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value);
  static_assert(Kokkos::is_view<View>::value);
  return Kokkos::create_mirror(Kokkos::view_alloc(typename View::memory_space{},
                                                  space,
                                                  Kokkos::WithoutInitializing),
                               v);
}

} // namespace KokkosExt

#endif
