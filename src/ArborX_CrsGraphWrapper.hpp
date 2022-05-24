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

#ifndef ARBORX_CRS_GRAPH_WRAPPER_HPP
#define ARBORX_CRS_GRAPH_WRAPPER_HPP

#include "ArborX_DetailsCrsGraphWrapperImpl.hpp"

namespace ArborX
{

template <typename Tree, typename ExecutionSpace, typename Predicates,
          typename CallbackOrView, typename View, typename... Args>
inline void query(Tree const &tree, ExecutionSpace const &space,
                  Predicates const &predicates,
                  CallbackOrView &&callback_or_view, View &&view,
                  Args &&...args)
{
  Kokkos::Profiling::pushRegion("ArborX::query");

  Details::CrsGraphWrapperImpl::
      check_valid_callback_if_first_argument_is_not_a_view(callback_or_view,
                                                           predicates, view);

  using Access = AccessTraits<Predicates, ArborX::PredicatesTag>;
  using Tag = typename Details::AccessTraitsHelper<Access>::tag;

  ArborX::Details::CrsGraphWrapperImpl::queryDispatch(
      Tag{}, tree, space, predicates,
      std::forward<CallbackOrView>(callback_or_view), std::forward<View>(view),
      std::forward<Args>(args)...);

  Kokkos::Profiling::popRegion();
}

} // namespace ArborX

#endif
