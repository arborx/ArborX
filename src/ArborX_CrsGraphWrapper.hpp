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

#ifndef ARBORX_CRS_GRAPH_WRAPPER_HPP
#define ARBORX_CRS_GRAPH_WRAPPER_HPP

#include "ArborX_DetailsCrsGraphWrapperImpl.hpp"

namespace ArborX
{

template <typename ExecutionSpace, typename Tree, typename Predicates,
          typename CallbackOrView, typename View, typename... Args>
inline void query_crs(ExecutionSpace const &space, Tree const &tree,
                      Predicates const &predicates,
                      CallbackOrView &&callback_or_view, View &&view,
                      Args &&... args)
{
  Details::CrsGraphWrapperImpl::query(
      space, tree, predicates, std::forward<CallbackOrView>(callback_or_view),
      std::forward<View>(view), std::forward<Args>(args)...);
}

} // namespace ArborX

#endif
