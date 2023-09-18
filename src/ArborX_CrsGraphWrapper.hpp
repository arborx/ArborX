/****************************************************************************
 * Copyright (c) 2017-2023 by the ArborX authors                            *
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

#include "ArborX_DetailsTreeTraversal.hpp"

namespace ArborX
{

template <typename Tree, typename ExecutionSpace, typename Predicates,
          typename CallbackOrView, typename View, typename... Args>
inline void query(Tree const &tree, ExecutionSpace const &space,
                  Predicates const &predicates,
                  CallbackOrView &&callback_or_view, View &&view,
                  Args &&...args)
{
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value);

  tree.query(space, predicates, std::forward<CallbackOrView>(callback_or_view),
             std::forward<View>(view), std::forward<Args>(args)...);
}

template <typename Tree, typename Predicate, typename Callback>
KOKKOS_FUNCTION void kernel_query(Tree const &tree, Predicate const &predicate,
                                  Callback const &callback)
{
  ArborX::Details::TreeTraversal<Tree, /* Dummy */ std::true_type, Callback,
                                 typename Predicate::Tag>
      tree_traversal(tree, callback);
  tree_traversal.search(predicate);
}

} // namespace ArborX

#endif
