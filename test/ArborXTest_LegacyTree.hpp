/****************************************************************************
 * Copyright (c) 2023 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_TEST_LEGACY_TREE_HPP
#define ARBORX_TEST_LEGACY_TREE_HPP

#include <ArborX_DetailsLegacy.hpp>

template <typename Tree>
class LegacyTree : public Tree
{
public:
  LegacyTree() = default;

  template <typename ExecutionSpace, typename Primitives>
  LegacyTree(ExecutionSpace const &space, Primitives const &primitives)
      : Tree(space,
             ArborX::Details::LegacyValues<Primitives,
                                           typename Tree::bounding_volume_type>{
                 primitives})
  {}

  template <typename ExecutionSpace, typename Predicates, typename Callback>
  void query(ExecutionSpace const &space, Predicates const &predicates,
             Callback const &callback) const
  {
    Tree::query(space, predicates, callback);
  }

  template <typename ExecutionSpace, typename Predicates, typename View,
            typename... Args>
  std::enable_if_t<Kokkos::is_view_v<std::decay_t<View>>>
  query(ExecutionSpace const &space, Predicates const &predicates, View &&view,
        Args &&...args) const
  {
    Tree::query(space, predicates, ArborX::Details::LegacyDefaultCallback{},
                std::forward<View>(view), std::forward<Args>(args)...);
  }

  template <typename ExecutionSpace, typename Predicates, typename Callback,
            typename OutputView, typename OffsetView, typename... Args>
  std::enable_if_t<!Kokkos::is_view_v<std::decay_t<Callback>>>
  query(ExecutionSpace const &space, Predicates const &predicates,
        Callback &&callback, OutputView &&out, OffsetView &&offset,
        Args &&...args) const
  {
    if constexpr (!ArborX::Details::is_tagged_post_callback<
                      std::decay_t<Callback>>::value)
    {
      Tree::query(
          space, predicates,
          ArborX::Details::LegacyCallbackWrapper<std::decay_t<Callback>>{
              std::forward<Callback>(callback)},
          std::forward<OutputView>(out), std::forward<OffsetView>(offset),
          std::forward<Args>(args)...);
    }
    else
    {
      Kokkos::View<int *, typename Tree::memory_space> indices(
          "Testing::indices", 0);
      Tree::query(space, predicates, ArborX::Details::LegacyDefaultCallback{},
                  indices, std::forward<OffsetView>(offset),
                  std::forward<Args>(args)...);
      callback(predicates, std::forward<OffsetView>(offset), indices,
               std::forward<OutputView>(out));
    }
  }
};

#endif
