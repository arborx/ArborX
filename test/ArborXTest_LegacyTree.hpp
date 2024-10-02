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

#include <detail/ArborX_AccessTraits.hpp>
#include <detail/ArborX_Algorithms.hpp>
#include <detail/ArborX_Callbacks.hpp>
#include <detail/ArborX_PairValueIndex.hpp>

#include <Kokkos_Macros.hpp>

#include <type_traits>
#include <utility>

template <typename Primitives, typename BoundingVolume>
class LegacyValues
{
  Primitives _primitives;
  using Access = ArborX::AccessTraits<Primitives, ArborX::PrimitivesTag>;

public:
  using memory_space = typename Access::memory_space;
  using index_type = unsigned;
  using value_type = ArborX::PairValueIndex<BoundingVolume, index_type>;
  using size_type =
      Kokkos::detected_t<ArborX::Details::AccessTraitsSizeArchetypeExpression,
                         Access, Primitives>;

  LegacyValues(Primitives const &primitives)
      : _primitives(primitives)
  {}

  KOKKOS_FUNCTION
  auto operator()(size_type i) const
  {
    using Primitive = std::decay_t<decltype(Access::get(_primitives, i))>;
    if constexpr (std::is_same_v<BoundingVolume, Primitive>)
    {
      return value_type{Access::get(_primitives, i), (index_type)i};
    }
    else
    {
      using ArborX::Details::expand;
      BoundingVolume bounding_volume{};
      expand(bounding_volume, Access::get(_primitives, i));
      return value_type{bounding_volume, (index_type)i};
    }
#if defined(KOKKOS_COMPILER_INTEL) && (KOKKOS_COMPILER_INTEL <= 2021)
    // FIXME_INTEL: workaround for spurious "missing return
    // statement at end of non-void function" warning
    return value_type{};
#endif
  }

  KOKKOS_FUNCTION
  size_type size() const { return Access::size(_primitives); }
};

template <typename Primitives, typename BoundingVolume>
struct ArborX::AccessTraits<LegacyValues<Primitives, BoundingVolume>,
                            ArborX::PrimitivesTag>
{
  using self_type = LegacyValues<Primitives, BoundingVolume>;

public:
  using memory_space = typename self_type::memory_space;

  KOKKOS_FUNCTION static decltype(auto) get(self_type const &w, int i)
  {
    return w(i);
  }

  KOKKOS_FUNCTION
  static decltype(auto) size(self_type const &w) { return w.size(); }
};

template <typename Callback>
struct LegacyCallbackWrapper
{
  Callback _callback;

  template <typename Predicate, typename Value, typename Index>
  KOKKOS_FUNCTION auto
  operator()(Predicate const &predicate,
             ArborX::PairValueIndex<Value, Index> const &value) const
  {
    return _callback(predicate, value.index);
  }

  template <typename Predicate, typename Value, typename Index, typename Output>
  KOKKOS_FUNCTION void
  operator()(Predicate const &predicate,
             ArborX::PairValueIndex<Value, Index> const &value,
             Output const &out) const
  {
    // APIv1 callback has the signature operator()(Query, int)
    // As we store PairValueIndex with potentially non int index (like
    // unsigned), we explicitly cast it here.
    _callback(predicate, (int)value.index, out);
  }
};

template <typename Tree>
class LegacyTree : public Tree
{
public:
  LegacyTree() = default;

  template <typename ExecutionSpace, typename Primitives>
  LegacyTree(ExecutionSpace const &space, Primitives const &primitives)
      : Tree(space,
             LegacyValues<Primitives, typename Tree::bounding_volume_type>{
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
      Tree::query(space, predicates,
                  LegacyCallbackWrapper<std::decay_t<Callback>>{
                      std::forward<Callback>(callback)},
                  std::forward<OutputView>(out),
                  std::forward<OffsetView>(offset),
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
