/****************************************************************************
 * Copyright (c) 2017-2024 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_DETAILS_DISTRIBUTED_TREE_NEAREST_HELPERS_HPP
#define ARBORX_DETAILS_DISTRIBUTED_TREE_NEAREST_HELPERS_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_Box.hpp>
#include <ArborX_DetailsHappyTreeFriends.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Point.hpp>
#include <ArborX_Ray.hpp>
#include <ArborX_Sphere.hpp>

namespace ArborX
{

namespace Experimental
{

// Constrained callback is a callback that a user promises to:
// - be not pure
// - be allowed to be called on non-final results
// - produce exactly one result for each match
template <class Callback>
struct ConstrainedDistributedNearestCallback
{
  Callback _callback;

  template <class... Args>
  KOKKOS_FUNCTION void operator()(Args &&...args) const
  {
    _callback((Args &&) args...);
  }
};

template <class Callback>
auto declare_callback_constrained(Callback const &callback)
{
  return ConstrainedDistributedNearestCallback<Callback>{callback};
}

} // namespace Experimental

namespace Details
{

struct DefaultCallbackWithRank
{
  int _rank;

  template <typename Predicate, typename Value, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const &, Value const &value,
                                  OutputFunctor const &out) const
  {
    out({value, _rank});
  }
};

template <class Callback>
struct is_constrained_callback : std::false_type
{};
template <class Callback>
struct is_constrained_callback<
    Experimental::ConstrainedDistributedNearestCallback<Callback>>
    : std::true_type
{};
template <>
struct is_constrained_callback<DefaultCallback> : std::true_type
{};
template <>
struct is_constrained_callback<DefaultCallbackWithRank> : std::true_type
{};

template <class Callback>
inline constexpr bool is_constrained_callback_v =
    is_constrained_callback<Callback>::value;

template <class Predicates, class Distances>
struct WithinDistanceFromPredicates
{
  Predicates predicates;
  Distances distances;
};
} // namespace Details

template <class Predicates, class Distances>
struct AccessTraits<
    Details::WithinDistanceFromPredicates<Predicates, Distances>, PredicatesTag>
{
  using Predicate = typename Predicates::value_type;
  using Geometry =
      std::decay_t<decltype(getGeometry(std::declval<Predicate const &>()))>;
  using Self = Details::WithinDistanceFromPredicates<Predicates, Distances>;

  using memory_space = typename Predicates::memory_space;
  using size_type = decltype(std::declval<Predicates const &>().size());

  static KOKKOS_FUNCTION size_type size(Self const &x)
  {
    return x.predicates.size();
  }
  template <class Dummy = Geometry,
            std::enable_if_t<std::is_same_v<Dummy, Geometry> &&
                             std::is_same_v<Dummy, Point>> * = nullptr>
  static KOKKOS_FUNCTION auto get(Self const &x, size_type i)
  {
    auto const point = getGeometry(x.predicates(i));
    auto const distance = x.distances(i);
    return intersects(Sphere{point, distance});
  }
  template <class Dummy = Geometry,
            std::enable_if_t<std::is_same_v<Dummy, Geometry> &&
                             std::is_same_v<Dummy, Box>> * = nullptr>
  static KOKKOS_FUNCTION auto get(Self const &x, size_type i)
  {
    auto box = getGeometry(x.predicates(i));
    auto &min_corner = box.minCorner();
    auto &max_corner = box.maxCorner();
    auto const distance = x.distances(i);
    for (int d = 0; d < 3; ++d)
    {
      min_corner[d] -= distance;
      max_corner[d] += distance;
    }
    return intersects(box);
  }
  template <class Dummy = Geometry,
            std::enable_if_t<std::is_same_v<Dummy, Geometry> &&
                             std::is_same_v<Dummy, Sphere>> * = nullptr>
  static KOKKOS_FUNCTION auto get(Self const &x, size_type i)
  {
    auto const sphere = getGeometry(x.predicates(i));
    auto const distance = x.distances(i);
    return intersects(Sphere{sphere.centroid(), distance + sphere.radius()});
  }
  template <
      class Dummy = Geometry,
      std::enable_if_t<std::is_same_v<Dummy, Geometry> &&
                       std::is_same_v<Dummy, Experimental::Ray>> * = nullptr>
  static KOKKOS_FUNCTION auto get(Self const &x, size_type i)
  {
    auto const ray = getGeometry(x.predicates(i));
    return intersects(ray);
  }
};

namespace Details
{

template <typename Tree, typename Callback, typename OutValue, bool UseValues>
struct CallbackWithDistance
{
  Tree _tree;
  Callback _callback;

  template <typename ExecutionSpace>
  CallbackWithDistance(ExecutionSpace const &, Tree const &tree,
                       Callback const &callback)
      : _tree(tree)
      , _callback(callback)
  {}

  template <typename Query, typename Value, typename Output>
  KOKKOS_FUNCTION void operator()(Query const &query, Value const &value,
                                  Output const &out) const
  {
    if constexpr (UseValues)
    {
      OutValue out_value;
      [[maybe_unused]] int count = 0;
      _callback(query, value, [&](OutValue const &ov) {
        out_value = ov;
        ++count;
      });
      KOKKOS_ASSERT(count == 1);
      out({out_value,
           distance(getGeometry(query), _tree.indexable_get()(value))});
    }
    else
      out(distance(getGeometry(query), _tree.indexable_get()(value)));
  }
};

template <typename MemorySpace, typename Callback, typename OutValue,
          bool UseValues>
struct CallbackWithDistance<
    BoundingVolumeHierarchy<MemorySpace, Details::LegacyDefaultTemplateValue,
                            Details::DefaultIndexableGetter,
                            ExperimentalHyperGeometry::Box<3, float>>,
    Callback, OutValue, UseValues>
{
  using Tree =
      BoundingVolumeHierarchy<MemorySpace, Details::LegacyDefaultTemplateValue,
                              Details::DefaultIndexableGetter,
                              ExperimentalHyperGeometry::Box<3, float>>;

  Tree _tree;
  Callback _callback;
  Kokkos::View<unsigned int *, typename Tree::memory_space> _rev_permute;

  template <typename ExecutionSpace>
  CallbackWithDistance(ExecutionSpace const &exec_space, Tree const &tree,
                       Callback const &callback)
      : _tree(tree)
      , _callback(callback)
  {
    // NOTE cannot have extended __host__ __device__  lambda in constructor with
    // NVCC
    computeReversePermutation(exec_space);
  }

  template <typename ExecutionSpace>
  void computeReversePermutation(ExecutionSpace const &exec_space)
  {
    auto const n = _tree.size();

    _rev_permute = Kokkos::View<unsigned int *, typename Tree::memory_space>(
        Kokkos::view_alloc(
            Kokkos::WithoutInitializing,
            "ArborX::DistributedTree::query::nearest::reverse_permutation"),
        n);
    if (!_tree.empty())
    {
      Kokkos::parallel_for(
          "ArborX::DistributedTree::query::nearest::"
          "compute_reverse_permutation",
          Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
          KOKKOS_CLASS_LAMBDA(int const i) {
            _rev_permute(HappyTreeFriends::getValue(_tree, i).index) = i;
          });
    }
  }

  template <typename Query, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Query const &query, int index,
                                  OutputFunctor const &out) const
  {
    // TODO: This breaks the abstraction of the distributed Tree not knowing
    // the details of the local tree. Right now, this is the only way. Will
    // need to be fixed with a proper callback abstraction.
    int const leaf_node_index = _rev_permute(index);
    auto const &leaf_node_bounding_volume =
        HappyTreeFriends::getIndexable(_tree, leaf_node_index);
    if constexpr (UseValues)
    {
      OutValue out_value;
      [[maybe_unused]] int count = 0;
      _callback(query, index, [&](OutValue const &ov) {
        out_value = ov;
        ++count;
      });
      KOKKOS_ASSERT(count == 1);
      out({out_value, distance(getGeometry(query), leaf_node_bounding_volume)});
    }
    else
      out(distance(getGeometry(query), leaf_node_bounding_volume));
  }
};

} // namespace Details
} // namespace ArborX

#endif
