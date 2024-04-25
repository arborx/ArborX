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
// - be not pure (i.e., allow output argument in the signature)
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
// We need DefaultCallback specialization for the case without a user-provided
// callback and using APIv2
template <>
struct is_constrained_callback<DefaultCallback> : std::true_type
{};
// We need DefaultCallbackWithRank specialization for the case without a
// user-provided callback and using APIv1
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

namespace Dispatch
{

template <typename Tag, typename Geometry>
struct approx_expand_by_radius;

template <typename Point>
struct approx_expand_by_radius<PointTag, Point>
{
  template <typename Float>
  KOKKOS_FUNCTION static auto apply(Point const &point, Float r)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Box>;
    using Coordinate = GeometryTraits::coordinate_type_t<Point>;
    auto const &hyper_point = reinterpret_cast<
        ExperimentalHyperGeometry::Point<DIM, Coordinate> const &>(point);
    return ExperimentalHyperGeometry::Sphere<DIM, Coordinate>{hyper_point, r};
  }
};

template <typename Box>
struct approx_expand_by_radius<BoxTag, Box>
{
  template <typename Float>
  KOKKOS_FUNCTION static auto apply(Box const &box, Float r)
  {
    Box new_box = box;
    auto &min_corner = new_box.minCorner();
    auto &max_corner = new_box.maxCorner();
    constexpr int DIM = GeometryTraits::dimension_v<Box>;
    for (int d = 0; d < DIM; ++d)
    {
      min_corner[d] -= r;
      max_corner[d] += r;
    }
    return new_box;
  }
};

template <typename Sphere>
struct approx_expand_by_radius<SphereTag, Sphere>
{
  template <typename Float>
  KOKKOS_FUNCTION static auto apply(Sphere const &sphere, Float r)
  {
    return Sphere{sphere.centroid(), sphere.radius() + r};
  }
};

template <typename Ray>
struct approx_expand_by_radius<RayTag, Ray>
{
  template <typename Float>
  KOKKOS_FUNCTION static auto const &apply(Ray const &ray, Float)
  {
    return ray;
  }
};

} // namespace Dispatch

template <typename Geometry, typename Float>
KOKKOS_INLINE_FUNCTION auto approx_expand_by_radius(Geometry const &geometry,
                                                    Float r)
{
  return Dispatch::approx_expand_by_radius<
      typename GeometryTraits::tag_t<Geometry>, Geometry>::apply(geometry, r);
}

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
  static KOKKOS_FUNCTION auto get(Self const &x, size_type i)
  {
    using Details::approx_expand_by_radius;
    return intersects(
        approx_expand_by_radius(getGeometry(x.predicates(i)), x.distances(i)));
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
        // NOTE: this will break if we are running multiple threads per query.
        // It could happen that the callback is called by different threads,
        // resulting in multiple outputs while having count = 1 in each thread.
        // As we don't envision this happening in the near future, it is ok for
        // now.
        ++count;
      });
      // If the user callback produces no output, we have nothing to attach the
      // distance to, which is problematic as we would not be able to do the
      // final filtering. If there are multiple outputs, it will currently
      // break our communication routines and filtering. We rely on 3-phase
      // nearest implementation for these cases.
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
    // NOTE cannot have extended __host__ __device__ lambda in constructor with
    // NVCC
    computeReversePermutation(exec_space);
  }

  template <typename ExecutionSpace>
  void computeReversePermutation(ExecutionSpace const &exec_space)
  {
    if (_tree.empty())
      return;

    auto const n = _tree.size();

    _rev_permute = Kokkos::View<unsigned int *, typename Tree::memory_space>(
        Kokkos::view_alloc(
            Kokkos::WithoutInitializing,
            "ArborX::DistributedTree::query::nearest::reverse_permutation"),
        n);
    Kokkos::parallel_for(
        "ArborX::DistributedTree::query::nearest::"
        "compute_reverse_permutation",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
        KOKKOS_CLASS_LAMBDA(int const i) {
          _rev_permute(HappyTreeFriends::getValue(_tree, i).index) = i;
        });
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
      // If the user callback produces no output, we have nothing to attach the
      // distance to, which is problematic as we would not be able to do the
      // final filtering. If there are multiple outputs, it will currently
      // break our communication routines and filtering. We rely on 3-phase
      // nearest implementation for these cases.
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
