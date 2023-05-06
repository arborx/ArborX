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

#ifndef ARBORX_MINIMUM_SPANNING_TREE_HPP
#define ARBORX_MINIMUM_SPANNING_TREE_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_DetailsKokkosExtMinMaxOperations.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_DetailsMutualReachabilityDistance.hpp>
#include <ArborX_DetailsTreeNodeLabeling.hpp>
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_DetailsWeightedEdge.hpp>
#include <ArborX_HyperBox.hpp>
#include <ArborX_LinearBVH.hpp>

#if KOKKOS_VERSION >= 40100
#include <Kokkos_BitManipulation.hpp>
#endif
#include <Kokkos_Core.hpp>

namespace ArborX::Details
{

enum class BoruvkaMode
{
  MST,
  HDBSCAN
};

constexpr int ROOT_CHAIN_VALUE = -2;
constexpr int FOLLOW_CHAIN_VALUE = -3;

class DirectedEdge
{
public:
  unsigned long long directed_edge = ULLONG_MAX;
  float weight = KokkosExt::ArithmeticTraits::infinity<float>::value;

private:
  static_assert(sizeof(unsigned long long) == 8);
  static constexpr int source_shift = 32;
  static constexpr int target_shift = 1;
  static constexpr unsigned long long mask_source =
      static_cast<unsigned long long>(UINT_MAX >> 1) << source_shift;
  static constexpr unsigned long long mask_target =
      static_cast<unsigned long long>(UINT_MAX >> 1) << target_shift;
  // clang-format off
  // | unused bit | 31 bits for the smallest of source and target | 31 bits for the largest of source and target | flag |
  // clang-format on
  static_assert((mask_source & mask_target) == 0, "implementation bug");
  // direction must be stored in the least significant bit
  static constexpr unsigned long long reverse_direction = 1;

  // performs lexicographical comparison by comparing first the weights and then
  // the unordered pair of vertices
  friend KOKKOS_FUNCTION constexpr bool operator<(DirectedEdge const &lhs,
                                                  DirectedEdge const &rhs)
  {
    return (lhs.weight != rhs.weight) ? (lhs.weight < rhs.weight)
                                      : (lhs.directed_edge < rhs.directed_edge);
  }

  KOKKOS_FUNCTION constexpr bool reverse() const
  {
    return (directed_edge & reverse_direction) == reverse_direction;
  }

public:
  KOKKOS_FUNCTION constexpr int source() const
  {
    return reverse() ? (directed_edge & mask_target) >> target_shift
                     : (directed_edge & mask_source) >> source_shift;
  }
  KOKKOS_FUNCTION constexpr int target() const
  {
    return reverse() ? (directed_edge & mask_source) >> source_shift
                     : (directed_edge & mask_target) >> target_shift;
  }
  KOKKOS_FUNCTION constexpr DirectedEdge(int source, int target, float weight)
      : directed_edge{(source < target)
                          ? (mask_source &
                             (static_cast<unsigned long long>(source)
                              << source_shift)) +
                                (mask_target &
                                 (static_cast<unsigned long long>(target)
                                  << target_shift))
                          : reverse_direction +
                                (mask_source &
                                 (static_cast<unsigned long long>(target)
                                  << source_shift)) +
                                (mask_target &
                                 (static_cast<unsigned long long>(source))
                                     << target_shift)}
      , weight{weight}
  {}
  KOKKOS_DEFAULTED_FUNCTION constexpr DirectedEdge() = default;
  KOKKOS_FUNCTION explicit constexpr operator WeightedEdge()
  {
    return {source(), target(), weight};
  }
};

template <class BVH, class Labels, class Weights, class Edges, class Metric,
          class Radii, class LowerBounds, bool UseSharedRadii>
struct FindComponentNearestNeighbors
{
  BVH _bvh;
  Labels _labels;
  Weights _weights;
  Edges _edges;
  Metric _metric;
  Radii _radii;
  LowerBounds _lower_bounds;

  struct WithLowerBounds
  {};

  template <class ExecutionSpace>
  FindComponentNearestNeighbors(ExecutionSpace const &space, BVH const &bvh,
                                Labels const &labels, Weights const &weights,
                                Edges const &edges, Metric const &metric,
                                Radii const &radii,
                                LowerBounds const &lower_bounds,
                                std::bool_constant<UseSharedRadii>)
      : _bvh(bvh)
      , _labels(labels)
      , _weights(weights)
      , _edges(edges)
      , _metric{metric}
      , _radii(radii)
      , _lower_bounds(lower_bounds)
  {
    int const n = bvh.size();
    ARBORX_ASSERT(labels.extent_int(0) == 2 * n - 1);
    ARBORX_ASSERT(edges.extent_int(0) == n);
    ARBORX_ASSERT(radii.extent_int(0) == n);

#ifdef KOKKOS_ENABLE_SERIAL
    if (std::is_same<ExecutionSpace, Kokkos::Serial>{})
    {
      Kokkos::parallel_for(
          "ArborX::MST::find_component_nearest_neighbors_with_lower_bounds",
          Kokkos::RangePolicy<ExecutionSpace, WithLowerBounds>(space, 0, n),
          *this);
    }
    else
#endif
    {
      Kokkos::parallel_for("ArborX::MST::find_component_nearest_neighbors",
                           Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
                           *this);
    }
  }

  KOKKOS_FUNCTION void operator()(WithLowerBounds, int i) const
  {
    auto const component = _labels(i);
    if (_lower_bounds(i) <= _radii(component))
    {
      this->operator()(i);
    }
  }

  KOKKOS_FUNCTION void operator()(int i) const
  {
    constexpr auto inf = KokkosExt::ArithmeticTraits::infinity<float>::value;

    auto const distance = [bounding_volume_i =
                               HappyTreeFriends::getLeafBoundingVolume(_bvh, i),
                           &bvh = _bvh](int j) {
      using Details::distance;
      auto &&bounding_volume_j =
          (HappyTreeFriends::isLeaf(bvh, j)
               ? HappyTreeFriends::getLeafBoundingVolume(bvh, j)
               : HappyTreeFriends::getInternalBoundingVolume(bvh, j));
      return distance(bounding_volume_i, bounding_volume_j);
    };

    auto const component = _labels(i);
    auto const predicate = [label_i = component, &labels = _labels](int j) {
      return label_i != labels(j);
    };
    auto const leaf_permutation_i = HappyTreeFriends::getValue(_bvh, i).index;

    DirectedEdge current_best{};

    // Use a reference for shared radii, and a copy otherwise.
    std::conditional_t<UseSharedRadii, float &, float> radius =
        _radii(component);

    constexpr int SENTINEL = -1;
    int stack[64];
    auto *stack_ptr = stack;
    *stack_ptr++ = SENTINEL;
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
    float stack_distance[64];
    auto *stack_distance_ptr = stack_distance;
    *stack_distance_ptr++ = 0;
#endif

    int node = HappyTreeFriends::getRoot(_bvh);
    float distance_node = 0;

    // Important! The truncation radius is computed using the provided metric,
    // rather than just assigning the Euclidean distance. This only works for
    // metrics that return a value greater or equal to Euclidean distance
    // (e.g., mutual reachability metric). Metrics that do not satisfy this
    // criteria may return wrong results.
    do
    {
      bool traverse_left = false;
      bool traverse_right = false;

      int left_child;
      int right_child;
      float distance_left = inf;
      float distance_right = inf;

      // Note it is <= instead of < when comparing with radius here and below.
      // The reason is that in Boruvka it matters which of the equidistant
      // points we take so that they don't create a cycle among component
      // connectivity. This requires us to uniquely resolve equidistant
      // neighbors, so we cannot skip any of them.
      if (distance_node <= radius)
      {
        // Insert children into the stack and make sure that the closest one
        // ends on top.
        left_child = HappyTreeFriends::getLeftChild(_bvh, node);
        right_child = HappyTreeFriends::getRightChild(_bvh, node);
        distance_left = distance(left_child);
        distance_right = distance(right_child);

        if (predicate(left_child) && distance_left <= radius)
        {
          if (HappyTreeFriends::isLeaf(_bvh, left_child))
          {
            float const candidate_dist =
                _metric(leaf_permutation_i,
                        HappyTreeFriends::getValue(_bvh, left_child).index,
                        distance_left);
            DirectedEdge const candidate_edge{i, left_child, candidate_dist};
            if (candidate_edge < current_best)
            {
              current_best = candidate_edge;
              if constexpr (UseSharedRadii)
                Kokkos::atomic_min(&radius, candidate_dist);
              else
                radius = candidate_dist;
            }
          }
          else
          {
            traverse_left = true;
          }
        }

        // Note: radius may have been already updated here from the left child
        if (predicate(right_child) && distance_right <= radius)
        {
          if (HappyTreeFriends::isLeaf(_bvh, right_child))
          {
            float const candidate_dist =
                _metric(leaf_permutation_i,
                        HappyTreeFriends::getValue(_bvh, right_child).index,
                        distance_right);
            DirectedEdge const candidate_edge{i, right_child, candidate_dist};
            if (candidate_edge < current_best)
            {
              current_best = candidate_edge;
              if constexpr (UseSharedRadii)
                Kokkos::atomic_min(&radius, candidate_dist);
              else
                radius = candidate_dist;
            }
          }
          else
          {
            traverse_right = true;
          }
        }
      }

      if (!traverse_left && !traverse_right)
      {
        node = *--stack_ptr;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        if (node != SENTINEL)
        {
          // This is a theoretically unnecessary duplication of distance
          // calculation for stack nodes. However, for Cuda it's better than
          // putting the distances in stack.
          distance_node = distance(node);
        }
#else
        distance_node = *--stack_distance_ptr;
#endif
      }
      else
      {
        node = (traverse_left &&
                (distance_left <= distance_right || !traverse_right))
                   ? left_child
                   : right_child;
        distance_node = (node == left_child ? distance_left : distance_right);
        if (traverse_left && traverse_right)
        {
          *stack_ptr++ = (node == left_child ? right_child : left_child);
#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
          *stack_distance_ptr++ =
              (node == left_child ? distance_right : distance_left);
#endif
        }
      }
    } while (node != SENTINEL);

    // This check is only here to reduce hammering the atomics for large
    // components. Otherwise, for a large number of points and a small number of
    // components it becomes extremely expensive.
    auto &component_weight = _weights(component);
    if (current_best.weight < inf && current_best.weight <= component_weight)
    {
      if (Kokkos::atomic_min_fetch(&component_weight, current_best.weight) ==
          current_best.weight)
      {
        _edges(i) = current_best;
      }
    }
  }
};

// For every component C, find the shortest edge (v, w) such that v is in C
// and w is not in C. The found edge is stored in component_out_edges(C).
template <class ExecutionSpace, class BVH, class Labels, class Weights,
          class Edges, class Metric, class Radii, class LowerBounds,
          bool UseSharedRadii>
FindComponentNearestNeighbors(ExecutionSpace, BVH, Labels, Weights, Edges,
                              Metric, Radii, LowerBounds,
                              std::bool_constant<UseSharedRadii>)
    -> FindComponentNearestNeighbors<BVH, Labels, Weights, Edges, Metric, Radii,
                                     LowerBounds, UseSharedRadii>;

template <class ExecutionSpace, class Labels, class ComponentOutEdges,
          class LowerBounds>
void updateLowerBounds(ExecutionSpace const &space, Labels const &labels,
                       ComponentOutEdges const &component_out_edges,
                       LowerBounds lower_bounds)
{
  auto const n = lower_bounds.extent(0);
  Kokkos::parallel_for(
      "ArborX::MST::update_lower_bounds",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        using KokkosExt::max;
        auto component = labels(i);
        auto const &edge = component_out_edges(component);
        lower_bounds(i) = max(lower_bounds(i), edge.weight);
      });
}

// workaround slow atomic min operations on edge type
template <class ExecutionSpace, class Labels, class Weights, class Edges>
void retrieveEdges(ExecutionSpace const &space, Labels const &labels,
                   Weights const &weights, Edges const &edges)
{
  auto const n = weights.extent(0);
  Kokkos::parallel_for(
      "ArborX::MST::reset_component_edges",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        auto const component = labels(i);
        if (i != component)
          return;
        auto const component_weight = weights(component);
        auto &component_edge = edges(component);
        // replace stale values by neutral element for min reduction
        if (component_edge.weight != component_weight)
        {
          component_edge = {};
          component_edge.weight = component_weight;
        }
      });
  Kokkos::parallel_for(
      "ArborX::MST::reduce_component_edges",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        auto const component = labels(i);
        auto const component_weight = weights(component);
        auto const &edge = edges(i);
        if (edge.weight == component_weight)
        {
          auto &component_edge = edges(component);
          Kokkos::atomic_min(&(component_edge.directed_edge),
                             edge.directed_edge);
        }
      });
}

struct LabelsTag
{};
struct UnidirectionalEdgesTag
{};
struct BidirectionalEdgesTag
{};

template <class Labels, class OutEdges, class Edges, class EdgesMapping,
          class EdgesCount, BoruvkaMode Mode>
struct UpdateComponentsAndEdges
{
  Labels _labels;
  OutEdges _out_edges;
  Edges _edges;
  EdgesMapping _edge_mapping;
  EdgesCount _num_edges;

  KOKKOS_FUNCTION auto computeNextComponent(int component) const
  {
    int next_component = _labels(_out_edges(component).target());
    int next_next_component = _labels(_out_edges(next_component).target());

    if (next_next_component != component)
    {
      // The component's edge is unidirectional
      return next_component;
    }
    // The component's edge is bidirectional, uniquely resolve the bidirectional
    // edge
    return KokkosExt::min(component, next_component);
  }

  KOKKOS_FUNCTION auto computeFinalComponent(int component) const
  {
    int prev_component = component;
    int next_component;
    while ((next_component = computeNextComponent(prev_component)) !=
           prev_component)
      prev_component = next_component;

    return next_component;
  }

  KOKKOS_FUNCTION void operator()(LabelsTag, int i) const
  {
    auto const component = _labels(i);
    auto const final_component = computeFinalComponent(component);
    _labels(i) = final_component;
  }

  KOKKOS_FUNCTION void operator()(UnidirectionalEdgesTag, int i) const
  {
    auto const component = _labels(i);
    if (i != component || computeNextComponent(component) == component)
      return;

    // append new edge at the "end" of the array (akin to
    // std::vector::push_back)
    auto const edge = static_cast<WeightedEdge>(_out_edges(i));
    auto const back =
        Kokkos::atomic_fetch_inc(&_num_edges()); // atomic post-increment
    _edges(back) = edge;

    if constexpr (Mode == BoruvkaMode::HDBSCAN)
      _edge_mapping(i) = back;
  }

  KOKKOS_FUNCTION void operator()(BidirectionalEdgesTag, int i) const
  {
    auto const component = _labels(i);
    if (i != component || computeNextComponent(component) != component)
      return;

    auto const &edge = _out_edges(i);
    _edge_mapping(i) = _edge_mapping(_labels(edge.target()));
  }
};

// Reverse node leaf permutation order back to original indices
template <class ExecutionSpace, class BVH, class Edges>
void finalizeEdges(ExecutionSpace const &space, BVH const &bvh,
                   Edges const &edges)
{
  int const n = bvh.size();
  ARBORX_ASSERT(edges.extent_int(0) == n - 1);
  Kokkos::parallel_for(
      "ArborX::MST::finalize_edges",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n - 1),
      KOKKOS_LAMBDA(int i) {
        edges(i).source =
            HappyTreeFriends::getValue(bvh, edges(i).source).index;
        edges(i).target =
            HappyTreeFriends::getValue(bvh, edges(i).target).index;
      });
}

template <class ExecutionSpace, class Labels, class Edges, class EdgesMapping,
          class SidedParents>
void updateSidedParents(ExecutionSpace const &space, Labels const &labels,
                        Edges const &edges, EdgesMapping const &edges_mapping,
                        SidedParents &sided_parents, int edges_start,
                        int edges_end)
{
  KokkosExt::ScopedProfileRegion guard("ArborX::MST::update_sided_parents");

  // Same as dendrogram alpha's standalone "updateSidedParents"
  Kokkos::parallel_for(
      "ArborX::MST::update_sided_parents",
      Kokkos::RangePolicy<ExecutionSpace>(space, edges_start, edges_end),
      KOKKOS_LAMBDA(int e) {
        auto const &edge = edges(e);

        // As the edge is within the same alpha vertex, labels of its vertices
        // are the same, so can take either
        int component = labels(edge.source);

        int const alpha_edge_index = edges_mapping(component);

        auto const &alpha_edge = edges(alpha_edge_index);

        if (edge < alpha_edge)
        {
          bool is_left_side = (labels(alpha_edge.source) == component);
          sided_parents(e) =
              2 * alpha_edge_index + static_cast<int>(is_left_side);
        }
        else
        {
          sided_parents(e) = FOLLOW_CHAIN_VALUE - alpha_edge_index;
        }
      });
}

template <class ExecutionSpace, class Labels, class OutEdges,
          class EdgesMapping, class BVH, class Parents>
void assignVertexParents(ExecutionSpace const &space, Labels const &labels,
                         OutEdges const &out_edges,
                         EdgesMapping const &edges_mapping, BVH const &bvh,
                         Parents parents)
{
  auto const n = edges_mapping.extent_int(0) + 1;
  int const vertices_offset = n - 1;

  Kokkos::parallel_for(
      "ArborX::MST::compute_vertex_parents",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int e) {
        auto const &edge = out_edges(e);

        int i = labels(edge.source());
        parents(HappyTreeFriends::getValue(bvh, i).index + vertices_offset) =
            edges_mapping(i);
      });
}

template <typename ExecutionSpace, typename Edges, typename SidedParents,
          typename Parents>
void computeParents(ExecutionSpace const &space, Edges const &edges,
                    SidedParents const &sided_parents, Parents &parents)
{
  KokkosExt::ScopedProfileRegion guard("ArborX::MST::compute_edge_parents");

  using MemorySpace = typename SidedParents::memory_space;

  auto num_edges = edges.size();

  // Encode both a sided parent and an edge weight into long long.
  // This way, once we sort based on this value, edges with the same sided
  // parent will already be sorted in increasing order.
  // The main reason for using long long values is the performance when
  // compared with sorting pairs. The second reason is that Kokkos's BinSort
  // does not support custom comparison operators.
  static_assert(sizeof(long long) >= sizeof(int) + sizeof(float));
  Kokkos::View<long long *, MemorySpace> keys(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::MST::keys"),
      num_edges);

  constexpr int shift = sizeof(int) * CHAR_BIT;

  Kokkos::parallel_for(
      "ArborX::MST::compute_sided_alpha_parents",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_edges),
      KOKKOS_LAMBDA(int const e) {
        long long key = sided_parents(e);
        auto const &edge = edges(e);
        if (key <= FOLLOW_CHAIN_VALUE)
        {
          int next = FOLLOW_CHAIN_VALUE - key;
          do
          {
            key = sided_parents(next);
            if (key <= FOLLOW_CHAIN_VALUE)
              next = FOLLOW_CHAIN_VALUE - key;
            else if (key >= 0)
            {
              next = key / 2;
              auto const &next_edge = edges(next);
              if (edge < next_edge)
                break;
            }
            else if (key == ROOT_CHAIN_VALUE)
              break;
          } while (true);
        }
        if (key == ROOT_CHAIN_VALUE)
          key = INT_MAX;

        // Comparison of weights as ints is the same as their comparison as
        // floats as long as they are positive and are not NaNs or inf
        static_assert(sizeof(int) == sizeof(float));
        keys(e) = (key << shift) +
#if KOKKOS_VERSION >= 40100
                  Kokkos::bit_cast<int>(edge.weight);
#else
                  reinterpret_cast<int const &>(edge.weight);
#endif
      });

  auto permute = sortObjects(space, keys);

  Kokkos::parallel_for(
      "ArborX::MST::compute_parents",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, num_edges),
      KOKKOS_LAMBDA(int const i) {
        int e = permute(i);
        if (i == (int)num_edges - 1)
        {
          // The parent of the root node is set to -1
          parents(e) = -1;
        }
        else if ((keys(i) >> shift) == (keys(i + 1) >> shift))
        {
          // For the edges belonging to the same chain, assign the parent of an
          // edge to the edge with the next larger value
          parents(e) = permute(i + 1);
        }
        else
        {
          // For an edge which points to the root of a chain, assign edge's
          // parent to be that root
          parents(e) = (keys(i) >> shift) / 2;
        }
      });
}

// Compute upper bound on the shortest edge of each component.
template <class ExecutionSpace, class BVH, class Labels, class Metric,
          class Radii>
void resetSharedRadii(ExecutionSpace const &space, BVH const &bvh,
                      Labels const &labels, Metric const &metric,
                      Radii const &radii)
{
  //  We will search for the shortest outgoing edge of a component. The better
  //  we initialize the upper bound on the distance (i.e., the smaller it is),
  //  the less traversal we will do and the faster it will be.
  //
  // Here, we use the knowledge that it is a self-collision problem. In other
  // words, we only have a single point cloud. We further use the fact that if
  // we sort predicates based on the Morton codes, it will match the order of
  // primitives (or be close enough, as some points with the same Morton codes
  // may be in a different order due to the unstable sort that we use). Thus, if
  // we take an index of a query, we assume that it matches the corresponding
  // primitive. If a label of that primitive is different from a label of its
  // neighbor (which is in fact its Morton neighbor), we compute the distance
  // between the two. The upper bound for a component is set to the minimum
  // distance between all such pairs. Given that the Morton neighbors are
  // typically close to each other, this should provide a reasonably low bound.
  auto const n = bvh.size();
  Kokkos::parallel_for(
      "ArborX::MST::reset_shared_radii",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n - 1),
      KOKKOS_LAMBDA(int i) {
        int const j = i + 1;
        auto const label_i = labels(i);
        auto const label_j = labels(j);
        if (label_i != label_j)
        {
          auto const r =
              metric(HappyTreeFriends::getValue(bvh, i).index,
                     HappyTreeFriends::getValue(bvh, j).index,
                     distance(HappyTreeFriends::getLeafBoundingVolume(bvh, i),
                              HappyTreeFriends::getLeafBoundingVolume(bvh, j)));
          Kokkos::atomic_min(&radii(label_i), r);
          Kokkos::atomic_min(&radii(label_j), r);
        }
      });
}

template <class MemorySpace, BoruvkaMode Mode = BoruvkaMode::MST>
struct MinimumSpanningTree
{
  Kokkos::View<WeightedEdge *, MemorySpace> edges;
  Kokkos::View<int *, MemorySpace> dendrogram_parents;
  Kokkos::View<float *, MemorySpace> dendrogram_parent_heights;

  template <class ExecutionSpace, class Primitives>
  MinimumSpanningTree(ExecutionSpace const &space, Primitives const &primitives,
                      int k = 1)
      : edges(Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                                 "ArborX::MST::edges"),
              AccessTraits<Primitives, PrimitivesTag>::size(primitives) - 1)
      , dendrogram_parents("ArborX::MST::dendrogram_parents", 0)
      , dendrogram_parent_heights("ArborX::MST::dendrogram_parent_heights", 0)
  {
    Kokkos::Profiling::pushRegion("ArborX::MST::MST");

    using Access = AccessTraits<Primitives, PrimitivesTag>;
    constexpr int dim = GeometryTraits::dimension_v<
        typename Details::AccessTraitsHelper<Access>::type>;
    using Box = ExperimentalHyperGeometry::Box<dim>;

    auto const n = AccessTraits<Primitives, PrimitivesTag>::size(primitives);

    Kokkos::Profiling::pushRegion("ArborX::MST::construction");
    BasicBoundingVolumeHierarchy<MemorySpace, Box> bvh(space, primitives);
    Kokkos::Profiling::popRegion();

    if (k > 1)
    {
      Kokkos::Profiling::pushRegion("ArborX::MST::compute_core_distances");
      Kokkos::View<float *, MemorySpace> core_distances(
          "ArborX::MST::core_distances", n);
      bvh.query(space, NearestK<Primitives>{primitives, k},
                MaxDistance<Primitives, decltype(core_distances)>{
                    primitives, core_distances});
      Kokkos::Profiling::popRegion();

      MutualReachability<decltype(core_distances)> mutual_reachability{
          core_distances};
      Kokkos::Profiling::pushRegion("ArborX::MST::boruvka");
      doBoruvka(space, bvh, mutual_reachability);
      Kokkos::Profiling::popRegion();
    }
    else
    {
      Kokkos::Profiling::pushRegion("ArborX::MST::boruvka");
      doBoruvka(space, bvh, Euclidean{});
      Kokkos::Profiling::popRegion();
    }

    finalizeEdges(space, bvh, edges);

    Kokkos::Profiling::popRegion();
  }

  // enclosing function for an extended __host__ __device__ lambda cannot have
  // private or protected access within its class
#ifndef KOKKOS_COMPILER_NVCC
private:
#endif
  template <class ExecutionSpace, class BVH, class Metric>
  void doBoruvka(ExecutionSpace const &space, BVH const &bvh,
                 Metric const &metric)
  {
    auto const n = bvh.size();
    Kokkos::View<int *, MemorySpace> tree_parents(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::MST::tree_parents"),
        2 * n - 1);
    findParents(space, bvh, tree_parents);

    Kokkos::Profiling::pushRegion("ArborX::MST::initialize_node_labels");
    Kokkos::View<int *, MemorySpace> labels(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::MST::labels"),
        2 * n - 1);
    iota(space, Kokkos::subview(labels, std::make_pair((decltype(n))0, n)));
    Kokkos::Profiling::popRegion();

    Kokkos::View<DirectedEdge *, MemorySpace> component_out_edges(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::MST::component_out_edges"),
        n);

    Kokkos::View<float *, MemorySpace> weights(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::MST::weights"),
        n);

    Kokkos::View<float *, MemorySpace> radii(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::MST::radii"),
        n);

    Kokkos::View<float *, MemorySpace> lower_bounds("ArborX::MST::lower_bounds",
                                                    0);

    constexpr bool use_lower_bounds =
#ifdef KOKKOS_ENABLE_SERIAL
        std::is_same<ExecutionSpace, Kokkos::Serial>::value;
#else
        false;
#endif

    // Shared radii may or may not be faster for CUDA depending on the problem.
    // In the ICPP'51 paper experiments, we ended up using it only in Serial.
    // But we would like to keep an option open for the future, so the code is
    // written to be able to run it if we want.
    constexpr bool use_shared_radii =
#ifdef KOKKOS_ENABLE_SERIAL
        std::is_same<ExecutionSpace, Kokkos::Serial>::value;
#else
        false;
#endif

    if constexpr (use_lower_bounds)
    {
      KokkosExt::reallocWithoutInitializing(space, lower_bounds, n);
      Kokkos::deep_copy(space, lower_bounds, 0);
    }

    Kokkos::Profiling::pushRegion("ArborX::MST::Boruvka_loop");
    Kokkos::View<int, MemorySpace> num_edges(
        Kokkos::view_alloc(space, "ArborX::MST::num_edges")); // initialize to 0

    Kokkos::View<int *, MemorySpace> edges_mapping("ArborX::MST::edges_mapping",
                                                   0);

    Kokkos::View<int *, MemorySpace> sided_parents("ArborX::MST::sided_parents",
                                                   0);
    if constexpr (Mode == BoruvkaMode::HDBSCAN)
    {
      KokkosExt::reallocWithoutInitializing(space, edges_mapping, n - 1);
      KokkosExt::reallocWithoutInitializing(space, sided_parents, n - 1);
      KokkosExt::reallocWithoutInitializing(space, dendrogram_parents,
                                            2 * n - 1);
    }

    // Boruvka iterations
    int iterations = 0;
    int num_components = n;
    [[maybe_unused]] int edges_start = 0;
    [[maybe_unused]] int edges_end = 0;
    do
    {
      Kokkos::Profiling::pushRegion("ArborX::Boruvka_" +
                                    std::to_string(++iterations) + "_" +
                                    std::to_string(num_components));

      // Propagate leaf node labels to internal nodes
      reduceLabels(space, tree_parents, labels);

      constexpr auto inf = KokkosExt::ArithmeticTraits::infinity<float>::value;
      constexpr DirectedEdge uninitialized_edge;
      Kokkos::deep_copy(space, component_out_edges, uninitialized_edge);
      Kokkos::deep_copy(space, weights, inf);
      Kokkos::deep_copy(space, radii, inf);
      resetSharedRadii(space, bvh, labels, metric, radii);

      FindComponentNearestNeighbors(
          space, bvh, labels, weights, component_out_edges, metric, radii,
          lower_bounds, std::bool_constant<use_shared_radii>());
      retrieveEdges(space, labels, weights, component_out_edges);
      if constexpr (use_lower_bounds)
      {
        updateLowerBounds(space, labels, component_out_edges, lower_bounds);
      }

      UpdateComponentsAndEdges<decltype(labels), decltype(component_out_edges),
                               decltype(edges), decltype(edges_mapping),
                               decltype(num_edges), Mode>
          f{labels, component_out_edges, edges, edges_mapping, num_edges};

      // For every component C and a found shortest edge `(u, w)`, add the
      // edge to the list of MST edges.
      Kokkos::parallel_for(
          "ArborX::MST::update_unidirectional_edges",
          Kokkos::RangePolicy<ExecutionSpace, UnidirectionalEdgesTag>(space, 0,
                                                                      n),
          f);

      int num_edges_host;
      Kokkos::deep_copy(space, num_edges_host, num_edges);
      space.fence();

      if constexpr (Mode == BoruvkaMode::HDBSCAN)
      {
        Kokkos::parallel_for(
            "ArborX::MST::update_bidirectional_edges",
            Kokkos::RangePolicy<ExecutionSpace, BidirectionalEdgesTag>(space, 0,
                                                                       n),
            f);

        if (iterations > 1)
          updateSidedParents(space, labels, edges, edges_mapping, sided_parents,
                             edges_start, edges_end);
        else
        {
          KokkosExt::ScopedProfileRegion guard(
              "ArborX::MST::compute_vertex_parents");
          assignVertexParents(space, labels, component_out_edges, edges_mapping,
                              bvh, dendrogram_parents);
        }
      }

      // For every component C and a found shortest edge `(u, w)`, merge C
      // with the component that w belongs to by updating the labels
      Kokkos::parallel_for(
          "ArborX::MST::update_labels",
          Kokkos::RangePolicy<ExecutionSpace, LabelsTag>(space, 0, n), f);

      num_components = static_cast<int>(n) - num_edges_host;

      edges_start = edges_end;
      edges_end = num_edges_host;

      Kokkos::Profiling::popRegion();
    } while (num_components > 1);

    // Deallocate some memory to reduce high water mark
    Kokkos::resize(edges_mapping, 0);
    Kokkos::resize(lower_bounds, 0);
    Kokkos::resize(radii, 0);
    Kokkos::resize(labels, 0);
    Kokkos::resize(weights, 0);
    Kokkos::resize(component_out_edges, 0);
    Kokkos::resize(tree_parents, 0);

    if constexpr (Mode == BoruvkaMode::HDBSCAN)
    {

      // Done with the recursion as there are no more alpha edges. Assign
      // all current edges to the root chain.
      Kokkos::deep_copy(space,
                        Kokkos::subview(sided_parents,
                                        std::make_pair(edges_start, edges_end)),
                        ROOT_CHAIN_VALUE);

      computeParents(space, edges, sided_parents, dendrogram_parents);

      KokkosExt::reallocWithoutInitializing(space, dendrogram_parent_heights,
                                            n - 1);
      Kokkos::parallel_for(
          "ArborX::MST::assign_dendrogram_parent_heights",
          Kokkos::RangePolicy<ExecutionSpace>(space, 0, n - 1),
          ARBORX_CLASS_LAMBDA(int const e) {
            dendrogram_parent_heights(e) = edges(e).weight;
          });
    }

    Kokkos::Profiling::popRegion();
  }
};

} // namespace ArborX::Details

#endif
