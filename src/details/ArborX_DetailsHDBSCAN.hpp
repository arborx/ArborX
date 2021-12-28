/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DETAILS_HDBSCAN_HPP
#define ARBORX_DETAILS_HDBSCAN_HPP

#include <ArborX_DetailsMutualReachabilityDistance.hpp>
#include <ArborX_DetailsTreeNodeLabeling.hpp>
#include <ArborX_LinearBVH.hpp>

namespace ArborX
{

namespace Details
{

// Boruvka's algorithm is guaranteed to converge only if all edges have unique
// weights. As the distances between pairs of points may not be unique, point
// indices are used to resolve edges uniquely.
KOKKOS_INLINE_FUNCTION
bool compareEdgesLess(int i1, int j1, float d1, int i2, int j2, float d2)
{
  using KokkosExt::max;
  using KokkosExt::min;

  if (d1 < d2)
    return true;
  if (d1 > d2)
    return false;

  auto vmin1 = min(i1, j1);
  auto vmin2 = min(i2, j2);
  if (vmin1 < vmin2)
    return true;
  if (vmin1 > vmin2)
    return false;

  auto vmax1 = max(i1, j1);
  auto vmax2 = max(i2, j2);
  if (vmax1 < vmax2)
    return true;
  return false;
}

struct EdgeTriple
{
  int i = -1;
  int j = -1;
  float d = KokkosExt::ArithmeticTraits::infinity<float>::value;
};

KOKKOS_FUNCTION inline bool operator<(EdgeTriple const &edge1,
                                      EdgeTriple const &edge2)
{
  if (compareEdgesLess(edge1.i, edge1.j, edge1.d, edge2.i, edge2.j, edge2.d))
    return true;
  return false;
}

template <typename ExecutionSpace, typename Predicates, typename BVH,
          typename BVHParents, typename Labels, typename CoreDistances,
          typename ComponentOutEdges>
void determineComponentEdges(ExecutionSpace const &exec_space,
                             Predicates const &predicates, BVH const &bvh,
                             BVHParents const &bvh_parents,
                             Labels const &labels,
                             CoreDistances const &core_distances,
                             ComponentOutEdges components_out_edges)
{
  Kokkos::Profiling::pushRegion("ArborX::HDBSCAN::determine_component_edges");

  using MemorySpace = typename Labels::memory_space;
  using Access = AccessTraits<Predicates, PredicatesTag>;

  auto const n = Access::size(predicates);

  ARBORX_ASSERT(labels.size() == n);

  auto NoInit = [](std::string const &label) {
    return Kokkos::view_alloc(Kokkos::WithoutInitializing, label);
  };

  Kokkos::View<int *, MemorySpace> bvh_nodes_labels(
      NoInit("ArborX::HDBSCAN::labels"), 2 * n - 1);

  constexpr int undetermined = -1;
  constexpr auto infinity = KokkosExt::ArithmeticTraits::infinity<float>::value;

  // Initialize leaf node labels
  // Cannot simply copy into subview, as we also apply permutations
  initLabels(exec_space, bvh, labels, bvh_nodes_labels);

  // Propagate leaf node labels to internal nodes
  reduceLabels(exec_space, bvh_parents, bvh_nodes_labels);

  // TODO: is there a smarter way to initialize radii?
  Kokkos::View<float *, MemorySpace> radii(NoInit("ArborX::HDBSCAN::radii"), n);
  Kokkos::deep_copy(exec_space, radii, infinity);

  ArborX::Details::MutualReachability<CoreDistances> const dmreach{
      core_distances};

  // Compared to the standard kNN algorithm, this one uses mutual reachability
  // distance to define closeness, and shares the value of the radius among all
  // points in the component.
  Kokkos::View<EdgeTriple *, MemorySpace> closest_component_edges(
      "ArborX::HDBSCAN::closest_component_dist", n);
  Kokkos::parallel_for(
      "ArborX::HDBSCAN::find_nearest_neighbors",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int ii) {
        using Details::HappyTreeFriends;

        auto const &nearest_predicate = Access::get(predicates, ii);

        auto const i = getData(nearest_predicate);

        auto const distance = [geometry = getGeometry(nearest_predicate),
                               &bvh](int j) {
          using Details::distance;
          return distance(geometry,
                          HappyTreeFriends::getBoundingVolume(bvh, j));
        };
        int component = labels(i);
        auto const predicate = [component, &bvh_nodes_labels](int j) {
          return bvh_nodes_labels(j) != component;
        };

        int nearest = undetermined;
        float nearest_dist = infinity;

        auto &radius = radii(component);

        constexpr int SENTINEL = -1;
        int stack[64];
        auto *stack_ptr = stack;
        *stack_ptr++ = SENTINEL;
#if !defined(__CUDA_ARCH__)
        float stack_distance[64];
        auto *stack_distance_ptr = stack_distance;
        *stack_distance_ptr++ = 0.f;
#endif

        int node = HappyTreeFriends::getRoot(bvh);
        int left_child;
        int right_child;

        float distance_left = 0.f;
        float distance_right = 0.f;
        float distance_node = 0.f;

        // Important! The truncation radius is set to mutual reachability
        // distance, rather than Euclidean distance. This works because mutual
        // reachability distance is not less than Euclidean. Thus, nodes
        // truncated by Euclidean would have mutual reachability distance
        // larger anyway.
        do
        {
          bool traverse_left = false;
          bool traverse_right = false;

          // Note it is <= instead of < when comparing with radius here and
          // below. The reason is that in Boruvka it matters which of the
          // equidistant points we take so that they don't create a cycle among
          // component connectivity. This requires us to uniquely resolve
          // equidistant neighbors, so we cannot skip any of them.
          if (distance_node <= radius)
          {
            // Insert children into the stack and make sure that the
            // closest one ends on top.
            left_child = HappyTreeFriends::getLeftChild(bvh, node);
            right_child = HappyTreeFriends::getRightChild(bvh, node);

            distance_left = distance(left_child);
            distance_right = distance(right_child);

            if (predicate(left_child) && distance_left <= radius)
            {
              if (HappyTreeFriends::isLeaf(bvh, left_child))
              {
                int candidate =
                    HappyTreeFriends::getLeafPermutationIndex(bvh, left_child);
                float candidate_dist = dmreach(i, candidate, distance_left);
                if (compareEdgesLess(i, candidate, candidate_dist, i, nearest,
                                     nearest_dist))
                {
                  nearest = candidate;
                  nearest_dist = candidate_dist;

                  float t = radius;
                  while (true)
                  {
                    float r = Kokkos::atomic_compare_exchange(&radius, t,
                                                              nearest_dist);
                    if (r == t ||         // this thread did the exchange
                        r < nearest_dist) // another thread did the exchange
                                          // with a smaller number
                      break;
                    t = r;
                  }
                }
              }
              else
              {
                traverse_left = true;
              }
            }

            // Note: radius may have been already updated here from the left
            // child
            if (predicate(right_child) && distance_right <= radius)
            {
              if (HappyTreeFriends::isLeaf(bvh, right_child))
              {
                int candidate =
                    HappyTreeFriends::getLeafPermutationIndex(bvh, right_child);
                float candidate_dist = dmreach(i, candidate, distance_right);
                if (compareEdgesLess(i, candidate, candidate_dist, i, nearest,
                                     nearest_dist))
                {
                  nearest = candidate;
                  nearest_dist = candidate_dist;

                  float t = radius;
                  while (true)
                  {
                    float r = Kokkos::atomic_compare_exchange(&radius, t,
                                                              nearest_dist);
                    if (r == t ||         // this thread did the exchange
                        r < nearest_dist) // another thread did the exchange
                                          // with a smaller number
                      break;
                    t = r;
                  }
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
#if defined(__CUDA_ARCH__)
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
            distance_node =
                (node == left_child ? distance_left : distance_right);
            if (traverse_left && traverse_right)
            {
              *stack_ptr++ = (node == left_child ? right_child : left_child);
#if !defined(__CUDA_ARCH__)
              *stack_distance_ptr++ =
                  (node == left_child ? distance_right : distance_left);
#endif
            }
          }
        } while (node != SENTINEL);

        auto &current_edge = closest_component_edges(component);

        // This check is only here to reduce hammering the atomics for large
        // components. Otherwise, for a large number of points and a small
        // number of components it becomes extremely expensive.
        if (compareEdgesLess(i, nearest, nearest_dist, current_edge.i,
                             current_edge.j, current_edge.d))
          Kokkos::atomic_min(&current_edge,
                             EdgeTriple{(int)i, nearest, nearest_dist});
      });

  Kokkos::parallel_for("ArborX::HDBSCAN::copy_component_edge",
                       Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                       KOKKOS_LAMBDA(int const i) {
                         int const component = labels(i);

                         // Copy only by a single thread in the component
                         if (component == i)
                         {
                           components_out_edges(component) = {
                               closest_component_edges(component).i,
                               closest_component_edges(component).j};
                         }
                       });

  Kokkos::Profiling::popRegion();
}

template <typename ExecutionSpace, typename ComponentOutEdges, typename Labels,
          typename MST>
int updateLabels(ExecutionSpace const &exec_space, int num_components_old,
                 ComponentOutEdges const &components_out_edges, Labels labels,
                 MST mst_edges)
{
  using MemorySpace = typename Labels::memory_space;

  Kokkos::Profiling::pushRegion("ArborX::HDBSCAN::update_labels");

  auto const n = labels.size();

  Kokkos::View<int, MemorySpace> num_edges("ArborX::HDBSCAN::num_edges");
  Kokkos::deep_copy(exec_space, num_edges, n - num_components_old);

  Kokkos::View<int, MemorySpace> num_components(
      "ArborX::HDBSCAN::num_components");
  Kokkos::deep_copy(num_components, 0);
  Kokkos::parallel_for(
      "ArborX::HDBSCAN::update_components",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int const i) {
        // Each Boruvka iterations creates connects several components
        // together. From a graph perspective, we first compute an edge for
        // each component leading to a component it will be connected to. The
        // edges are directed. Following these edges, one eventually encounters
        // two components pointing to each other. We choose the component with
        // a minimum index of those two as the component label for all
        // components leading to it. We call that component "final component".
        auto computeNext = [&labels, &components_out_edges](int component) {
          int next_component = labels(components_out_edges(component).second);
          int next_next_component =
              labels(components_out_edges(next_component).second);

          if (next_next_component != component)
          {
            // The component's edge is unidirectional
            return next_component;
          }
          // The component's edge is bidirectional, uniquely resolve the
          // bidirectional edge
          return KokkosExt::min(component, next_component);
        };

        int component = labels(i);
        int prev_component = component;
        int next_component;
        while ((next_component = computeNext(prev_component)) != prev_component)
          prev_component = next_component;

        int &final_component = next_component;
        labels(i) = final_component;

        bool const is_component_representative = (i == component);
        if (!is_component_representative)
          return;

        // Now, only a single thread for each component is present.

        bool const is_terminal_component = (final_component == component);
        if (is_terminal_component)
        {
          // Multiple components get merged into a single one. The counter
          // needs to be updated once, thus it is updated by the
          // representative of the final component.
          Kokkos::atomic_increment(&num_components());
        }
        else
        {
          // All non-final components store their outgoing edges as part of
          // MST. This will also include the edge leading to the final
          // component.
          int edge = Kokkos::atomic_fetch_add(&num_edges(), 1);
          mst_edges(edge) = components_out_edges(component);
        }
      });
  auto num_components_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, num_components);

  Kokkos::Profiling::popRegion();

  return num_components_host();
}

} // namespace Details
} // namespace ArborX

#endif
