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
#include <ArborX_DetailsKokkosExtScopedProfileRegion.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_DetailsMinimumSpanningTree.hpp>
#include <ArborX_DetailsMutualReachabilityDistance.hpp>
#include <ArborX_DetailsTreeNodeLabeling.hpp>
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_DetailsWeightedEdge.hpp>
#include <ArborX_LinearBVH.hpp>

namespace ArborX::Details
{

template <class MemorySpace, BoruvkaMode Mode = BoruvkaMode::MST>
struct MinimumSpanningTree
{
  using memory_space = MemorySpace;
  static_assert(Kokkos::is_memory_space<MemorySpace>::value);

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
    using Point = typename AccessTraitsHelper<Access>::type;

    auto const n = AccessTraits<Primitives, PrimitivesTag>::size(primitives);

    Kokkos::Profiling::pushRegion("ArborX::MST::construction");
    BasicBoundingVolumeHierarchy<MemorySpace, PairIndexVolume<Point>> bvh(
        space, Details::LegacyValues<Primitives, Point>{primitives});
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
          KOKKOS_CLASS_LAMBDA(int const e) {
            dendrogram_parent_heights(e) = edges(e).weight;
          });
    }

    Kokkos::Profiling::popRegion();
  }
};

} // namespace ArborX::Details

#endif
