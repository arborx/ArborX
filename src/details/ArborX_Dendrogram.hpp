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
#ifndef ARBORX_DENDROGRAM_HPP
#define ARBORX_DENDROGRAM_HPP

#include <ArborX_DetailsDendrogram.hpp>
#include <ArborX_DetailsKokkosExtScopedProfileRegion.hpp>
#include <ArborX_DetailsKokkosExtSort.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_DetailsWeightedEdge.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX::Experimental
{

enum class DendrogramImplementation
{
  ALPHA,
  BORUVKA,
  UNION_FIND
};

template <typename MemorySpace>
struct Dendrogram
{
  Kokkos::View<int *, MemorySpace> _parents;
  Kokkos::View<float *, MemorySpace> _parent_heights;

  Dendrogram(Kokkos::View<int *, MemorySpace> parents,
             Kokkos::View<float *, MemorySpace> parent_heights)
      : _parents(parents)
      , _parent_heights(parent_heights)
  {}

  template <typename ExecutionSpace>
  Dendrogram(ExecutionSpace const &exec_space,
             Kokkos::View<Details::WeightedEdge *, MemorySpace> edges,
             DendrogramImplementation impl = DendrogramImplementation::ALPHA)
      : _parents("ArborX::Dendrogram::parents", 0)
      , _parent_heights("ArborX::Dendrogram::parent_heights", 0)
  {
    KokkosExt::ScopedProfileRegion guard("ArborX::Dendrogram::Dendrogram");

    auto const num_edges = edges.size();
    auto const num_vertices = num_edges + 1;

    KokkosExt::reallocWithoutInitializing(exec_space, _parents,
                                          num_edges + num_vertices);
    KokkosExt::reallocWithoutInitializing(exec_space, _parent_heights,
                                          num_edges);

    Kokkos::View<Details::UnweightedEdge *, MemorySpace> unweighted_edges(
        Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                           "ArborX::Dendrogram::unweighted_edges"),
        num_edges);
    splitEdges(exec_space, edges, unweighted_edges, _parent_heights);

    Kokkos::Profiling::pushRegion("ArborX::Dendrogram::sort_edges");
    KokkosExt::sortByKey(exec_space, _parent_heights, unweighted_edges);
    Kokkos::Profiling::popRegion();

    using ConstEdges =
        Kokkos::View<Details::UnweightedEdge const *, MemorySpace>;
    switch (impl)
    {
    case DendrogramImplementation::UNION_FIND:
      Details::dendrogramUnionFind(exec_space, ConstEdges(unweighted_edges),
                                   _parents);
      break;

    case DendrogramImplementation::ALPHA:
      dendrogramAlpha(exec_space, ConstEdges(unweighted_edges), _parents);
      break;

    default:
      Kokkos::abort("ArborX implementation bug");
    }
  }

  template <typename ExecutionSpace>
  void splitEdges(
      ExecutionSpace const &exec_space,
      Kokkos::View<Details::WeightedEdge *, MemorySpace> edges,
      Kokkos::View<Details::UnweightedEdge *, MemorySpace> unweighted_edges,
      Kokkos::View<float *, MemorySpace> weights)
  {
    Kokkos::parallel_for(
        "ArborX::Dendrogram::copy_weights_and_edges",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, edges.size()),
        KOKKOS_LAMBDA(int const e) {
          weights(e) = edges(e).weight;
          unweighted_edges(e) = {edges(e).source, edges(e).target};
        });
  }

  template <typename ExecutionSpace>
  void dendrogramAlpha(
      ExecutionSpace const &exec_space,
      Kokkos::View<Details::UnweightedEdge const *, MemorySpace> edges,
      Kokkos::View<int *, MemorySpace> &parents)
  {
    KokkosExt::ScopedProfileRegion guard(
        "ArborX::Dendrogram::dendrogram_alpha");

    auto const num_global_edges = edges.size();

    using Details::ALPHA_FOLLOW_OFFSET;
    using Details::ALPHA_ROOT_CHAIN;

    Kokkos::View<int *, MemorySpace> sided_level_parents(
        Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                           "ArborX::Dendrogram::sided_parents"),
        edges.size());
    Kokkos::deep_copy(exec_space, sided_level_parents, ALPHA_FOLLOW_OFFSET);

    Kokkos::View<int *, MemorySpace> global_map(
        Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                           "ArborX::Dendrogram::global_map"),
        num_global_edges);
    iota(exec_space, global_map);

    int level = 0;
    do
    {
      KokkosExt::ScopedProfileRegion level_guard("ArborX::Dendrogram::level_" +
                                                 std::to_string(level));
      int num_edges = edges.size();

      // Step 1: find alpha edges of the current MST
      Kokkos::Profiling::pushRegion("ArborX::Dendrogram::find_alpha_edges");
      auto smallest_vertex_incident_edges =
          Details::findSmallestVertexIncidentEdges(exec_space, edges);
      auto alpha_edge_indices = Details::findAlphaEdges(
          exec_space, edges, smallest_vertex_incident_edges);
      Kokkos::Profiling::popRegion();

      if (level == 0)
      {
        // Step 6: build vertex parents
        KokkosExt::ScopedProfileRegion guard(
            "ArborX::Dendrogram::compute_vertex_parents");
        assignVertexParents(exec_space, edges, smallest_vertex_incident_edges,
                            parents);
      }

      int num_alpha_edges = alpha_edge_indices.size();
      if (num_alpha_edges == 0)
      {
        // Done with the recursion as there are no more alpha edges. Assign all
        // current edges to the root chain.
        Kokkos::parallel_for(
            "ArborX::Dendrogram::assign_remaining_side_parents",
            Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_edges),
            KOKKOS_LAMBDA(int const e) {
              sided_level_parents(global_map(e)) = ALPHA_ROOT_CHAIN;
            });
        break;
      }

      auto const largest_alpha_index =
          KokkosExt::lastElement(exec_space, alpha_edge_indices);

      // Step 2: construct virtual alpha-vertices
      auto alpha_vertices =
          Details::assignAlphaVertices(exec_space, edges, alpha_edge_indices);

      // Step 3: build alpha incidence matrix
      Kokkos::View<int *, MemorySpace> alpha_mat_offsets(
          "ArborX::Dendrogram::alpha_mat_offsets", 0);
      Kokkos::View<int *, MemorySpace> alpha_mat_edges(
          "ArborX::Dendrogram::alpha_mat_edges", 0);
      Details::buildAlphaIncidenceMatrix(exec_space, edges, alpha_edge_indices,
                                         alpha_vertices, alpha_mat_offsets,
                                         alpha_mat_edges);

      // Step 4: update sided parents
      Details::updateSidedParents(
          exec_space, edges, largest_alpha_index, alpha_vertices,
          alpha_mat_offsets, alpha_mat_edges, global_map, sided_level_parents);

      Kokkos::resize(alpha_mat_offsets, 0); // deallocate
      Kokkos::resize(alpha_mat_edges, 0);   // deallocate

      // Step 5: compress edges
      Kokkos::Profiling::pushRegion("ArborX::Dendrogram::compress");
      auto compressed_edges = Details::buildAlphaMST(
          exec_space, edges, alpha_edge_indices, alpha_vertices);
      auto compressed_global_map = Details::compressGlobalMap(
          exec_space, global_map, alpha_edge_indices);
      Kokkos::Profiling::popRegion();

      // Prepare for the next iteration
      global_map = compressed_global_map;
      edges = compressed_edges;

      ++level;

    } while (true);

    // Step 6: build edge parents
    auto edge_parents =
        Kokkos::subview(parents, Kokkos::make_pair(0, (int)num_global_edges));
    Details::computeParents(exec_space, sided_level_parents, edge_parents);
  }
};

} // namespace ArborX::Experimental

#endif
