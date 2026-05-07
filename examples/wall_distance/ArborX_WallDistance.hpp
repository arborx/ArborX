/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_WALL_DISTANCE_HPP
#define ARBORX_WALL_DISTANCE_HPP

#include "ArborX_WallDistanceHelpers.hpp"
#include <ArborX_DistributedTree.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Segment.hpp>

#include <Kokkos_DynRankView.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <string>

#include <Panzer_IntegrationRule.hpp>
#include <Panzer_STK_Interface.hpp>
#include <Panzer_Workset.hpp>
#include <Panzer_Workset_Utilities.hpp> // getIntegrationRuleIndex

namespace ArborX::Experimental
{

template <typename MemorySpace, int DIM, typename Coordinate = double,
          bool ReplicateSides = true>
class WallDistance
{
  static_assert(DIM == 2 || DIM == 3, "ArborX::Experimental::WallDistance is "
                                      "only implemented for 2D and 3D meshes");

  using Geometry = std::conditional_t<DIM == 2, Segment<DIM, Coordinate>,
                                      Triangle<DIM, Coordinate>>;
  using Index =
      std::conditional_t<ReplicateSides,
                         BoundingVolumeHierarchy<MemorySpace, Geometry>,
                         DistributedTree<MemorySpace, Geometry>>;

  Index _index;

public:
  KOKKOS_DEFAULTED_FUNCTION
  WallDistance() = default;

  template <typename ExecutionSpace>
  WallDistance(ExecutionSpace const &space,
               panzer_stk::STK_Interface const &mesh,
               std::vector<std::string> const &wall_names);

  template <typename ExecutionSpace, typename WorksetDistances>
  void distance(ExecutionSpace const &space,
                std::vector<panzer::Workset> const &worksets,
                panzer::IntegrationRule const &ir,
                WorksetDistances &workset_distances);

  template <typename ExecutionSpace, typename Points, typename Distances>
  void distance(ExecutionSpace const &space, Points const &points,
                Distances &distances);
};

template <typename MemorySpace, int DIM, typename Coordinate,
          bool ReplicateSides>
template <typename ExecutionSpace>
WallDistance<MemorySpace, DIM, Coordinate, ReplicateSides>::WallDistance(
    ExecutionSpace const &space, panzer_stk::STK_Interface const &mesh,
    std::vector<std::string> const &wall_names)
{
  std::string prefix = "ArborX::WallDistance::WallDistance";
  Kokkos::Profiling::ScopedRegion guard(prefix);
  prefix += "::";

  Kokkos::DynRankView<Coordinate, MemorySpace> local_sides;
  Details::getLocalSides(mesh, wall_names, local_sides);

  auto key = Details::get_topology_key(mesh);

  MPI_Comm comm = Teuchos::getRawMpiComm(*mesh.getComm());
  if constexpr (ReplicateSides)
  {
    Kokkos::View<Coordinate ***, MemorySpace> global_sides(
        prefix + "global_sides", 0, 0, 0);
    Details::gatherGlobalSides(comm, space, local_sides, global_sides);

    _index = BoundingVolumeHierarchy(
        space,
        Details::Geometries<DIM, decltype(global_sides)>{key, global_sides});
  }
  else
  {
    _index = DistributedTree(
        comm, space,
        Details::Geometries<DIM, decltype(local_sides)>{key, local_sides});
  }
}

template <typename MemorySpace, int DIM, typename Coordinate,
          bool ReplicateSides>
template <typename ExecutionSpace, typename Points, typename Distances>
void WallDistance<MemorySpace, DIM, Coordinate, ReplicateSides>::distance(
    ExecutionSpace const &space, Points const &points, Distances &distances)
{
  std::string prefix = "ArborX::WallDistance::distance";
  Kokkos::Profiling::ScopedRegion guard(prefix);
  prefix += "::";

  auto queries = ArborX::Experimental::make_nearest(points, 1);

  Kokkos::View<int *, MemorySpace> offset(prefix + "offset", 0);
  if constexpr (ReplicateSides)
    _index.query(space, queries, Details::WallDistanceCallback{}, distances,
                 offset);
  else
    _index.query(space, queries,
                 declare_callback_constrained(Details::WallDistanceCallback{}),
                 distances, offset);
}

template <typename MemorySpace, int DIM, typename Coordinate,
          bool ReplicateSides>
template <typename ExecutionSpace, typename WorksetDistances>
void WallDistance<MemorySpace, DIM, Coordinate, ReplicateSides>::distance(
    ExecutionSpace const &space, std::vector<panzer::Workset> const &worksets,
    panzer::IntegrationRule const &ir, WorksetDistances &workset_distances)
{
  std::string prefix = "ArborX::WallDistance::workset_distance";
  Kokkos::Profiling::ScopedRegion guard(prefix);
  prefix += "::";

  // FIXME: do we require that workset_distances is already allocated with the
  // right size?
  int const num_worksets = worksets.size();
  int const num_int_points_per_cell = workset_distances.extent(2);

  KOKKOS_ASSERT((int)workset_distances.extent(0) == num_worksets);
  KOKKOS_ASSERT((int)workset_distances.extent(2) == num_int_points_per_cell);

  auto ir_index = panzer::getIntegrationRuleIndex(ir.order(), worksets[0]);

  int num_queries = 0;
  std::vector<int> workset_sizes(num_worksets);
  for (int workset_id = 0; workset_id < num_worksets; ++workset_id)
  {
    auto const &workset = worksets[workset_id];
    auto const num_cells = workset.num_cells;
    workset_sizes[workset_id] = num_cells;
    num_queries += num_cells * num_int_points_per_cell;
  }

  using Point = ArborX::Point<DIM, Coordinate>;
  Kokkos::View<Point *, MemorySpace> points(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing, prefix + "points"),
      num_queries);
  for (int workset_id = 0, queries_offset = 0; workset_id < num_worksets;
       ++workset_id)
  {
    auto const &workset = worksets[workset_id];
    auto const num_cells = workset.num_cells;

    if (num_cells == 0)
      continue;

    auto const &ip_coords = workset.int_rules[ir_index]->ip_coordinates;

    Kokkos::parallel_for(
        prefix + "create_queries", Kokkos::RangePolicy(space, 0, num_cells),
        KOKKOS_LAMBDA(int cell) {
          auto offset = queries_offset + cell * num_int_points_per_cell;
          Point p;
          for (int int_point = 0; int_point < num_int_points_per_cell;
               ++int_point)
          {
            for (int d = 0; d < DIM; ++d)
              p[d] = ip_coords(cell, int_point, d);
            points(offset++) = p;
          }
        });
    queries_offset += num_cells * num_int_points_per_cell;
  }

  Kokkos::View<Coordinate *, MemorySpace> distances(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         prefix + "distances"),
      0);
  distance(space, points, distances);

  for (int workset_id = 0, workset_offset = 0; workset_id < num_worksets;
       ++workset_id)
  {
    auto const num_cells = workset_sizes[workset_id];
    if (num_cells == 0)
      continue;

    Kokkos::parallel_for(
        "ArborX::WallDistance::reshape_distances",
        Kokkos::RangePolicy(space, 0, num_cells), KOKKOS_LAMBDA(int cell) {
          auto offset = workset_offset + cell * num_int_points_per_cell;
          for (int int_point = 0; int_point < num_int_points_per_cell;
               ++int_point)
            workset_distances(workset_id, cell, int_point) =
                distances(offset++);
        });
    workset_offset += num_cells * num_int_points_per_cell;
  }
}

} // namespace ArborX::Experimental

#endif
