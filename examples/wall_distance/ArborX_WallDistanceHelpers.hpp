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

#ifndef ARBORX_WALL_DISTANCE_HELPERS_HPP
#define ARBORX_WALL_DISTANCE_HELPERS_HPP

#include <ArborX_Segment.hpp>
#include <ArborX_Triangle.hpp>
#include <detail/ArborX_AccessTraits.hpp>
#include <detail/ArborX_Predicates.hpp>

#include <Kokkos_DynRankView.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include <numeric> // exclusive_scan
#include <vector>

#include <Panzer_STK_Interface.hpp>

namespace ArborX::Details
{

template <int DIM, typename Sides>
struct Geometries
{
  static_assert(DIM == 2 || DIM == 3);
  int _topology_key;
  Sides _sides;
};

} // namespace ArborX::Details

template <int DIM, typename Sides>
struct ArborX::AccessTraits<ArborX::Details::Geometries<DIM, Sides>>
{
  using Self = ArborX::Details::Geometries<DIM, Sides>;
  using memory_space = typename Sides::memory_space;

  KOKKOS_FUNCTION static auto size(Self const &self)
  {
    if (self._topology_key == shards::Hexahedron<8>::key)
      return 2 * self._sides.extent(0);
    else
      return self._sides.extent(0);
  }

  KOKKOS_FUNCTION static auto get(Self const &self, size_t i)
  {
    using namespace ArborX::Details;

    auto const &sides = self._sides;
    if constexpr (DIM == 2)
    {
      return ArborX::Experimental::Segment{{sides(i, 0, 0), sides(i, 0, 1)},
                                           {sides(i, 1, 0), sides(i, 1, 1)}};
    }
    else
    {
      if (self._topology_key == shards::Tetrahedron<4>::key)
      {
        return ArborX::Triangle{
            {sides(i, 0, 0), sides(i, 0, 1), sides(i, 0, 2)},
            {sides(i, 1, 0), sides(i, 1, 1), sides(i, 1, 2)},
            {sides(i, 2, 0), sides(i, 2, 1), sides(i, 2, 2)}};
      }

      KOKKOS_ASSERT(self._topology_key == shards::Hexahedron<8>::key);

      // Split quad side into two triangles
      bool const odd = (i % 2);
      int j = (odd ? 1 : 2);
      int k = (odd ? 2 : 3);
      i /= 2;
      return ArborX::Triangle{{sides(i, 0, 0), sides(i, 0, 1), sides(i, 0, 2)},
                              {sides(i, j, 0), sides(i, j, 1), sides(i, j, 2)},
                              {sides(i, k, 0), sides(i, k, 1), sides(i, k, 2)}};
    }
  }
};

namespace ArborX::Details
{

template <typename View>
inline auto create_layout_right_mirror_view_no_init(View const &src)
{
  static_assert(View::rank == 3);

  using HostMemorySpace = typename View::traits::host_mirror_space;

  constexpr bool has_compatible_layout =
      (std::is_same_v<typename View::array_layout, Kokkos::LayoutRight> ||
       (View::rank == 1 &&
        (std::is_same_v<typename View::array_layout, Kokkos::LayoutLeft> ||
         std::is_same_v<typename View::array_layout, Kokkos::LayoutRight>)));
  constexpr bool has_compatible_memory_space =
      std::is_same_v<typename View::memory_space, HostMemorySpace>;

  if constexpr (has_compatible_layout && has_compatible_memory_space)
  {
    return src;
  }
  else
  {
    typename HostMemorySpace::execution_space space;
    auto mirror_view = Kokkos::View<typename View::traits::data_type,
                                    Kokkos::LayoutRight, HostMemorySpace>(
        Kokkos::view_alloc(
            space, HostMemorySpace{}, Kokkos::WithoutInitializing,
            std::string(src.label()).append("_layout_right_mirror")),
        src.extent(0), src.extent(1), src.extent(2));
    space.fence();
    return mirror_view;
  }
}

template <typename LocalSides>
void getLocalSides(panzer_stk::STK_Interface const &mesh,
                   std::vector<std::string> const &wall_names,
                   LocalSides &local_sides)
{
  // Get sideset names of sidesets
  std::vector<std::string> sideset_block_names;
  mesh.getSidesetNames(sideset_block_names);
  KOKKOS_ASSERT(!sideset_block_names.empty());

  // Get the local set of sides declared as walls from all block/sideset
  // combinations.
  std::vector<stk::mesh::Entity> local_side_entities;
  for (auto const &wall_name : wall_names)
  {
    if (std::find(sideset_block_names.begin(), sideset_block_names.end(),
                  wall_name) == sideset_block_names.end())
    {
      throw std::runtime_error("Sideset " + wall_name + " not found in mesh");
    }

    std::vector<stk::mesh::Entity> sideset_sides;
    mesh.getMySides(wall_name, sideset_sides);
    local_side_entities.insert(local_side_entities.end(), sideset_sides.begin(),
                               sideset_sides.end());
  }

  mesh.getElementVertices(local_side_entities, local_sides);
}

template <typename ExecutionSpace, typename LocalSides, typename GlobalSides>
static void gatherGlobalSides(MPI_Comm comm, ExecutionSpace const &space,
                              LocalSides const &local_sides,
                              GlobalSides &global_sides)
{
  Kokkos::Profiling::ScopedRegion guard(
      "ArborX::WallDistance::gatherGlobalSides");

  using MemorySpace = typename GlobalSides::memory_space;

  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  int num_local = local_sides.extent(0);
  int extent1 = local_sides.extent(1);
  int extent2 = local_sides.extent(2);

  // Some ranks may contain no data, which will result in data_size being 0. As
  // we need all ranks the same knowledge of data_size, we inform all ranks on
  // the correct value.
  for (int i = 1; i <= 2; ++i)
  {
    auto &extent = (i == 1) ? extent1 : extent2;

    int max_extent = 0;
    MPI_Allreduce(&extent, &max_extent, 1, MPI_INT, MPI_MAX, comm);
    KOKKOS_ASSERT(extent == 0 || extent == max_extent);
    extent = max_extent;
  }
  auto data_size = extent1 * extent2;

  // Compose gather communication pattern.
  std::vector<int> global_counts(comm_size, 0);
  global_counts[comm_rank] = num_local;
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                static_cast<void *>(global_counts.data()), 1, MPI_INT, comm);

  std::vector<int> offsets(comm_size + 1, 0);
  std::exclusive_scan(global_counts.begin(), global_counts.end(),
                      offsets.begin(), 0);
  offsets[comm_size] = offsets[comm_size - 1] + global_counts.back();
  int num_global_sides = offsets.back();

  Kokkos::resize(Kokkos::view_alloc(Kokkos::WithoutInitializing), global_sides,
                 num_global_sides, extent1, extent2);

  // Create host-side mirror for sides
  // Have to be careful with layouts
  auto local_sides_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, local_sides);
  Kokkos::fence();
  auto global_sides_host =
      create_layout_right_mirror_view_no_init(global_sides);

  auto const offset_rank = offsets[comm_rank];
  for (int i = 0; i < num_local; ++i)
    for (int j = 0; j < extent1; ++j)
      for (int k = 0; k < extent2; ++k)
        global_sides_host(offset_rank + i, j, k) = local_sides_host(i, j, k);

  for (int rank = 0; rank < comm_size; ++rank)
  {
    offsets[rank] *= data_size;
    global_counts[rank] *= data_size;
  }

  // FIXME: hardcoded to MPI_DOUBLE
  MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, global_sides_host.data(),
                 global_counts.data(), offsets.data(), MPI_DOUBLE, comm);

  // For multi-dimensional views, we need to first copy into a separate
  // storage because of a different layout
  auto tmp_view = Kokkos::create_mirror_view_and_copy(
      Kokkos::view_alloc(space, MemorySpace{}), global_sides_host);
  Kokkos::deep_copy(space, global_sides, tmp_view);
  space.fence();
}

// Check that the topologies in all element blocks are the same, and are ones
// from the list and return the key
static int get_topology_key(panzer_stk::STK_Interface const &mesh)
{
  std::vector<int> accepted_topologies = {
      shards::Tetrahedron<4>::key, shards::Hexahedron<8>::key,
      shards::Triangle<3>::key, shards::Quadrilateral<4>::key};

  std::vector<std::string> elem_block_names;
  mesh.getElementBlockNames(elem_block_names);

  auto equal_topologies = [&mesh](std::string const &a, std::string const &b) {
    return mesh.getCellTopology(a)->getKey() ==
           mesh.getCellTopology(b)->getKey();
  };

  if (std::adjacent_find(elem_block_names.begin(), elem_block_names.end(),
                         equal_topologies) != elem_block_names.end())
    throw std::runtime_error("Different topologies in element blocks");

  auto key = mesh.getCellTopology(elem_block_names[0])->getKey();
  if (std::find(accepted_topologies.begin(), accepted_topologies.end(), key) ==
      accepted_topologies.end())
    throw std::runtime_error(
        "Block topology is not Tet4, Hex8, Tri3, or Quad4");
  return key;
}

struct WallDistanceCallback
{
  template <typename Query, typename Value, typename Output>
  KOKKOS_FUNCTION void operator()(Query const &query, Value const &value,
                                  Output const &output) const
  {
    output(distance(getGeometry(query), value));
  }
};

} // namespace ArborX::Details

#endif
