/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#ifndef DTK_FINE_SEARCH_DEF_HPP
#define DTK_FINE_SEARCH_DEF_HPP

#include <DTK_DBC.hpp>
#include <DTK_FineSearchFunctor.hpp>
#include <Kokkos_Core.hpp>

namespace DataTransferKit
{
template <typename DeviceType>
void FineSearch<DeviceType>::search(
    Kokkos::View<Coordinate **, DeviceType> reference_points,
    Kokkos::View<bool *, DeviceType> point_in_cell,
    Kokkos::View<Coordinate **, DeviceType> physical_points,
    Kokkos::View<Coordinate ***, DeviceType> cells,
    Kokkos::View<unsigned int *, DeviceType> coarse_search_output_cells,
    shards::CellTopology cell_topo )
{
    // Check the size of the Views
    DTK_REQUIRE( reference_points.extent( 0 ) == point_in_cell.extent( 0 ) );
    DTK_REQUIRE( reference_points.extent( 0 ) == physical_points.extent( 0 ) );
    DTK_REQUIRE( reference_points.extent( 1 ) == physical_points.extent( 1 ) );
    DTK_REQUIRE( reference_points.extent( 1 ) == cells.extent( 2 ) );

    using ExecutionSpace = typename DeviceType::execution_space;
    int const n_ref_pts = reference_points.extent( 0 );

    // Perform the fine search. We hide the template parameters used by
    // Intrepid2, using the CellType template.
    // Note that if the Newton solver does not converge, Intrepid2 will just
    // return the last results and there is no way to know that the coordinates
    // in the reference frames where not found.
    unsigned int const cell_topo_key = cell_topo.getKey();
    if ( cell_topo_key ==
         shards::getCellTopologyData<shards::Hexahedron<8>>()->key )
    {
        Functor::FineSearch<CellType::Hexahedron_8, DeviceType> search_functor(
            threshold, reference_points, point_in_cell, physical_points, cells,
            coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_hex_8" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );
    }
    else if ( cell_topo_key ==
              shards::getCellTopologyData<shards::Hexahedron<27>>()->key )
    {
        Functor::FineSearch<CellType::Hexahedron_27, DeviceType> search_functor(
            threshold, reference_points, point_in_cell, physical_points, cells,
            coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_hex_27" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );
    }
    else if ( cell_topo_key ==
              shards::getCellTopologyData<shards::Pyramid<5>>()->key )
    {
        Functor::FineSearch<CellType::Pyramid_5, DeviceType> search_functor(
            threshold, reference_points, point_in_cell, physical_points, cells,
            coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_pyr_5" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );
    }
    else if ( cell_topo_key ==
              shards::getCellTopologyData<shards::Quadrilateral<4>>()->key )
    {
        Functor::FineSearch<CellType::Quadrilateral_4, DeviceType>
            search_functor( threshold, reference_points, point_in_cell,
                            physical_points, cells,
                            coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_quad_4" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );
    }
    else if ( cell_topo_key ==
              shards::getCellTopologyData<shards::Quadrilateral<9>>()->key )
    {
        Functor::FineSearch<CellType::Quadrilateral_9, DeviceType>
            search_functor( threshold, reference_points, point_in_cell,
                            physical_points, cells,
                            coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_quad_9" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );
    }
    else if ( cell_topo_key ==
              shards::getCellTopologyData<shards::Tetrahedron<4>>()->key )
    {
        Functor::FineSearch<CellType::Tetrahedron_4, DeviceType> search_functor(
            threshold, reference_points, point_in_cell, physical_points, cells,
            coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_tet_4" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );
    }
    else if ( cell_topo_key ==
              shards::getCellTopologyData<shards::Tetrahedron<10>>()->key )
    {
        Functor::FineSearch<CellType::Tetrahedron_10, DeviceType>
            search_functor( threshold, reference_points, point_in_cell,
                            physical_points, cells,
                            coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_tet_10" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );
    }
    else if ( cell_topo_key ==
              shards::getCellTopologyData<shards::Triangle<3>>()->key )
    {
        Functor::FineSearch<CellType::Triangle_3, DeviceType> search_functor(
            threshold, reference_points, point_in_cell, physical_points, cells,
            coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_tri_3" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );
    }
    else if ( cell_topo_key ==
              shards::getCellTopologyData<shards::Triangle<6>>()->key )
    {
        Functor::FineSearch<CellType::Triangle_6, DeviceType> search_functor(
            threshold, reference_points, point_in_cell, physical_points, cells,
            coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_tri_6" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );
    }
    else if ( cell_topo_key ==
              shards::getCellTopologyData<shards::Wedge<6>>()->key )
    {
        Functor::FineSearch<CellType::Wedge_6, DeviceType> search_functor(
            threshold, reference_points, point_in_cell, physical_points, cells,
            coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_wedge_6" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );
    }
    else if ( cell_topo_key ==
              shards::getCellTopologyData<shards::Wedge<18>>()->key )
    {
        Functor::FineSearch<CellType::Wedge_18, DeviceType> search_functor(
            threshold, reference_points, point_in_cell, physical_points, cells,
            coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_wedge_18" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );
    }
    else
    {
        throw std::runtime_error( "Not implemented" );
    }
    Kokkos::fence();
}
}

// Explicit instantiation macro
#define DTK_FINESEARCH_INSTANT( NODE )                                         \
    template class FineSearch<typename NODE::device_type>;

#endif
