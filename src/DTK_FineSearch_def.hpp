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
    Kokkos::View<unsigned int *, DeviceType> coarse_search_output_points,
    Kokkos::View<unsigned int *, DeviceType> coarse_search_output_cells,
    shards::CellTopology cell_topo )
{
    // Check the size of the Views
    DTK_REQUIRE( reference_points.extent( 0 ) == point_in_cell.extent( 0 ) );
    DTK_REQUIRE( reference_points.extent( 1 ) == physical_points.extent( 1 ) );
    DTK_REQUIRE( reference_points.extent( 1 ) == cells.extent( 2 ) );
    DTK_REQUIRE( coarse_search_output_cells.extent( 0 ) ==
                 coarse_search_output_points.extent( 0 ) );

    using ExecutionSpace = typename DeviceType::execution_space;
    int const n_ref_pts = reference_points.extent( 0 );

    shards::Hexahedron<8> constexpr hex_8;
    shards::Hexahedron<27> constexpr hex_27;
    shards::Pyramid<5> constexpr pyr_5;
    shards::Quadrilateral<4> constexpr quad_4;
    shards::Quadrilateral<9> constexpr quad_9;
    shards::Tetrahedron<4> constexpr tet_4;
    shards::Tetrahedron<10> constexpr tet_10;
    shards::Triangle<3> constexpr tri_3;
    shards::Triangle<6> constexpr tri_6;
    shards::Wedge<6> constexpr wedge_6;
    shards::Wedge<18> constexpr wedge_18;

    // Perform the fine search. We hide the template parameters used by
    // Intrepid2, using the CellType template.
    switch ( cell_topo.getKey() )
    {
    case hex_8.key:
    {
        Functor::FineSearch<CellType::Hexahedron_8, DeviceType> search_functor(
            reference_points, point_in_cell, physical_points, cells,
            coarse_search_output_points, coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_hex_8" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );

        break;
    }
    case hex_27.key:
    {
        Functor::FineSearch<CellType::Hexahedron_27, DeviceType> search_functor(
            reference_points, point_in_cell, physical_points, cells,
            coarse_search_output_points, coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_hex_27" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );

        break;
    }
    case pyr_5.key:
    {
        Functor::FineSearch<CellType::Pyramid_5, DeviceType> search_functor(
            reference_points, point_in_cell, physical_points, cells,
            coarse_search_output_points, coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_pyr_5" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );

        break;
    }
    case quad_4.key:
    {
        Functor::FineSearch<CellType::Quadrilateral_4, DeviceType>
            search_functor( reference_points, point_in_cell, physical_points,
                            cells, coarse_search_output_points,
                            coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_quad_4" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );

        break;
    }
    case quad_9.key:
    {
        Functor::FineSearch<CellType::Quadrilateral_9, DeviceType>
            search_functor( reference_points, point_in_cell, physical_points,
                            cells, coarse_search_output_points,
                            coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_quad_9" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );

        break;
    }
    case tet_4.key:
    {
        Functor::FineSearch<CellType::Tetrahedron_4, DeviceType> search_functor(
            reference_points, point_in_cell, physical_points, cells,
            coarse_search_output_points, coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_tet_4" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );

        break;
    }
    case tet_10.key:
    {
        Functor::FineSearch<CellType::Tetrahedron_10, DeviceType>
            search_functor( reference_points, point_in_cell, physical_points,
                            cells, coarse_search_output_points,
                            coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_tet_10" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );

        break;
    }
    case tri_3.key:
    {
        Functor::FineSearch<CellType::Triangle_3, DeviceType> search_functor(
            reference_points, point_in_cell, physical_points, cells,
            coarse_search_output_points, coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_tri_3" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );

        break;
    }
    case tri_6.key:
    {
        Functor::FineSearch<CellType::Triangle_6, DeviceType> search_functor(
            reference_points, point_in_cell, physical_points, cells,
            coarse_search_output_points, coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_tri_6" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );

        break;
    }
    case wedge_6.key:
    {
        Functor::FineSearch<CellType::Wedge_6, DeviceType> search_functor(
            reference_points, point_in_cell, physical_points, cells,
            coarse_search_output_points, coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_wedge_6" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );

        break;
    }
    case wedge_18.key:
    {
        Functor::FineSearch<CellType::Wedge_18, DeviceType> search_functor(
            reference_points, point_in_cell, physical_points, cells,
            coarse_search_output_points, coarse_search_output_cells );
        Kokkos::parallel_for(
            REGION_NAME( "compute_pos_in_ref_space_wedge_18" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_ref_pts ),
            search_functor );

        break;
    }
    default:
    {
        throw std::runtime_error( "Not implemented" );
    }
    }
    Kokkos::fence();

    // Get ride of bogus warning about variables not being used with gcc 7.1
    std::ignore = hex_8;
    std::ignore = hex_27;
    std::ignore = pyr_5;
    std::ignore = quad_4;
    std::ignore = quad_9;
    std::ignore = tet_4;
    std::ignore = tet_10;
    std::ignore = tri_3;
    std::ignore = tri_6;
    std::ignore = wedge_6;
    std::ignore = wedge_18;
}
}

// Explicit instantiation macro
#define DTK_FINESEARCH_INSTANT( NODE )                                         \
    template class FineSearch<typename NODE::device_type>;

#endif
