/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#include <DTK_FineSearch.hpp>

#include <Shards_CellTopologyManagedData.hpp>
#include <Teuchos_UnitTestHarness.hpp>

// We only test shards::Hexahedron<8> and shards::Quadrilateral<4> Testing all
// the topologies would require a lot of code (need to create a bunch of meshes)
// and the only difference in the search is the template parameters in the
// Functor.

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( FineSearch, hex_8, DeviceType )
{
    unsigned int constexpr dim = 3;
    shards::CellTopology cell_topology(
        shards::getCellTopologyData<shards::Hexahedron<8>>() );
    unsigned int constexpr n_ref_pts = 5;

    Kokkos::View<double * [dim], DeviceType> reference_points( "ref_pts",
                                                               n_ref_pts );
    Kokkos::View<bool *, DeviceType> point_in_cell( "pt_in_cell", n_ref_pts );
    // Physical points are (1.5, 0.5, 0.3) and (2.5, 0.3, 0.5)
    Kokkos::View<double * [dim], DeviceType> physical_points( "phys_pts", 2 );
    physical_points( 0, 0 ) = 1.5;
    physical_points( 0, 1 ) = 0.5;
    physical_points( 0, 2 ) = 0.3;
    physical_points( 1, 0 ) = 2.5;
    physical_points( 1, 1 ) = 0.3;
    physical_points( 1, 2 ) = 0.5;
    // Vertices of the cells
    Kokkos::View<double * * [dim], DeviceType> cells( "cell_nodes", 3, 8 );
    // First cell
    cells( 0, 0, 0 ) = 0.;
    cells( 0, 0, 1 ) = 0.;
    cells( 0, 0, 2 ) = 0.;
    cells( 0, 1, 0 ) = 1.;
    cells( 0, 1, 1 ) = 0.;
    cells( 0, 1, 2 ) = 0.;
    cells( 0, 2, 0 ) = 1.;
    cells( 0, 2, 1 ) = 1.;
    cells( 0, 2, 2 ) = 0.;
    cells( 0, 3, 0 ) = 0.;
    cells( 0, 3, 1 ) = 1.;
    cells( 0, 3, 2 ) = 0.;
    cells( 0, 4, 0 ) = 0.;
    cells( 0, 4, 1 ) = 0.;
    cells( 0, 4, 2 ) = 1.;
    cells( 0, 5, 0 ) = 1.;
    cells( 0, 5, 1 ) = 0.;
    cells( 0, 5, 2 ) = 1.;
    cells( 0, 6, 0 ) = 1.;
    cells( 0, 6, 1 ) = 1.;
    cells( 0, 6, 2 ) = 1.;
    cells( 0, 7, 0 ) = 0.;
    cells( 0, 7, 1 ) = 1.;
    cells( 0, 7, 2 ) = 1.;
    // Second cell
    cells( 1, 0, 0 ) = 1.;
    cells( 1, 0, 1 ) = 0.;
    cells( 1, 0, 2 ) = 0.;
    cells( 1, 1, 0 ) = 2.;
    cells( 1, 1, 1 ) = 0.;
    cells( 1, 1, 2 ) = 0.;
    cells( 1, 2, 0 ) = 2.;
    cells( 1, 2, 1 ) = 1.;
    cells( 1, 2, 2 ) = 0.;
    cells( 1, 3, 0 ) = 1.;
    cells( 1, 3, 1 ) = 1.;
    cells( 1, 3, 2 ) = 0.;
    cells( 1, 4, 0 ) = 1.;
    cells( 1, 4, 1 ) = 0.;
    cells( 1, 4, 2 ) = 1.;
    cells( 1, 5, 0 ) = 2.;
    cells( 1, 5, 1 ) = 0.;
    cells( 1, 5, 2 ) = 1.;
    cells( 1, 6, 0 ) = 2.;
    cells( 1, 6, 1 ) = 1.;
    cells( 1, 6, 2 ) = 1.;
    cells( 1, 7, 0 ) = 1.;
    cells( 1, 7, 1 ) = 1.;
    cells( 1, 7, 2 ) = 1.;
    // Third cell
    cells( 2, 0, 0 ) = 2.;
    cells( 2, 0, 1 ) = 0.;
    cells( 2, 0, 2 ) = 0.;
    cells( 2, 1, 0 ) = 3.;
    cells( 2, 1, 1 ) = 0.;
    cells( 2, 1, 2 ) = 0.;
    cells( 2, 2, 0 ) = 3.;
    cells( 2, 2, 1 ) = 1.;
    cells( 2, 2, 2 ) = 0.;
    cells( 2, 3, 0 ) = 2.;
    cells( 2, 3, 1 ) = 1.;
    cells( 2, 3, 2 ) = 0.;
    cells( 2, 4, 0 ) = 2.;
    cells( 2, 4, 1 ) = 0.;
    cells( 2, 4, 2 ) = 1.;
    cells( 2, 5, 0 ) = 3.;
    cells( 2, 5, 1 ) = 0.;
    cells( 2, 5, 2 ) = 1.;
    cells( 2, 6, 0 ) = 3.;
    cells( 2, 6, 1 ) = 1.;
    cells( 2, 6, 2 ) = 1.;
    cells( 2, 7, 0 ) = 2.;
    cells( 2, 7, 1 ) = 1.;
    cells( 2, 7, 2 ) = 1.;
    // Coarse search output: points
    Kokkos::View<unsigned int *, DeviceType> coarse_srch_pts( "coarse_srch_pts",
                                                              5 );
    coarse_srch_pts( 0 ) = 0;
    coarse_srch_pts( 1 ) = 0;
    coarse_srch_pts( 2 ) = 0;
    coarse_srch_pts( 3 ) = 1;
    coarse_srch_pts( 4 ) = 1;
    // Coarse search output: cells
    Kokkos::View<unsigned int *, DeviceType> coarse_srch_cells(
        "coarse_srch_cells", 5 );
    coarse_srch_cells( 0 ) = 0;
    coarse_srch_cells( 1 ) = 1;
    coarse_srch_cells( 2 ) = 2;
    coarse_srch_cells( 3 ) = 1;
    coarse_srch_cells( 4 ) = 2;

    DataTransferKit::FineSearch<DeviceType>::search(
        reference_points, point_in_cell, physical_points, cells,
        coarse_srch_pts, coarse_srch_cells, cell_topology );

    auto reference_points_host = Kokkos::create_mirror_view( reference_points );
    Kokkos::deep_copy( reference_points_host, reference_points );
    auto point_in_cell_host = Kokkos::create_mirror_view( point_in_cell );
    Kokkos::deep_copy( point_in_cell_host, point_in_cell );

    std::vector<std::array<double, dim>> reference_points_ref = {
        {{2., 0., -0.4}},
        {{0., 0., -0.4}},
        {{-2., 0., -0.4}},
        {{2., -0.4, 0.}},
        {{0., -0.4, 0.}}};
    std::vector<bool> point_in_cell_ref = {false, true, false, false, true};

    double const tol = 1e-14;
    for ( unsigned int i = 0; i < n_ref_pts; ++i )
    {
        for ( unsigned int j = 0; j < dim; ++j )
            TEST_ASSERT( std::abs( reference_points_host( i, j ) -
                                   reference_points_ref[i][j] ) < tol );
        TEST_EQUALITY( point_in_cell_host( i ), point_in_cell_ref[i] );
    }
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( FineSearch, quad_4, DeviceType )
{
    unsigned int constexpr dim = 2;
    shards::CellTopology cell_topology(
        shards::getCellTopologyData<shards::Quadrilateral<4>>() );
    unsigned int constexpr n_ref_pts = 5;

    Kokkos::View<double * [dim], DeviceType> reference_points( "ref_pts",
                                                               n_ref_pts );
    Kokkos::View<bool *, DeviceType> point_in_cell( "pt_in_cell", n_ref_pts );
    // Physical points are (1.5, 0.5) and (2.5, 0.3)
    Kokkos::View<double * [dim], DeviceType> physical_points( "phys_pts", 2 );
    physical_points( 0, 0 ) = 1.5;
    physical_points( 0, 1 ) = 0.5;
    physical_points( 1, 0 ) = 2.5;
    physical_points( 1, 1 ) = 0.3;
    // Vertices of the cells
    Kokkos::View<double * * [dim], DeviceType> cells( "cell_nodes", 3, 4 );
    // First cell
    cells( 0, 0, 0 ) = 0.;
    cells( 0, 0, 1 ) = 0.;
    cells( 0, 1, 0 ) = 1.;
    cells( 0, 1, 1 ) = 0.;
    cells( 0, 2, 0 ) = 1.;
    cells( 0, 2, 1 ) = 1.;
    cells( 0, 3, 0 ) = 0.;
    cells( 0, 3, 1 ) = 1.;
    // Second cell
    cells( 1, 0, 0 ) = 1.;
    cells( 1, 0, 1 ) = 0.;
    cells( 1, 1, 0 ) = 2.;
    cells( 1, 1, 1 ) = 0.;
    cells( 1, 2, 0 ) = 2.;
    cells( 1, 2, 1 ) = 1.;
    cells( 1, 3, 0 ) = 1.;
    cells( 1, 3, 1 ) = 1.;
    // Third cell
    cells( 2, 0, 0 ) = 2.;
    cells( 2, 0, 1 ) = 0.;
    cells( 2, 1, 0 ) = 3.;
    cells( 2, 1, 1 ) = 0.;
    cells( 2, 2, 0 ) = 3.;
    cells( 2, 2, 1 ) = 1.;
    cells( 2, 3, 0 ) = 2.;
    cells( 2, 3, 1 ) = 1.;
    // Coarse search output: points
    Kokkos::View<unsigned int *, DeviceType> coarse_srch_pts( "coarse_srch_pts",
                                                              5 );
    coarse_srch_pts( 0 ) = 0;
    coarse_srch_pts( 1 ) = 0;
    coarse_srch_pts( 2 ) = 0;
    coarse_srch_pts( 3 ) = 1;
    coarse_srch_pts( 4 ) = 1;
    // Coarse search output: cells
    Kokkos::View<unsigned int *, DeviceType> coarse_srch_cells(
        "coarse_srch_cells", 5 );
    coarse_srch_cells( 0 ) = 0;
    coarse_srch_cells( 1 ) = 1;
    coarse_srch_cells( 2 ) = 2;
    coarse_srch_cells( 3 ) = 1;
    coarse_srch_cells( 4 ) = 2;

    DataTransferKit::FineSearch<DeviceType>::search(
        reference_points, point_in_cell, physical_points, cells,
        coarse_srch_pts, coarse_srch_cells, cell_topology );

    auto reference_points_host = Kokkos::create_mirror_view( reference_points );
    Kokkos::deep_copy( reference_points_host, reference_points );
    auto point_in_cell_host = Kokkos::create_mirror_view( point_in_cell );
    Kokkos::deep_copy( point_in_cell_host, point_in_cell );

    std::vector<std::array<double, dim>> reference_points_ref = {
        {{2., 0.}}, {{0., 0.}}, {{-2., 0.}}, {{2., -0.4}}, {{0., -0.4}}};
    std::vector<bool> point_in_cell_ref = {false, true, false, false, true};

    double const tol = 1e-14;
    for ( unsigned int i = 0; i < n_ref_pts; ++i )
    {
        for ( unsigned int j = 0; j < dim; ++j )
            TEST_ASSERT( std::abs( reference_points_host( i, j ) -
                                   reference_points_ref[i][j] ) < tol );
        TEST_EQUALITY( point_in_cell_host( i ), point_in_cell_ref[i] );
    }
}

// Include the test macros.
#include "DataTransferKitSearch_ETIHelperMacros.h"

// Create the test group
#define UNIT_TEST_GROUP( NODE )                                                \
    using DeviceType##NODE = typename NODE::device_type;                       \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( FineSearch, hex_8,                   \
                                          DeviceType##NODE )                   \
    using DeviceType##NODE = typename NODE::device_type;                       \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( FineSearch, quad_4, DeviceType##NODE )

// Demangle the types
DTK_ETI_MANGLING_TYPEDEFS()

// Instantiate the tests
DTK_INSTANTIATE_N( UNIT_TEST_GROUP )
