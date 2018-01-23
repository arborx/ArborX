/****************************************************************************
 * Copyright (c) 2012-2018 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_StandardCatchMacros.hpp>

#include <Kokkos_DefaultNode.hpp>

#include <DTK_ConfigDefs.hpp>

#include <DTK_LinearBVH.hpp>

#include <cmath>
#include <functional>
#include <random>

namespace details = DataTransferKit::Details;

std::vector<std::array<double, 3>>
make_stuctured_cloud( double Lx, double Ly, double Lz, int nx, int ny, int nz )
{
    std::vector<std::array<double, 3>> cloud( nx * ny * nz );
    std::function<int( int, int, int )> ind = [nx, ny]( int i, int j, int k ) {
        return i + j * nx + k * ( nx * ny );
    };
    double x, y, z;
    for ( int i = 0; i < nx; ++i )
        for ( int j = 0; j < ny; ++j )
            for ( int k = 0; k < nz; ++k )
            {
                x = i * Lx / ( nx - 1 );
                y = j * Ly / ( ny - 1 );
                z = k * Lz / ( nz - 1 );
                cloud[ind( i, j, k )] = {{x, y, z}};
            }
    return cloud;
}

std::vector<std::array<double, 3>> make_random_cloud( double Lx, double Ly,
                                                      double Lz, int n )
{
    std::vector<std::array<double, 3>> cloud( n );
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution_x( 0.0, Lx );
    std::uniform_real_distribution<double> distribution_y( 0.0, Ly );
    std::uniform_real_distribution<double> distribution_z( 0.0, Lz );
    for ( int i = 0; i < n; ++i )
    {
        double x = distribution_x( generator );
        double y = distribution_y( generator );
        double z = distribution_z( generator );
        cloud[i] = {{x, y, z}};
    }
    return cloud;
}

template <class NO>
int main_( Teuchos::CommandLineProcessor &clp, int argc, char *argv[] )
{
    using DeviceType = typename NO::device_type;
    using ExecutionSpace = typename DeviceType::execution_space;

    double Lx = 100.0;
    double Ly = 100.0;
    double Lz = 100.0;
    int nx = 11;
    int ny = 11;
    int nz = 11;
    int n_points = 100;
    std::string mode = "radius";

    clp.setOption( "nx", &nx, "source mesh points in x-direction." );
    clp.setOption( "ny", &ny, "source mesh points in y-direction." );
    clp.setOption( "nz", &nz, "source mesh points in z-direction." );
    clp.setOption( "N", &n_points,
                   "number of target mesh points (distributed randomly)." );
    clp.setOption( "mode", &mode, "mode: (knn | radius)" );

    clp.recogniseAllOptions( true );
    switch ( clp.parse( argc, argv ) )
    {
    case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:
        return EXIT_SUCCESS;
    case Teuchos::CommandLineProcessor::PARSE_ERROR:
    case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION:
        return EXIT_FAILURE;
    case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:
        break;
    }

    // contruct a cloud of points (nodes of a structured grid)
    auto cloud = make_stuctured_cloud( Lx, Ly, Lz, nx, ny, nz );
    int n = cloud.size();

    Kokkos::View<DataTransferKit::Box *, DeviceType> bounding_boxes(
        "bounding_boxes", n );
    auto bounding_boxes_host = Kokkos::create_mirror_view( bounding_boxes );
    // build bounding volume hierarchy
    for ( int i = 0; i < n; ++i )
    {
        auto const &point = cloud[i];
        double x = std::get<0>( point );
        double y = std::get<1>( point );
        double z = std::get<2>( point );
        bounding_boxes_host[i] = {{{x, y, z}}, {{x, y, z}}};
    }
    Kokkos::deep_copy( bounding_boxes, bounding_boxes_host );

    DataTransferKit::BVH<DeviceType> bvh( bounding_boxes );

    // random points for radius search and kNN queries
    auto queries = make_random_cloud( Lx, Ly, Lz, n_points );
    Kokkos::View<double * [3], ExecutionSpace> point_coords( "point_coords",
                                                             n_points );

    auto point_coords_host = Kokkos::create_mirror_view( point_coords );
    for ( int i = 0; i < n_points; ++i )
    {
        auto const &point = queries[i];
        point_coords_host( i, 0 ) = std::get<0>( point );
        point_coords_host( i, 1 ) = std::get<1>( point );
        point_coords_host( i, 2 ) = std::get<2>( point );
    }
    Kokkos::deep_copy( point_coords, point_coords_host );

    std::default_random_engine generator;

    if ( mode == "knn" )
    {
        Kokkos::View<int *, ExecutionSpace> k( "distribution_k", n_points );
        auto k_host = Kokkos::create_mirror_view( k );

        // use random number k of for the kNN search
        int max_k = std::floor( sqrt( nx * nx + ny * ny + nz * nz ) );
        std::uniform_int_distribution<int> distribution_k( 1, max_k );
        for ( int i = 0; i < n_points; ++i )
        {
            k_host[i] = distribution_k( generator );
        }
        Kokkos::deep_copy( k, k_host );

        Kokkos::View<details::Nearest<DataTransferKit::Point> *, DeviceType>
            nearest_queries( "nearest_queries", n_points );
        Kokkos::parallel_for(
            DTK_MARK_REGION( "register_nearest_queries" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_points ),
            KOKKOS_LAMBDA( int i ) {
                nearest_queries( i ) = details::nearest<DataTransferKit::Point>(
                    {{point_coords( i, 0 ), point_coords( i, 1 ),
                      point_coords( i, 2 )}},
                    k( i ) );
            } );
        Kokkos::fence();

        // do the search
        Kokkos::View<int *, DeviceType> offset_nearest( "offset_nearest" );
        Kokkos::View<int *, DeviceType> indices_nearest( "indices_nearest" );
        bvh.query( nearest_queries, indices_nearest, offset_nearest );
    }
    else if ( mode == "radius" )
    {
        Kokkos::View<double *, ExecutionSpace> radii( "radii", n_points );
        auto radii_host = Kokkos::create_mirror_view( radii );

        // use random radius for the search
        // set the limit of approximately 100 points by
        // solving n_points*pi*r^2/(Lx^2 + Ly^2 + Lz^2) <= 100
        const int approx_points = 100;
        double max_radius = sqrt(
            approx_points * ( Lx * Lx + Ly * Ly + Lz * Lz ) / ( n * M_PI ) );
        std::uniform_real_distribution<double> distribution_radius(
            0.0, max_radius );
        for ( int i = 0; i < n_points; ++i )
        {
            radii_host[i] = distribution_radius( generator );
        }
        Kokkos::deep_copy( radii, radii_host );

        Kokkos::View<details::Within *, DeviceType> within_queries(
            "within_queries", n_points );
        Kokkos::parallel_for(
            DTK_MARK_REGION( "register_within_queries" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_points ),
            KOKKOS_LAMBDA( int i ) {
                within_queries( i ) = details::within(
                    {{point_coords( i, 0 ), point_coords( i, 1 ),
                      point_coords( i, 2 )}},
                    radii( i ) );
            } );
        Kokkos::fence();

        Kokkos::View<int *, DeviceType> offset_within( "offset_within" );
        Kokkos::View<int *, DeviceType> indices_within( "indices_within" );
        bvh.query( within_queries, indices_within, offset_within );
    }

    return 0;
}

int main( int argc, char *argv[] )
{
    Kokkos::initialize( argc, argv );

    bool success = false;
    bool verbose = true;

    int rv = 0;
    try
    {
        const bool throwExceptions = false;

        Teuchos::CommandLineProcessor clp( throwExceptions );

        std::string node = "";
        clp.setOption( "node", &node, "node type (serial | openmp | cuda)" );

        clp.recogniseAllOptions( false );
        switch ( clp.parse( argc, argv, NULL ) )
        {
        case Teuchos::CommandLineProcessor::PARSE_ERROR:
            rv = 1;
        case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:
        case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION:
        case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:
            break;
        }

        if ( rv )
        {
            // do nothing, just skip other if clauses
        }
        else if ( node == "" )
        {
            typedef KokkosClassic::DefaultNode::DefaultNodeType Node;
            rv = main_<Node>( clp, argc, argv );
        }
        else if ( node == "serial" )
        {
#ifdef KOKKOS_HAVE_SERIAL
            typedef Kokkos::Compat::KokkosSerialWrapperNode Node;
            rv = main_<Node>( clp, argc, argv );
#else
            throw std::runtime_error( "Serial node type is disabled" );
#endif
        }
        else if ( node == "openmp" )
        {
#ifdef KOKKOS_HAVE_OPENMP
            typedef Kokkos::Compat::KokkosOpenMPWrapperNode Node;
            rv = main_<Node>( clp, argc, argv );
#else
            throw std::runtime_error( "OpenMP node type is disabled" );
#endif
        }
        else if ( node == "cuda" )
        {
#ifdef KOKKOS_HAVE_CUDA
            typedef Kokkos::Compat::KokkosCudaWrapperNode Node;
            rv = main_<Node>( clp, argc, argv );
#else
            throw std::runtime_error( "CUDA node type is disabled" );
#endif
        }
        else
        {
            throw std::runtime_error( "Unrecognized node type" );
        }

        if ( rv )
            success = false;
    }
    TEUCHOS_STANDARD_CATCH_STATEMENTS( verbose, std::cerr, success );

    Kokkos::finalize();

    return ( success ? EXIT_SUCCESS : EXIT_FAILURE );
}
