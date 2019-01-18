/****************************************************************************
 * Copyright (c) 2012-2019 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <DTK_DistributedSearchTree.hpp>

#include <Kokkos_DefaultNode.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_StandardCatchMacros.hpp>
#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <cmath> // cbrt
#include <random>

template <class NO>
int main_( Teuchos::CommandLineProcessor &clp, int argc, char *argv[] )
{
    Teuchos::Time timer( "timer" );
    Teuchos::TimeMonitor time_monitor( timer );

    using DeviceType = typename NO::device_type;
    using ExecutionSpace = typename DeviceType::execution_space;

    int n_values = 50000;
    int n_queries = 20000;
    int n_neighbors = 10;
    double overlap = 0.;
    int partition_dim = 3;
    bool perform_knn_search = true;
    bool perform_radius_search = true;

    clp.setOption( "values", &n_values,
                   "number of indexable values (source) per MPI rank" );
    clp.setOption( "queries", &n_queries,
                   "number of queries (target) per MPI rank" );
    clp.setOption( "neighbors", &n_neighbors,
                   "desired number of results per query" );
    clp.setOption( "overlap", &overlap,
                   "overlap of the point clouds. 0 means the clouds are built "
                   "next to each other. 1 means that there are built at the "
                   "same place. Negative values and values larger than two "
                   "means that the clouds are separated" );
    clp.setOption( "partition_dim", &partition_dim,
                   "number of dimension used by the partitioning of the global "
                   "point cloud. 1 -> local clouds are aligned on a line, 2 -> "
                   "local clouds form a board, 3 -> local clouds form a box" );
    clp.setOption( "perform-knn-search", "do-not-perform-knn-search",
                   &perform_knn_search,
                   "whether or not to perform kNN search" );
    clp.setOption( "perform-radius-search", "do-not-perform-radius-search",
                   &perform_radius_search,
                   "whether or not to perform radius search" );

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

    Teuchos::RCP<const Teuchos::Comm<int>> comm =
        Teuchos::DefaultComm<int>::getComm();
    int rank = Teuchos::rank( *comm );
    Kokkos::View<DataTransferKit::Point *, DeviceType> random_points(
        "random_points" );
    {
        // Random points are "reused" between building the tree and performing
        // queries. Note that this means that for the points in the middle of
        // the local domains there won't be any communication.
        auto n = std::max( n_values, n_queries );
        Kokkos::resize( random_points, n );

        auto random_points_host = Kokkos::create_mirror_view( random_points );

        // Generate random points uniformely distributed within a box.
        auto const a = std::cbrt( n_values );
        std::uniform_real_distribution<double> distribution( -a, +a );
        std::default_random_engine generator;
        auto random = [&distribution, &generator]() {
            return distribution( generator );
        };

        double offset_x = 0.;
        double offset_y = 0.;
        double offset_z = 0.;
        // Change the geometry of the problem. In 1D, all the point clouds are
        // aligned on a line. In 2D, the point clouds create a board and in 3D,
        // they create a box.
        switch ( partition_dim )
        {
        case 1:
        {
            offset_x = 2. * ( 1. - overlap ) * a * rank;

            break;
        }
        case 2:
        {
            int n_procs = Teuchos::size( *comm );
            int i_max = std::ceil( std::sqrt( n_procs ) );
            int i = rank % i_max;
            int j = rank / i_max;
            offset_x = 2. * ( 1. - overlap ) * a * i;
            offset_y = 2. * ( 1. - overlap ) * a * j;

            break;
        }
        case 3:
        {
            int n_procs = Teuchos::size( *comm );
            int i_max = std::ceil( std::cbrt( n_procs ) );
            int j_max = i_max;
            int i = rank % i_max;
            int j = ( rank / i_max ) % j_max;
            int k = rank / ( i_max * j_max );
            offset_x = 2. * ( 1. - overlap ) * a * i;
            offset_y = 2. * ( 1. - overlap ) * a * j;
            offset_z = 2. * ( 1. - overlap ) * a * k;

            break;
        }
        default:
        {
            throw std::runtime_error( "partition_dim should be 1, 2, or 3" );
        }
        }

        for ( int i = 0; i < n; ++i )
            random_points_host( i ) = {{offset_x + random(),
                                        offset_y + random(),
                                        offset_z + random()}};
        Kokkos::deep_copy( random_points, random_points_host );
    }

    Kokkos::View<DataTransferKit::Box *, DeviceType> bounding_boxes(
        Kokkos::ViewAllocateWithoutInitializing( "bounding_boxes" ), n_values );
    Kokkos::parallel_for( "bvh_driver:construct_bounding_boxes",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n_values ),
                          KOKKOS_LAMBDA( int i ) {
                              double const x = random_points( i )[0];
                              double const y = random_points( i )[1];
                              double const z = random_points( i )[2];
                              bounding_boxes( i ) = {
                                  {{x - 1., y - 1., z - 1.}},
                                  {{x + 1., y + 1., z + 1.}}};
                          } );
    Kokkos::fence();

    auto construction = time_monitor.getNewTimer( "construction" );
    comm->barrier();
    construction->start();
    DataTransferKit::DistributedSearchTree<DeviceType> distributed_tree(
        *( Teuchos::rcp_dynamic_cast<Teuchos::MpiComm<int> const>( comm )
               ->getRawMpiComm() ),
        bounding_boxes );
    construction->stop();

    std::ostream &os = std::cout;
    if ( rank == 0 )
        os << "contruction done\n";

    if ( perform_knn_search )
    {
        Kokkos::View<DataTransferKit::Nearest<DataTransferKit::Point> *,
                     DeviceType>
            queries( Kokkos::ViewAllocateWithoutInitializing( "queries" ),
                     n_queries );
        Kokkos::parallel_for(
            "bvh_driver:setup_knn_search_queries",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
            KOKKOS_LAMBDA( int i ) {
                queries( i ) = DataTransferKit::nearest<DataTransferKit::Point>(
                    random_points( i ), n_neighbors );
            } );
        Kokkos::fence();

        Kokkos::View<int *, DeviceType> offset( "offset" );
        Kokkos::View<int *, DeviceType> indices( "indices" );
        Kokkos::View<int *, DeviceType> ranks( "ranks" );

        auto knn = time_monitor.getNewTimer( "knn" );
        comm->barrier();
        knn->start();
        distributed_tree.query( queries, indices, offset, ranks );
        knn->stop();

        if ( rank == 0 )
            os << "knn done\n";
    }

    if ( perform_radius_search )
    {
        Kokkos::View<DataTransferKit::Within *, DeviceType> queries(
            Kokkos::ViewAllocateWithoutInitializing( "queries" ), n_queries );
        // radius chosen in order to control the number of results per query
        // NOTE: minus "1+sqrt(3)/2 \approx 1.37" matches the size of the boxes
        // inserted into the tree (mid-point between half-edge and
        // half-diagonal)
        double const r = 2. * std::cbrt( static_cast<double>( n_neighbors ) *
                                         3. / ( 4. * M_PI ) ) -
                         ( 1. + std::sqrt( 3. ) ) / 2.;
        Kokkos::parallel_for(
            "bvh_driver:setup_radius_search_queries",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
            KOKKOS_LAMBDA( int i ) {
                queries( i ) = DataTransferKit::within( random_points( i ), r );
            } );
        Kokkos::fence();

        Kokkos::View<int *, DeviceType> offset( "offset" );
        Kokkos::View<int *, DeviceType> indices( "indices" );
        Kokkos::View<int *, DeviceType> ranks( "ranks" );

        auto radius = time_monitor.getNewTimer( "radius" );
        comm->barrier();
        radius->start();
        distributed_tree.query( queries, indices, offset, ranks );
        radius->stop();

        if ( rank == 0 )
            os << "radius done\n";
    }
    time_monitor.summarize( comm.ptr() );

    return 0;
}

int main( int argc, char *argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    bool success = true;
    bool verbose = true;

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
            success = false;
        case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:
        case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION:
        case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:
            break;
        }

        if ( !success )
        {
            // do nothing, just skip other if clauses
        }
        else if ( node == "" )
        {
            typedef KokkosClassic::DefaultNode::DefaultNodeType Node;
            main_<Node>( clp, argc, argv );
        }
        else if ( node == "serial" )
        {
#ifdef KOKKOS_ENABLE_SERIAL
            typedef Kokkos::Compat::KokkosSerialWrapperNode Node;
            main_<Node>( clp, argc, argv );
#else
            throw std::runtime_error( "Serial node type is disabled" );
#endif
        }
        else if ( node == "openmp" )
        {
#ifdef KOKKOS_ENABLE_OPENMP
            typedef Kokkos::Compat::KokkosOpenMPWrapperNode Node;
            main_<Node>( clp, argc, argv );
#else
            throw std::runtime_error( "OpenMP node type is disabled" );
#endif
        }
        else if ( node == "cuda" )
        {
#ifdef KOKKOS_ENABLE_CUDA
            typedef Kokkos::Compat::KokkosCudaWrapperNode Node;
            main_<Node>( clp, argc, argv );
#else
            throw std::runtime_error( "CUDA node type is disabled" );
#endif
        }
        else
        {
            throw std::runtime_error( "Unrecognized node type" );
        }
    }
    TEUCHOS_STANDARD_CATCH_STATEMENTS( verbose, std::cerr, success );

    Kokkos::finalize();

    MPI_Finalize();

    return ( success ? EXIT_SUCCESS : EXIT_FAILURE );
}
