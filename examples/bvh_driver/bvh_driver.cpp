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

#include <DTK_LinearBVH.hpp>

#include <Kokkos_DefaultNode.hpp>
#include <Teuchos_CommandLineProcessor.hpp>

#include <benchmark/benchmark.h>

#include <chrono>
#include <cmath> // cbrt
#include <cstdlib>
#include <random>

template <typename DeviceType>
Kokkos::View<DataTransferKit::Box *, DeviceType> constructBoxes( int n_values )
{
    Kokkos::View<DataTransferKit::Point *, DeviceType> random_points(
        Kokkos::ViewAllocateWithoutInitializing( "random_points" ), n_values );
    auto random_points_host = Kokkos::create_mirror_view( random_points );
    // Generate random points uniformely distributed within a box.  The edge
    // length of the box chosen such that object density (here objects will be
    // boxes 2x2x2 centered around a random point) will remain constant as
    // problem size is changed.
    auto const a = std::cbrt( n_values );
    std::uniform_real_distribution<double> distribution( -a, +a );
    std::default_random_engine generator;
    auto random = [&distribution, &generator]() {
        return distribution( generator );
    };
    for ( int i = 0; i < n_values; ++i )
        random_points_host( i ) = {{random(), random(), random()}};
    Kokkos::deep_copy( random_points, random_points_host );

    using ExecutionSpace = typename DeviceType::execution_space;
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
    return bounding_boxes;
}

template <typename DeviceType>
Kokkos::View<DataTransferKit::Nearest<DataTransferKit::Point> *, DeviceType>
makeNearestQueries( int n_values, int n_queries, int n_neighbors )
{
    Kokkos::View<DataTransferKit::Point *, DeviceType> random_points(
        Kokkos::ViewAllocateWithoutInitializing( "random_points" ), n_queries );
    auto random_points_host = Kokkos::create_mirror_view( random_points );
    auto const a = std::cbrt( n_values );
    std::uniform_real_distribution<double> distribution( -a, +a );
    std::default_random_engine generator;
    auto random = [&distribution, &generator]() {
        return distribution( generator );
    };
    for ( int i = 0; i < n_queries; ++i )
        random_points_host( i ) = {{random(), random(), random()}};
    Kokkos::deep_copy( random_points, random_points_host );

    Kokkos::View<DataTransferKit::Nearest<DataTransferKit::Point> *, DeviceType>
        queries( Kokkos::ViewAllocateWithoutInitializing( "queries" ),
                 n_queries );
    using ExecutionSpace = typename DeviceType::execution_space;
    Kokkos::parallel_for(
        "bvh_driver:setup_knn_search_queries",
        Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
        KOKKOS_LAMBDA( int i ) {
            queries( i ) = DataTransferKit::nearest<DataTransferKit::Point>(
                random_points( i ), n_neighbors );
        } );
    Kokkos::fence();
    return queries;
}

template <typename DeviceType>
Kokkos::View<DataTransferKit::Within *, DeviceType>
makeSpatialQueries( int n_values, int n_queries, int n_neighbors )
{
    Kokkos::View<DataTransferKit::Point *, DeviceType> random_points(
        Kokkos::ViewAllocateWithoutInitializing( "random_points" ), n_queries );
    auto random_points_host = Kokkos::create_mirror_view( random_points );
    auto const a = std::cbrt( n_values );
    std::uniform_real_distribution<double> distribution( -a, +a );
    std::default_random_engine generator;
    auto random = [&distribution, &generator]() {
        return distribution( generator );
    };
    for ( int i = 0; i < n_queries; ++i )
        random_points_host( i ) = {{random(), random(), random()}};
    Kokkos::deep_copy( random_points, random_points_host );

    Kokkos::View<DataTransferKit::Within *, DeviceType> queries(
        Kokkos::ViewAllocateWithoutInitializing( "queries" ), n_queries );
    // radius chosen in order to control the number of results per query
    // NOTE: minus "1+sqrt(3)/2 \approx 1.37" matches the size of the boxes
    // inserted into the tree (mid-point between half-edge and half-diagonal)
    double const r = 2. * std::cbrt( static_cast<double>( n_neighbors ) * 3. /
                                     ( 4. * M_PI ) ) -
                     ( 1. + std::sqrt( 3. ) ) / 2.;
    using ExecutionSpace = typename DeviceType::execution_space;
    Kokkos::parallel_for( "bvh_driver:setup_radius_search_queries",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
                          KOKKOS_LAMBDA( int i ) {
                              queries( i ) = DataTransferKit::within(
                                  random_points( i ), r );
                          } );
    Kokkos::fence();
    return queries;
}

template <class DeviceType>
void BM_construction( benchmark::State &state )
{
    int const n_values = state.range( 0 );
    auto bounding_boxes = constructBoxes<DeviceType>( n_values );

    for ( auto _ : state )
    {
        auto const start = std::chrono::high_resolution_clock::now();
        DataTransferKit::BVH<DeviceType> bvh( bounding_boxes );
        auto const end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        state.SetIterationTime( elapsed_seconds.count() );
    }
}

template <class DeviceType>
void BM_knn_search( benchmark::State &state )
{
    int const n_values = state.range( 0 );
    int const n_queries = state.range( 1 );
    int const n_neighbors = state.range( 2 );

    DataTransferKit::BVH<DeviceType> bvh(
        constructBoxes<DeviceType>( n_values ) );
    auto const queries =
        makeNearestQueries<DeviceType>( n_values, n_queries, n_neighbors );

    for ( auto _ : state )
    {
        Kokkos::View<int *, DeviceType> offset( "offset" );
        Kokkos::View<int *, DeviceType> indices( "indices" );
        auto const start = std::chrono::high_resolution_clock::now();
        bvh.query( queries, indices, offset );
        auto const end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        state.SetIterationTime( elapsed_seconds.count() );
    }
}

template <class DeviceType>
void BM_radius_search( benchmark::State &state )
{
    int const n_values = state.range( 0 );
    int const n_queries = state.range( 1 );
    int const n_neighbors = state.range( 2 );
    int const buffer_size = state.range( 3 );

    DataTransferKit::BVH<DeviceType> bvh(
        constructBoxes<DeviceType>( n_values ) );
    auto const queries =
        makeSpatialQueries<DeviceType>( n_values, n_queries, n_neighbors );

    for ( auto _ : state )
    {
        Kokkos::View<int *, DeviceType> offset( "offset" );
        Kokkos::View<int *, DeviceType> indices( "indices" );
        auto const start = std::chrono::high_resolution_clock::now();
        bvh.query( queries, indices, offset, buffer_size );
        auto const end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        state.SetIterationTime( elapsed_seconds.count() );
    }
}

class KokkosScopeGuard
{
  public:
    KokkosScopeGuard( int &argc, char *argv[] )
    {
        Kokkos::initialize( argc, argv );
    }
    ~KokkosScopeGuard() { Kokkos::finalize(); }
};

#define REGISTER_BENCHMARK( DeviceType )                                       \
    BENCHMARK_TEMPLATE( BM_construction, DeviceType )                          \
        ->Arg( n_values )                                                      \
        ->UseManualTime()                                                      \
        ->Unit( benchmark::kMicrosecond );                                     \
    BENCHMARK_TEMPLATE( BM_knn_search, DeviceType )                            \
        ->Args( {n_values, n_queries, n_neighbors} )                           \
        ->UseManualTime()                                                      \
        ->Unit( benchmark::kMicrosecond );                                     \
    BENCHMARK_TEMPLATE( BM_radius_search, DeviceType )                         \
        ->Args( {n_values, n_queries, n_neighbors, buffer_size} )              \
        ->UseManualTime()                                                      \
        ->Unit( benchmark::kMicrosecond );

int main( int argc, char *argv[] )
{
    KokkosScopeGuard guard( argc, argv );

    bool const throw_exceptions = false;
    bool const recognise_all_options = false;
    Teuchos::CommandLineProcessor clp( throw_exceptions,
                                       recognise_all_options );
    int n_values = 50000;
    int n_queries = 20000;
    int n_neighbors = 10;
    int buffer_size = 0;
    clp.setOption( "values", &n_values, "number of indexable values (source)" );
    clp.setOption( "queries", &n_queries, "number of queries (target)" );
    clp.setOption( "neighbors", &n_neighbors,
                   "desired number of results per query" );
    clp.setOption( "buffer", &buffer_size,
                   "size for buffer optimization in radius search" );

    switch ( clp.parse( argc, argv, NULL ) )
    {
    case Teuchos::CommandLineProcessor::PARSE_ERROR:
        return EXIT_FAILURE;
    case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION:
    case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:
        clp.printHelpMessage( "benchmark", std::cout );
    case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:
        break;
    }

    // benchmark::Initialize() calls exit(0) when `--help` so register
    // Kokkos::finalize() to be called on normal program termination.
    std::atexit( Kokkos::finalize );
    benchmark::Initialize( &argc, argv );

    // Throw if an option is not recognised
    clp.throwExceptions( true );
    clp.recogniseAllOptions( true );
    switch ( clp.parse( argc, argv, NULL ) )
    {
    case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION:
    case Teuchos::CommandLineProcessor::PARSE_ERROR:
        return EXIT_FAILURE;
    case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:
    case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:
        break;
    }

#ifdef KOKKOS_ENABLE_SERIAL
    using Serial = Kokkos::Compat::KokkosSerialWrapperNode::device_type;
    REGISTER_BENCHMARK( Serial );
#endif

#ifdef KOKKOS_ENABLE_OPENMP
    using OpenMP = Kokkos::Compat::KokkosOpenMPWrapperNode::device_type;
    REGISTER_BENCHMARK( OpenMP );
#endif

#ifdef KOKKOS_ENABLE_CUDA
    using Cuda = Kokkos::Compat::KokkosCudaWrapperNode::device_type;
    REGISTER_BENCHMARK( Cuda );
#endif

    benchmark::RunSpecifiedBenchmarks();

    return EXIT_SUCCESS;
}
