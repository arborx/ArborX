/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_StandardCatchMacros.hpp>

#include <Kokkos_DefaultNode.hpp>

#include <DTK_ConfigDefs.hpp>

#include <DTK_LinearBVH.hpp>
#include <details/DTK_DetailsTreeTraversal.hpp>

#include <cmath>
#include <random>

const int MAX_N_KNN = 1000;
const int MAX_N_WITHIN = 10000;

template <typename NO>
class RandomWithinLambda
{
  public:
    using DeviceType = typename DataTransferKit::BVH<NO>::DeviceType;
    using ExecutionSpace = typename DeviceType::execution_space;
    using MemorySpace = typename DeviceType::memory_space;

    using PointsView = Kokkos::View<double * [3], ExecutionSpace>;
    using RView = Kokkos::View<double *, ExecutionSpace>;

  public:
    RandomWithinLambda( PointsView point_coords, RView radii,
                        DataTransferKit::BVH<NO> bvh )
        : _point_coords( point_coords )
        , _radii( radii )
        , _bvh( bvh )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i ) const
    {
        DataTransferKit::Details::Within within_predicate(
            {_point_coords( i, 0 ), _point_coords( i, 1 ),
             _point_coords( i, 2 )},
            _radii( i ) );
        unsigned int constexpr max_n_indices = MAX_N_WITHIN;
        int indices[max_n_indices];

        unsigned int n_indices = 0;
        DataTransferKit::Details::spatial_query(
            _bvh, within_predicate, indices, n_indices, max_n_indices );
    }

  private:
    PointsView _point_coords;
    RView _radii;
    DataTransferKit::BVH<NO> _bvh;
};

template <typename NO>
class RandomNearestLambda
{
  public:
    using DeviceType = typename DataTransferKit::BVH<NO>::DeviceType;
    using ExecutionSpace = typename DeviceType::execution_space;

    using PointsView = Kokkos::View<double * [3], ExecutionSpace>;
    using KView = Kokkos::View<int *, ExecutionSpace>;

  public:
    RandomNearestLambda( PointsView point_coords, KView k,
                         DataTransferKit::BVH<NO> bvh )
        : _point_coords( point_coords )
        , _k( k )
        , _bvh( bvh )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i ) const
    {
        unsigned int constexpr max_n_indices = MAX_N_KNN;
        int indices[max_n_indices];
        unsigned int n_indices = 0;
        DataTransferKit::Details::nearest_query(
            _bvh, {_point_coords( i, 0 ), _point_coords( i, 1 ),
                   _point_coords( i, 2 )},
            _k[i], indices, n_indices, max_n_indices );
    }

  private:
    PointsView _point_coords;
    KView _k;
    DataTransferKit::BVH<NO> _bvh;
};

std::vector<std::array<double, 3>>
make_stuctured_cloud( double Lx, double Ly, double Lz, int nx, int ny, int nz )
{
    std::vector<std::array<double, 3>> cloud( nx * ny * nz );
    std::function<int( int, int, int )> ind = [nx, ny, nz](
        int i, int j, int k ) { return i + j * nx + k * ( nx * ny ); };
    double x, y, z;
    for ( int i = 0; i < nx; ++i )
        for ( int j = 0; j < ny; ++j )
            for ( int k = 0; k < nz; ++k )
            {
                x = i * Lx / ( nx - 1 );
                y = j * Ly / ( ny - 1 );
                z = k * Lz / ( nz - 1 );
                cloud[ind( i, j, k )] = {x, y, z};
            }
    return cloud;
}

std::vector<std::array<double, 3>> make_random_cloud( double Lx, double Ly,
                                                      double Lz, int n )
{
    std::vector<std::array<double, 3>> cloud( n );
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution_x( 0.0, Lz );
    std::uniform_real_distribution<double> distribution_y( 0.0, Ly );
    std::uniform_real_distribution<double> distribution_z( 0.0, Lz );
    for ( int i = 0; i < n; ++i )
    {
        double x = distribution_x( generator );
        double y = distribution_y( generator );
        double z = distribution_z( generator );
        cloud[i] = {x, y, z};
    }
    return cloud;
}

template <class NO>
int main_( Teuchos::CommandLineProcessor &clp, int argc, char *argv[] )
{
    using DeviceType = typename DataTransferKit::BVH<NO>::DeviceType;
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
        bounding_boxes_host[i] = {
            x, x, y, y, z, z,
        };
    }

    Kokkos::deep_copy( bounding_boxes, bounding_boxes_host );

    DataTransferKit::BVH<NO> bvh( bounding_boxes );

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

        if ( max_k >= MAX_N_KNN )
            throw std::runtime_error(
                "Abort: some hardcoded arrays may overflow." );

        // do the search
        RandomNearestLambda<NO> random_nearest_lambda( point_coords, k, bvh );

        Kokkos::parallel_for(
            "random_nearest",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_points ),
            random_nearest_lambda );
        Kokkos::fence();
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

        if ( 2 * approx_points >= MAX_N_WITHIN )
            throw std::runtime_error(
                "Abort: some hardcoded arrays may overflow." );

        // do the search
        RandomWithinLambda<NO> random_within_lambda( point_coords, radii, bvh );

        Kokkos::parallel_for(
            "random_within", Kokkos::RangePolicy<ExecutionSpace>( 0, n_points ),
            random_within_lambda );
        Kokkos::fence();
    }

    return 0;
}

int main( int argc, char *argv[] )
{
    bool success = false;
    bool verbose = true;

    try
    {
        Kokkos::initialize();

        const bool throwExceptions = false;

        Teuchos::CommandLineProcessor clp( throwExceptions );

        std::string node = "";
        clp.setOption( "node", &node, "node type (serial | openmp | cuda)" );

        clp.recogniseAllOptions( false );
        switch ( clp.parse( argc, argv, NULL ) )
        {
        case Teuchos::CommandLineProcessor::PARSE_ERROR:
            return EXIT_FAILURE;
        case Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED:
        case Teuchos::CommandLineProcessor::PARSE_UNRECOGNIZED_OPTION:
        case Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL:
            break;
        }

        if ( node == "" )
        {
            typedef KokkosClassic::DefaultNode::DefaultNodeType Node;
            return main_<Node>( clp, argc, argv );
        }
        else if ( node == "serial" )
        {
#ifdef KOKKOS_HAVE_SERIAL
            typedef Kokkos::Compat::KokkosSerialWrapperNode Node;

            return main_<Node>( clp, argc, argv );
#else
            throw std::runtime_error( "Serial node type is disabled" );
#endif
        }
        else if ( node == "openmp" )
        {
#ifdef KOKKOS_HAVE_OPENMP
            typedef Kokkos::Compat::KokkosOpenMPWrapperNode Node;

            return main_<Node>( clp, argc, argv );
#else
            throw std::runtime_error( "OpenMP node type is disabled" );
#endif
        }
        else if ( node == "cuda" )
        {
#ifdef KOKKOS_HAVE_CUDA
            typedef Kokkos::Compat::KokkosCudaWrapperNode Node;

            return main_<Node>( clp, argc, argv );
#else
            throw std::runtime_error( "CUDA node type is disabled" );
#endif
        }
        else
        {
            throw std::runtime_error( "Unrecognized node type" );
        }
        Kokkos::finalize();
    }
    TEUCHOS_STANDARD_CATCH_STATEMENTS( verbose, std::cerr, success );

    return ( success ? EXIT_SUCCESS : EXIT_FAILURE );
}
