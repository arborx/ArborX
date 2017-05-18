/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#include <DTK_DetailsPredicate.hpp>
#include <DTK_DetailsTreeTraversal.hpp>

#include <DTK_LinearBVH.hpp>
#include <details/DTK_DetailsTreeTraversal.hpp>

#include <Teuchos_UnitTestHarness.hpp>

#include <boost/geometry/index/rtree.hpp>

#include <algorithm>
#include <bitset>
#include <iostream>
#include <random>

namespace details = DataTransferKit::Details;

template <typename DeviceType>
class FillBoxes
{
  public:
    KOKKOS_INLINE_FUNCTION
    FillBoxes( Kokkos::View<DataTransferKit::Box *, DeviceType> boxes )
        : _boxes( boxes )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i ) const
    {
        if ( i == 0 )
            _boxes[0] = {0, 0, 0, 0, 0, 0};
        else
            _boxes[1] = {1, 1, 1, 1, 1, 1};
    }

  private:
    Kokkos::View<DataTransferKit::Box *, DeviceType> _boxes;
};

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( LinearBVH, tag_dispatching, NO )
{
    using DeviceType = typename DataTransferKit::BVH<NO>::DeviceType;
    using ExecutionSpace = typename DeviceType::execution_space;
    int const n = 2;
    Kokkos::View<DataTransferKit::Box *, DeviceType> boxes( "boxes", n );
    FillBoxes<DeviceType> fill_boxes_functor( boxes );
    Kokkos::parallel_for( "file_boxes_functor",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                          fill_boxes_functor );
    Kokkos::fence();

    DataTransferKit::BVH<NO> bvh( boxes );
    auto do_nothing = KOKKOS_LAMBDA( int ){};
    DataTransferKit::Point p1 = {0., 0., 0.};
    details::TreeTraversal<NO>::query( bvh, details::nearest( p1, 1 ),
                                       do_nothing );

    details::TreeTraversal<NO>::query( bvh, details::within( p1, 0.5 ),
                                       do_nothing );
}

template <typename NO>
class FillBoundingBoxes
{
  public:
    using DeviceType = typename DataTransferKit::BVH<NO>::DeviceType;

    FillBoundingBoxes(
        Kokkos::View<DataTransferKit::Box *, DeviceType> bounding_boxes,
        double Lx, double Ly, double Lz, double eps, int nx, int ny, int nz )
        : _bounding_boxes( bounding_boxes )
        , _Lx( Lx )
        , _Ly( Ly )
        , _Lz( Lz )
        , _nx( nx )
        , _ny( ny )
        , _nz( nz )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i ) const
    {
        for ( int j = 0; j < _ny; ++j )
            for ( int k = 0; k < _nz; ++k )
            {
                _bounding_boxes[i + j * _nx + k * ( _nx * _ny )] = {
                    i * _Lx / ( _nx - 1 ) - _eps, i * _Lx / ( _nx - 1 ) + _eps,
                    j * _Ly / ( _ny - 1 ) - _eps, j * _Ly / ( _ny - 1 ) + _eps,
                    k * _Lz / ( _nz - 1 ) - _eps, k * _Lz / ( _nz - 1 ) + _eps,
                };
            }
    }

  private:
    Kokkos::View<DataTransferKit::Box *, DeviceType> _bounding_boxes;
    double _Lx;
    double _Ly;
    double _Lz;
    double _eps;
    int const _nx;
    int const _ny;
    int const _nz;
};

template <typename NO>
class CheckIdentity
{
  public:
    using DeviceType = typename DataTransferKit::BVH<NO>::DeviceType;
    CheckIdentity(
        Kokkos::View<DataTransferKit::Box *, DeviceType> bounding_boxes,
        DataTransferKit::BVH<NO> bvh,
        Kokkos::View<int * [2], DeviceType> identity )
        : _bounding_boxes( bounding_boxes )
        , _bvh( bvh )
        , _identity( identity )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i ) const
    {
        unsigned int constexpr max_n_indices = 10;
        int indices[max_n_indices];
        unsigned int n_indices = 0;
        details::spatial_query( _bvh, details::overlap( _bounding_boxes[i] ),
                                indices, n_indices, max_n_indices );
        _identity( i, 0 ) = n_indices;
        _identity( i, 1 ) = indices[0];
    }

  private:
    Kokkos::View<DataTransferKit::Box *, DeviceType> _bounding_boxes;
    DataTransferKit::BVH<NO> _bvh;
    Kokkos::View<int * [2], DeviceType> _identity;
};

template <typename NO>
class CheckFirstNeighbor
{
  public:
    using DeviceType = typename DataTransferKit::BVH<NO>::DeviceType;
    CheckFirstNeighbor(
        Kokkos::View<DataTransferKit::Box *, DeviceType> bounding_boxes,
        DataTransferKit::BVH<NO> bvh,
        Kokkos::View<int * [2], DeviceType> first_neighbor, int nx, int ny,
        int nz )
        : _bounding_boxes( bounding_boxes )
        , _bvh( bvh )
        , _first_neighbor( first_neighbor )
        , _nx( nx )
        , _ny( ny )
        , _nz( nz )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i ) const
    {
        for ( int j = 0; j < _ny; ++j )
            for ( int k = 0; k < _nz; ++k )
            {
                int const index = i + j * _nx + k * ( _nx * _ny );
                unsigned int constexpr max_n_indices = 10000;
                int indices[max_n_indices];
                unsigned int n_indices = 0;
                details::spatial_query(
                    _bvh, details::overlap( _bounding_boxes[index] ), indices,
                    n_indices, max_n_indices );
                _first_neighbor( index, 0 ) = n_indices;
                // Only check the first element because we don't know how many
                // elements there are when we build the View. To check the other
                // points, we need to first compute all the points using Boost.
                // Then, we need to copy the points in a View and create another
                // view with the offset.
                _first_neighbor( index, 1 ) = indices[0];
            }
    }

  private:
    Kokkos::View<DataTransferKit::Box *, DeviceType> _bounding_boxes;
    DataTransferKit::BVH<NO> _bvh;
    Kokkos::View<int * [2], DeviceType> _first_neighbor;
    int _nx;
    int _ny;
    int _nz;
};

template <typename NO>
class CheckRandom
{
  public:
    using DeviceType = typename DataTransferKit::BVH<NO>::DeviceType;
    CheckRandom( Kokkos::View<DataTransferKit::Box *, DeviceType> aabb,
                 DataTransferKit::BVH<NO> bvh,
                 Kokkos::View<int * [2], DeviceType> random )
        : _aabb( aabb )
        , _bvh( bvh )
        , _random( random )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i ) const
    {
        unsigned int constexpr max_n_indices = 1000;
        int indices[max_n_indices];
        unsigned int n_indices = 0;
        details::spatial_query( _bvh, details::overlap( _aabb[i] ), indices,
                                n_indices, max_n_indices );
        _random( i, 0 ) = n_indices;
        _random( i, 1 ) = indices[0];
    }

  private:
    Kokkos::View<DataTransferKit::Box *, DeviceType> _aabb;
    DataTransferKit::BVH<NO> _bvh;
    Kokkos::View<int * [2], DeviceType> _random;
};

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( LinearBVH, structured_grid, NO )
{
    double Lx = 100.0;
    double Ly = 100.0;
    double Lz = 100.0;
    int constexpr nx = 11;
    int constexpr ny = 11;
    int constexpr nz = 11;
    int constexpr n = nx * ny * nz;
    double eps = 1.0e-6;

    using DeviceType = typename DataTransferKit::BVH<NO>::DeviceType;
    using ExecutionSpace = typename DeviceType::execution_space;
    Kokkos::View<DataTransferKit::Box *, DeviceType> bounding_boxes(
        "bounding_boxes", n );
    FillBoundingBoxes<NO> fill_bounding_boxes( bounding_boxes, Lx, Ly, Lz, eps,
                                               nx, ny, nz );
    Kokkos::parallel_for( "fill_bounding_boxes",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, nx ),
                          fill_bounding_boxes );
    Kokkos::fence();

    DataTransferKit::BVH<NO> bvh( bounding_boxes );

    // (i) use same objects for the queries than the objects we constructed the
    // BVH
    Kokkos::View<int * [2], DeviceType> identity( "identity", n );
    CheckIdentity<NO> check_identity( bounding_boxes, bvh, identity );

    Kokkos::parallel_for( "check_identity",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                          check_identity );
    Kokkos::fence();

    auto identity_host = Kokkos::create_mirror_view( identity );
    Kokkos::deep_copy( identity_host, identity );
    // we expect the collision list to be diag(0, 1, ..., nx*ny*nz-1)
    for ( int i = 0; i < n; ++i )
    {
        TEST_EQUALITY( identity_host( i, 0 ), 1 );
        TEST_EQUALITY( identity_host( i, 1 ), i );
    }

    // (ii) use bounding boxes that overlap with first neighbors
    // Compute the reference solution.

    auto bounding_boxes_host = Kokkos::create_mirror_view( bounding_boxes );
    std::function<int( int, int, int )> ind = [nx, ny, nz](
        int i, int j, int k ) { return i + j * nx + k * ( nx * ny ); };
    std::vector<std::set<int>> ref( n );
    for ( int i = 0; i < nx; ++i )
        for ( int j = 0; j < ny; ++j )
            for ( int k = 0; k < nz; ++k )
            {
                int const index = ind( i, j, k );
                // bounding box around nodes of the structured grid will overlap
                // with neighboring nodes
                bounding_boxes_host[index] = {
                    ( i - 1 ) * Lx / ( nx - 1 ), ( i + 1 ) * Lx / ( nx - 1 ),
                    ( j - 1 ) * Ly / ( ny - 1 ), ( j + 1 ) * Ly / ( ny - 1 ),
                    ( k - 1 ) * Lz / ( nz - 1 ), ( k + 1 ) * Lz / ( nz - 1 ),
                };
                // fill in reference solution to check againt the collision list
                // computed during the tree traversal
                if ( ( i > 0 ) && ( j > 0 ) && ( k > 0 ) )
                    ref[index].emplace( ind( i - 1, j - 1, k - 1 ) );
                if ( ( i > 0 ) && ( k > 0 ) )
                    ref[index].emplace( ind( i - 1, j, k - 1 ) );
                if ( ( i > 0 ) && ( j < ny - 1 ) && ( k > 0 ) )
                    ref[index].emplace( ind( i - 1, j + 1, k - 1 ) );
                if ( ( i > 0 ) && ( j > 0 ) )
                    ref[index].emplace( ind( i - 1, j - 1, k ) );
                if ( i > 0 )
                    ref[index].emplace( ind( i - 1, j, k ) );
                if ( ( i > 0 ) && ( j < ny - 1 ) )
                    ref[index].emplace( ind( i - 1, j + 1, k ) );
                if ( ( i > 0 ) && ( j > 0 ) && ( k < nz - 1 ) )
                    ref[index].emplace( ind( i - 1, j - 1, k + 1 ) );
                if ( ( i > 0 ) && ( k < nz - 1 ) )
                    ref[index].emplace( ind( i - 1, j, k + 1 ) );
                if ( ( i > 0 ) && ( j < ny - 1 ) && ( k < nz - 1 ) )
                    ref[index].emplace( ind( i - 1, j + 1, k + 1 ) );

                if ( ( j > 0 ) && ( k > 0 ) )
                    ref[index].emplace( ind( i, j - 1, k - 1 ) );
                if ( k > 0 )
                    ref[index].emplace( ind( i, j, k - 1 ) );
                if ( ( j < ny - 1 ) && ( k > 0 ) )
                    ref[index].emplace( ind( i, j + 1, k - 1 ) );
                if ( j > 0 )
                    ref[index].emplace( ind( i, j - 1, k ) );
                if ( true )
                    ref[index].emplace( ind( i, j, k ) );
                if ( j < ny - 1 )
                    ref[index].emplace( ind( i, j + 1, k ) );
                if ( ( j > 0 ) && ( k < nz - 1 ) )
                    ref[index].emplace( ind( i, j - 1, k + 1 ) );
                if ( k < nz - 1 )
                    ref[index].emplace( ind( i, j, k + 1 ) );
                if ( ( j < ny - 1 ) && ( k < nz - 1 ) )
                    ref[index].emplace( ind( i, j + 1, k + 1 ) );

                if ( ( i < nx - 1 ) && ( j > 0 ) && ( k > 0 ) )
                    ref[index].emplace( ind( i + 1, j - 1, k - 1 ) );
                if ( ( i < nx - 1 ) && ( k > 0 ) )
                    ref[index].emplace( ind( i + 1, j, k - 1 ) );
                if ( ( i < nx - 1 ) && ( j < ny - 1 ) && ( k > 0 ) )
                    ref[index].emplace( ind( i + 1, j + 1, k - 1 ) );
                if ( ( i < nx - 1 ) && ( j > 0 ) )
                    ref[index].emplace( ind( i + 1, j - 1, k ) );
                if ( i < nx - 1 )
                    ref[index].emplace( ind( i + 1, j, k ) );
                if ( ( i < nx - 1 ) && ( j < ny - 1 ) )
                    ref[index].emplace( ind( i + 1, j + 1, k ) );
                if ( ( i < nx - 1 ) && ( j > 0 ) && ( k < nz - 1 ) )
                    ref[index].emplace( ind( i + 1, j - 1, k + 1 ) );
                if ( ( i < nx - 1 ) && ( k < nz - 1 ) )
                    ref[index].emplace( ind( i + 1, j, k + 1 ) );
                if ( ( i < nx - 1 ) && ( j < ny - 1 ) && ( k < nz - 1 ) )
                    ref[index].emplace( ind( i + 1, j + 1, k + 1 ) );
            }

    Kokkos::deep_copy( bounding_boxes, bounding_boxes_host );
    Kokkos::View<int * [2], DeviceType> first_neighbor( "first_neighbor", n );

    CheckFirstNeighbor<NO> check_first_neighbor( bounding_boxes, bvh,
                                                 first_neighbor, nx, ny, nz );

    Kokkos::parallel_for( "check_first_neighbor",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, nx ),
                          check_first_neighbor );
    Kokkos::fence();

    auto first_neighbor_host = Kokkos::create_mirror_view( first_neighbor );
    Kokkos::deep_copy( first_neighbor_host, first_neighbor );

    for ( int i = 0; i < nx; ++i )
        for ( int j = 0; j < ny; ++j )
            for ( int k = 0; k < nz; ++k )
            {
                int index = ind( i, j, k );
                TEST_EQUALITY( first_neighbor_host( index, 0 ),
                               static_cast<int>( ref[index].size() ) );
                TEST_EQUALITY(
                    ref[index].count( first_neighbor_host( index, 1 ) ), 1 );
            }

    // (iii) use random points
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution_x( 0.0, Lz );
    std::uniform_real_distribution<double> distribution_y( 0.0, Ly );
    std::uniform_real_distribution<double> distribution_z( 0.0, Lz );

    int nn = 1000;
    int count = 0; // drop point if mapped into [0.5-eps], 0.5+eps]^3
    Kokkos::View<DataTransferKit::Box *, ExecutionSpace> aabb( "aabb", nn );
    auto aabb_host = Kokkos::create_mirror_view( aabb );
    std::vector<int> indices( nn );
    for ( int l = 0; l < nn; ++l )
    {
        double x = distribution_x( generator );
        double y = distribution_y( generator );
        double z = distribution_z( generator );
        aabb_host[l] = {
            x - 0.5 * Lx / ( nx - 1 ), x + 0.5 * Lx / ( nx - 1 ),
            y - 0.5 * Ly / ( ny - 1 ), y + 0.5 * Ly / ( ny - 1 ),
            z - 0.5 * Lz / ( nz - 1 ), z + 0.5 * Lz / ( nz - 1 ),
        };

        int i = std::round( x / Lx * ( nx - 1 ) );
        int j = std::round( y / Ly * ( ny - 1 ) );
        int k = std::round( z / Lz * ( nz - 1 ) );
        // drop point if it the bounding box is going to overlap with more than
        // one bounding box
        if ( ( std::abs( x / Lx * ( nx - 1 ) -
                         std::floor( x / Lx * ( nx - 1 ) ) - 0.5 ) < eps ) ||
             ( std::abs( y / Ly * ( ny - 1 ) -
                         std::floor( y / Ly * ( ny - 1 ) ) - 0.5 ) < eps ) ||
             ( std::abs( z / Lz * ( nz - 1 ) -
                         std::floor( z / Lz * ( nz - 1 ) ) - 0.5 ) < eps ) )
        {
            ++count;
            continue;
        }
        // Save the indices for the check
        indices[l] = ind( i, j, k );
    }

    Kokkos::deep_copy( aabb, aabb_host );
    Kokkos::View<int * [2], DeviceType> random( "random", nn );
    CheckRandom<NO> check_random( aabb, bvh, random );

    Kokkos::parallel_for( "check_random",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, nn ),
                          check_random );
    Kokkos::fence();
    auto random_host = Kokkos::create_mirror_view( random );
    Kokkos::deep_copy( random_host, random );

    for ( int i = 0; i < nn; ++i )
    {
        TEST_EQUALITY( random_host( i, 0 ), 1 );
        TEST_EQUALITY( random_host( i, 1 ), indices[i] );
    }

    // make sure we did not drop all points
    TEST_COMPARE( count, <, n );
}

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

template <typename NO>
class RandomWithinLambda
{
  public:
    using DeviceType = typename DataTransferKit::BVH<NO>::DeviceType;
    using ExecutionSpace = typename DeviceType::execution_space;

    RandomWithinLambda( Kokkos::View<double * [3], ExecutionSpace> point_coords,
                        Kokkos::View<double *, ExecutionSpace> radii,
                        Kokkos::View<int * [2], ExecutionSpace> within_n_pts,
                        DataTransferKit::BVH<NO> bvh )
        : _point_coords( point_coords )
        , _radii( radii )
        , _within_n_pts( within_n_pts )
        , _bvh( bvh )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i ) const
    {
        details::Within within_predicate( {_point_coords( i, 0 ),
                                           _point_coords( i, 1 ),
                                           _point_coords( i, 2 )},
                                          _radii( i ) );
        unsigned int constexpr max_n_indices = 10000;
        int indices[max_n_indices];
        unsigned int n_indices = 0;
        details::spatial_query( _bvh, within_predicate, indices, n_indices,
                                max_n_indices );
        _within_n_pts( i, 0 ) = n_indices;
        _within_n_pts( i, 1 ) = indices[0];
    }

  private:
    Kokkos::View<double * [3], ExecutionSpace> _point_coords;
    Kokkos::View<double *, ExecutionSpace> _radii;
    Kokkos::View<int * [2], ExecutionSpace> _within_n_pts;
    DataTransferKit::BVH<NO> _bvh;
};

template <typename NO>
class RandomNearestLambda
{
  public:
    using DeviceType = typename DataTransferKit::BVH<NO>::DeviceType;
    using ExecutionSpace = typename DeviceType::execution_space;

    RandomNearestLambda(
        Kokkos::View<double * [3], ExecutionSpace> point_coords,
        Kokkos::View<int * [2], ExecutionSpace> nearest_n_pts,
        Kokkos::View<int *, ExecutionSpace> k, DataTransferKit::BVH<NO> bvh )
        : _point_coords( point_coords )
        , _nearest_n_pts( nearest_n_pts )
        , _k( k )
        , _bvh( bvh )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i ) const
    {
        unsigned int constexpr max_n_indices = 1000;
        int indices[max_n_indices];
        unsigned int n_indices = 0;
        details::nearest_query( _bvh,
                                {_point_coords( i, 0 ), _point_coords( i, 1 ),
                                 _point_coords( i, 2 )},
                                _k[i], indices, n_indices, max_n_indices );
        _nearest_n_pts( i, 0 ) = n_indices;
        _nearest_n_pts( i, 1 ) = indices[0];
    }

  private:
    Kokkos::View<double * [3], ExecutionSpace> _point_coords;
    Kokkos::View<int * [2], ExecutionSpace> _nearest_n_pts;
    Kokkos::View<int *, ExecutionSpace> _k;
    DataTransferKit::BVH<NO> _bvh;
};

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( LinearBVH, rtree, NO )
{
    namespace bg = boost::geometry;
    namespace bgi = boost::geometry::index;
    using BPoint = bg::model::point<double, 3, bg::cs::cartesian>;
    using RTree = bgi::rtree<std::pair<BPoint, int>, bgi::linear<16>>;

    // contruct a cloud of points (nodes of a structured grid)
    double Lx = 10.0;
    double Ly = 10.0;
    double Lz = 10.0;
    int nx = 11;
    int ny = 11;
    int nz = 11;
    auto cloud = make_stuctured_cloud( Lx, Ly, Lz, nx, ny, nz );
    int n = cloud.size();

    // create a R-tree to compare radius search results against
    RTree rtree;
    for ( int i = 0; i < n; ++i )
    {
        auto const &point = cloud[i];
        double x = std::get<0>( point );
        double y = std::get<1>( point );
        double z = std::get<2>( point );
        rtree.insert( std::make_pair( BPoint( x, y, z ), i ) );
    }
    using DeviceType = typename DataTransferKit::BVH<NO>::DeviceType;
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
    // compare our solution against Boost R-tree
    int const n_points = 100;
    auto queries = make_random_cloud( Lx, Ly, Lz, n_points );
    using ExecutionSpace = typename DeviceType::execution_space;
    Kokkos::View<double * [3], ExecutionSpace> point_coords( "point_coords",
                                                             n_points );
    auto point_coords_host = Kokkos::create_mirror_view( point_coords );
    Kokkos::View<double *, ExecutionSpace> radii( "radii", n_points );
    auto radii_host = Kokkos::create_mirror_view( radii );
    Kokkos::View<int * [2], ExecutionSpace> within_n_pts( "within_n_pts",
                                                          n_points );
    Kokkos::View<int * [2], ExecutionSpace> nearest_n_pts( "nearest_n_pts",
                                                           n_points );
    Kokkos::View<int *, ExecutionSpace> k( "distribution_k", n_points );
    auto k_host = Kokkos::create_mirror_view( k );
    std::vector<std::vector<std::pair<BPoint, int>>> returned_values_within(
        n_points );
    std::vector<std::vector<std::pair<BPoint, int>>> returned_values_nearest(
        n_points );
    // use random radius for the search and random number k of for the kNN
    // search
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution_radius(
        0.0, std::sqrt( Lx * Lx + Ly * Ly + Lz * Lz ) );
    std::uniform_int_distribution<int> distribution_k(
        1, std::floor( sqrt( nx * nx + ny * ny + nz * nz ) ) );
    for ( unsigned int i = 0; i < n_points; ++i )
    {
        auto const &point = queries[i];
        double x = std::get<0>( point );
        double y = std::get<1>( point );
        double z = std::get<2>( point );
        BPoint centroid( x, y, z );
        radii_host[i] = distribution_radius( generator );
        k_host[i] = distribution_k( generator );
        double radius = radii_host[i];

        // COMMENT: Did not implement proper radius search yet
        // This use available tree traversal for axis-aligned bounding box and
        // filters out candidates afterwards.
        // The coordinates of the points in the structured cloud (source) are
        // accessed directly and we use Boost to compute the distance.
        point_coords_host( i, 0 ) = x;
        point_coords_host( i, 1 ) = y;
        point_coords_host( i, 2 ) = z;

        // use the R-tree to obtain a reference solution
        rtree.query( bgi::satisfies( [centroid, radius](
                         std::pair<BPoint, int> const &val ) {
                         return bg::distance( centroid, val.first ) <= radius;
                     } ),
                     std::back_inserter( returned_values_within[i] ) );

        // k nearest neighbors
        rtree.query( bgi::nearest( BPoint( x, y, z ), k_host[i] ),
                     std::back_inserter( returned_values_nearest[i] ) );
    }

    Kokkos::deep_copy( point_coords, point_coords_host );
    Kokkos::deep_copy( radii, radii_host );
    Kokkos::deep_copy( k, k_host );

    Kokkos::View<int *, ExecutionSpace> offset( "offset", n_points + 1 );
    Kokkos::parallel_for(
        "first_pass", Kokkos::RangePolicy<ExecutionSpace>( 0, n_points ),
        KOKKOS_LAMBDA( int i ) {
            offset( i ) = 0;
            int count = details::TreeTraversal<NO>::query(
                bvh,
                details::nearest( {point_coords( i, 0 ), point_coords( i, 1 ),
                                   point_coords( i, 2 )},
                                  k( i ) ),
                [offset, i]( int index ) { offset( i )++; } );
            assert( count == offset( i ) );
        } );
    Kokkos::fence();

    Kokkos::parallel_scan(
        "compute_offset",
        Kokkos::RangePolicy<ExecutionSpace>( 0, n_points + 1 ),
        KOKKOS_LAMBDA( int i, int &update, bool final_pass ) {
            int const offset_i = offset( i );
            if ( final_pass )
                offset( i ) = update;
            update += offset_i;
        } );
    Kokkos::fence();

    // COMMENT: I feel like there is more efficient way to copy a single element
    // of a View on the device to the host...
    auto total_count = Kokkos::subview( offset, n_points );
    auto total_count_host = Kokkos::create_mirror_view( total_count );
    Kokkos::deep_copy( total_count_host, total_count );
    Kokkos::View<int *, ExecutionSpace> indices( "indices", total_count( 0 ) );
    Kokkos::parallel_for(
        "second_pass", Kokkos::RangePolicy<ExecutionSpace>( 0, n_points ),
        KOKKOS_LAMBDA( int i ) {
            int count = 0;
            details::TreeTraversal<NO>::query(
                bvh,
                details::nearest( {point_coords( i, 0 ), point_coords( i, 1 ),
                                   point_coords( i, 2 )},
                                  k( i ) ),
                [indices, offset, i, &count]( int index ) {
                    indices( offset( i ) + count++ ) = index;
                } );
            // assert( count == offset( i ) );
        } );
    Kokkos::fence();
    auto indices_host = Kokkos::create_mirror_view( indices );
    auto offset_host = Kokkos::create_mirror_view( offset );
    Kokkos::deep_copy( indices_host, indices );
    Kokkos::deep_copy( offset_host, offset );

    for ( int i = 0; i < n_points; ++i )
    {
        auto const &ref = returned_values_nearest[i];
        std::set<int> ref_ids;
        for ( auto const &id : ref )
            ref_ids.emplace( id.second );
        for ( int j = offset_host( i ); j < offset_host( i + 1 ); ++j )
        {
            TEST_ASSERT( ref_ids.count( indices_host( j ) ) != 0 );
        }
    }

    RandomWithinLambda<NO> random_within_lambda( point_coords, radii,
                                                 within_n_pts, bvh );

    Kokkos::parallel_for( "random_within",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n_points ),
                          random_within_lambda );
    Kokkos::fence();

    auto within_n_pts_host = Kokkos::create_mirror_view( within_n_pts );
    Kokkos::deep_copy( within_n_pts_host, within_n_pts );

    for ( int i = 0; i < n_points; ++i )
    {
        auto const &ref = returned_values_within[i];
        TEST_EQUALITY( within_n_pts_host( i, 0 ),
                       static_cast<int>( ref.size() ) );
        std::set<int> ref_ids;
        for ( auto const &id : ref )
            ref_ids.emplace( id.second );

        if ( ref.size() > 0 )
            TEST_EQUALITY( ref_ids.count( within_n_pts_host( i, 1 ) ), 1 );
    }

    RandomNearestLambda<NO> random_nearest_lambda( point_coords, nearest_n_pts,
                                                   k, bvh );

    Kokkos::parallel_for( "random_nearest",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n_points ),
                          random_nearest_lambda );
    Kokkos::fence();

    auto nearest_n_pts_host = Kokkos::create_mirror_view( nearest_n_pts );
    Kokkos::deep_copy( nearest_n_pts_host, nearest_n_pts );

    for ( int i = 0; i < n_points; ++i )
    {
        auto const &ref = returned_values_nearest[i];
        TEST_EQUALITY( nearest_n_pts_host( i, 0 ),
                       static_cast<int>( ref.size() ) );
        std::set<int> ref_ids;
        for ( auto const &id : ref )
            ref_ids.emplace( id.second );

        TEST_EQUALITY( ref_ids.count( nearest_n_pts_host( i, 1 ) ), 1 );
    }
}

// Include the test macros.
#include "DataTransferKitSearch_ETIHelperMacros.h"

// Create the test group
#define UNIT_TEST_GROUP( NODE )                                                \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( LinearBVH, tag_dispatching, NODE )   \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( LinearBVH, structured_grid, NODE )   \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( LinearBVH, rtree, NODE )

// Demangle the types
DTK_ETI_MANGLING_TYPEDEFS()

// Instantiate the tests
DTK_INSTANTIATE_N( UNIT_TEST_GROUP )
