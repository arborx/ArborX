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
#include <tuple>

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
            _boxes[0] = {{0, 0, 0, 0, 0, 0}};
        else
            _boxes[1] = {{1, 1, 1, 1, 1, 1}};
    }

  private:
    Kokkos::View<DataTransferKit::Box *, DeviceType> _boxes;
};

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( LinearBVH, tag_dispatching, DeviceType )
{
    using ExecutionSpace = typename DeviceType::execution_space;
    int const n = 2;
    Kokkos::View<DataTransferKit::Box *, DeviceType> boxes( "boxes", n );
    FillBoxes<DeviceType> fill_boxes_functor( boxes );
    Kokkos::parallel_for( "file_boxes_functor",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                          fill_boxes_functor );
    Kokkos::fence();

    DataTransferKit::BVH<DeviceType> bvh( boxes );
    auto do_nothing_1 = KOKKOS_LAMBDA( int ){};
    auto do_nothing_2 = KOKKOS_LAMBDA( int, double ){};
    DataTransferKit::Point p1 = {{0., 0., 0.}};
    details::TreeTraversal<DeviceType>::query( bvh, details::nearest( p1, 1 ),
                                               do_nothing_2 );

    details::TreeTraversal<DeviceType>::query( bvh, details::within( p1, 0.5 ),
                                               do_nothing_1 );
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( LinearBVH, bounds, DeviceType )
{
    Kokkos::View<DataTransferKit::Box *, DeviceType> boxes( "boxes", 2 );
    auto boxes_host = Kokkos::create_mirror_view( boxes );
    boxes_host( 0 ) = DataTransferKit::Box( {2., 2., 4., 4., 6., 6.} );
    boxes_host( 1 ) = DataTransferKit::Box( {1., 1., 3., 3., 5., 5.} );
    Kokkos::deep_copy( boxes, boxes_host );
    DataTransferKit::BVH<DeviceType> bvh( boxes );
    auto bounds = bvh.bounds();
    TEST_EQUALITY( bounds[0], 1.0 );
    TEST_EQUALITY( bounds[1], 2.0 );
    TEST_EQUALITY( bounds[2], 3.0 );
    TEST_EQUALITY( bounds[3], 4.0 );
    TEST_EQUALITY( bounds[4], 5.0 );
    TEST_EQUALITY( bounds[5], 6.0 );
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( LinearBVH, queries, DeviceType )
{
    Kokkos::View<DataTransferKit::Box *, DeviceType> boxes( "boxes", 2 );
    auto boxes_host = Kokkos::create_mirror_view( boxes );
    boxes_host( 0 ) = DataTransferKit::Box( {0., 0., 0., 0., 0., 0.} );
    boxes_host( 1 ) = DataTransferKit::Box( {1., 1., 1., 1., 1., 1.} );
    Kokkos::deep_copy( boxes, boxes_host );
    DataTransferKit::BVH<DeviceType> bvh( boxes );

    // `out` and `successs` need to be captured by reference  They come from the
    // test assertion macros expansion.
    auto check_results = [&bvh, &out, &success](
        std::vector<DataTransferKit::Box> const &overlap_boxes,
        std::vector<int> const &indices_ref,
        std::vector<int> const &offset_ref ) {
        Kokkos::View<DataTransferKit::Details::Overlap *, DeviceType> queries(
            "queries", overlap_boxes.size() );
        auto queries_host = Kokkos::create_mirror_view( queries );
        for ( int i = 0; i < queries.extent_int( 0 ); ++i )
            queries_host( i ) =
                DataTransferKit::Details::Overlap( overlap_boxes[i] );
        Kokkos::deep_copy( queries, queries_host );

        Kokkos::View<int *, DeviceType> indices( "indices" );
        Kokkos::View<int *, DeviceType> offset( "offset" );
        bvh.query( queries, indices, offset );

        auto indices_host = Kokkos::create_mirror_view( indices );
        deep_copy( indices_host, indices );
        auto offset_host = Kokkos::create_mirror_view( offset );
        deep_copy( offset_host, offset );

        TEST_COMPARE_ARRAYS( indices_host, indices_ref );
        TEST_COMPARE_ARRAYS( offset_host, offset_ref );
    };

    // single query overlap with nothing
    check_results( {DataTransferKit::Box()}, {}, {0, 0} );

    // single query overlap with both
    check_results( {DataTransferKit::Box( {0., 1., 0., 1., 0., 1.} )}, {1, 0},
                   {0, 2} );

    // single query overlap with only one
    check_results( {DataTransferKit::Box( {0.5, 1.5, 0.5, 1.5, 0.5, 1.5} )},
                   {1}, {0, 1} );

    // a couple queries both overlap with nothing
    check_results( {DataTransferKit::Box(), DataTransferKit::Box()}, {},
                   {0, 0, 0} );

    // a couple queries first overlap with nothing second with only one
    check_results( {DataTransferKit::Box(),
                    DataTransferKit::Box( {0., 0., 0., 0., 0., 0.} )},
                   {0}, {0, 0, 1} );

    // no query
    check_results( {}, {}, {0} );
    // QUESTION: does it make sense to have len( offset ) = 1 ???
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( LinearBVH, nearest_queries, DeviceType )
{
    Kokkos::View<DataTransferKit::Box *, DeviceType> boxes( "boxes", 2 );
    auto boxes_host = Kokkos::create_mirror_view( boxes );
    boxes_host( 0 ) = DataTransferKit::Box( {0., 0., 0., 0., 0., 0.} );
    boxes_host( 1 ) = DataTransferKit::Box( {1., 1., 1., 1., 1., 1.} );
    Kokkos::deep_copy( boxes, boxes_host );
    DataTransferKit::BVH<DeviceType> bvh( boxes );

    std::vector<std::pair<DataTransferKit::Point, int>> points = {
        {{{0., 0., 0.}}, 2}, {{{1., 0., 0.}}, 4},
    };
    std::vector<int> indices_ref = {0, 1, 0, 1, -1, -1};
    std::vector<int> offset_ref = {0, 2, 6};
    double const infty = std::numeric_limits<double>::max();
    std::vector<double> distances_ref = {0.,         sqrt( 3. ), 1.,
                                         sqrt( 2. ), infty,      infty};

    Kokkos::View<DataTransferKit::Details::Nearest *, DeviceType> queries(
        "nearest_queries", points.size() );
    auto queries_host = Kokkos::create_mirror_view( queries );
    for ( int i = 0; i < queries.extent_int( 0 ); ++i )
        queries_host( i ) = DataTransferKit::Details::Nearest(
            points[i].first, points[i].second );
    Kokkos::deep_copy( queries, queries_host );

    Kokkos::View<int *, DeviceType> indices( "indices" );
    Kokkos::View<int *, DeviceType> offset( "offset" );
    Kokkos::View<double *, DeviceType> distances( "distances" );
    bvh.query( queries, indices, offset, distances );

    auto indices_host = Kokkos::create_mirror_view( indices );
    auto offset_host = Kokkos::create_mirror_view( offset );
    auto distances_host = Kokkos::create_mirror_view( distances );
    Kokkos::deep_copy( indices_host, indices );
    Kokkos::deep_copy( offset_host, offset );
    Kokkos::deep_copy( distances_host, distances );

    TEST_COMPARE_ARRAYS( indices_host, indices_ref );
    TEST_COMPARE_ARRAYS( offset_host, offset_ref );
    TEST_COMPARE_ARRAYS( distances_host, distances_ref );
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( LinearBVH, empty, DeviceType )
{
    Kokkos::View<DataTransferKit::Box *, DeviceType> boxes( "boxes", 1 );
    auto boxes_host = Kokkos::create_mirror_view( boxes );
    boxes_host( 0 ) = DataTransferKit::Box( {1., 2., 3., 4., 5., 6.} );
    Kokkos::deep_copy( boxes, boxes_host );
    DataTransferKit::BVH<DeviceType> bvh( boxes );
    TEST_ASSERT( !bvh.empty() );
    TEST_EQUALITY( bvh.size(), 1 );
    auto bounds = bvh.bounds();
    TEST_EQUALITY( bounds[0], 1.0 );
    TEST_EQUALITY( bounds[1], 2.0 );
    TEST_EQUALITY( bounds[2], 3.0 );
    TEST_EQUALITY( bounds[3], 4.0 );
    TEST_EQUALITY( bounds[4], 5.0 );
    TEST_EQUALITY( bounds[5], 6.0 );

    Kokkos::resize( boxes, 0 );
    DataTransferKit::BVH<DeviceType> empty_bvh( boxes );
    TEST_ASSERT( empty_bvh.empty() );
    TEST_EQUALITY( empty_bvh.size(), 0 );
    bounds = empty_bvh.bounds();
    DataTransferKit::Box empty_box;
    TEST_EQUALITY( bounds[0], empty_box[0] );
    TEST_EQUALITY( bounds[1], empty_box[1] );
    TEST_EQUALITY( bounds[2], empty_box[2] );
    TEST_EQUALITY( bounds[3], empty_box[3] );
    TEST_EQUALITY( bounds[4], empty_box[4] );
    TEST_EQUALITY( bounds[5], empty_box[5] );

    Kokkos::View<DataTransferKit::Details::Overlap *, DeviceType> queries(
        "queries", 2 );
    auto queries_host = Kokkos::create_mirror_view( queries );
    queries_host( 0 ) = DataTransferKit::Details::Overlap(
        DataTransferKit::Box( {0.0, 0.5, 0.0, 0.5, 0.0, 0.5} ) );
    queries_host( 1 ) = DataTransferKit::Details::Overlap(
        DataTransferKit::Box( {0.0, 10.0, 0.0, 10.0, 0.0, 10.0} ) );
    Kokkos::deep_copy( queries, queries_host );
    Kokkos::View<int *, DeviceType> indices( "indices" );
    Kokkos::View<int *, DeviceType> offset( "offset" );
    bvh.query( queries, indices, offset );

    // This helped catching a bug where we assumed that any leaf node in the
    // stack (for spatial queries) does satisfy the predicate which is not
    // true when the tree was built from only one object.  In that case
    // TreeTraversal::getRoot returns directly the leaf and we still need to
    // check the predicate before insertion in the stack.
    auto indices_host = Kokkos::create_mirror_view( indices );
    auto offset_host = Kokkos::create_mirror_view( offset );
    Kokkos::deep_copy( indices_host, indices );
    Kokkos::deep_copy( offset_host, offset );
    TEST_EQUALITY( indices_host.extent( 0 ), 1 );
    TEST_EQUALITY( indices_host( 0 ), 0 );
    TEST_EQUALITY( offset_host.extent( 0 ), 3 );
    TEST_EQUALITY( offset_host( 0 ), 0 );
    TEST_EQUALITY( offset_host( 1 ), 0 );
    TEST_EQUALITY( offset_host( 2 ), 1 );

    // empty tree won't find anything
    empty_bvh.query( queries, indices, offset );
    indices_host = Kokkos::create_mirror_view( indices );
    offset_host = Kokkos::create_mirror_view( offset );
    Kokkos::deep_copy( indices_host, indices );
    Kokkos::deep_copy( offset_host, offset );
    TEST_EQUALITY( indices_host.extent( 0 ), 0 );
    TEST_EQUALITY( offset_host.extent( 0 ), 3 );
    TEST_EQUALITY( offset_host( 0 ), 0 );
    TEST_EQUALITY( offset_host( 1 ), 0 );
    TEST_EQUALITY( offset_host( 2 ), 0 );

    TEST_ASSERT( details::TreeTraversal<DeviceType>::getRoot( bvh ) );
    TEST_ASSERT( !details::TreeTraversal<DeviceType>::getRoot( empty_bvh ) );
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( LinearBVH, structured_grid, DeviceType )
{
    double Lx = 100.0;
    double Ly = 100.0;
    double Lz = 100.0;
    int constexpr nx = 11;
    int constexpr ny = 11;
    int constexpr nz = 11;
    int constexpr n = nx * ny * nz;

    using ExecutionSpace = typename DeviceType::execution_space;

    Kokkos::View<DataTransferKit::Box *, DeviceType> bounding_boxes(
        "bounding_boxes", n );
    Kokkos::parallel_for(
        "fill_bounding_boxes", Kokkos::RangePolicy<ExecutionSpace>( 0, nx ),
        KOKKOS_LAMBDA( int i ) {
            double x, y, z;
            for ( int j = 0; j < ny; ++j )
                for ( int k = 0; k < nz; ++k )
                {
                    x = i * Lx / ( nx - 1 );
                    y = j * Ly / ( ny - 1 );
                    z = k * Lz / ( nz - 1 );
                    bounding_boxes[i + j * nx + k * ( nx * ny )] = {x, x, y,
                                                                    y, z, z};
                }
        } );
    Kokkos::fence();

    DataTransferKit::BVH<DeviceType> bvh( bounding_boxes );

    // (i) use same objects for the queries than the objects we constructed the
    // BVH
    // i-2  i-1  i  i+1
    //
    //  o    o   o   o   j+1
    //          ---
    //  o    o | x | o   j
    //          ---
    //  o    o   o   o   j-1
    //
    //  o    o   o   o   j-2
    //
    Kokkos::View<DataTransferKit::Details::Overlap *, DeviceType> queries(
        "queries", n );
    Kokkos::parallel_for(
        "fill_queries", Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
        KOKKOS_LAMBDA( int i ) {
            queries( i ) =
                DataTransferKit::Details::Overlap( bounding_boxes( i ) );
        } );
    Kokkos::fence();

    Kokkos::View<int *, DeviceType> indices( "indices", n );
    Kokkos::View<int *, DeviceType> offset( "offset", n );

    bvh.query( queries, indices, offset );

    auto indices_host = Kokkos::create_mirror_view( indices );
    Kokkos::deep_copy( indices_host, indices );
    auto offset_host = Kokkos::create_mirror_view( offset );
    Kokkos::deep_copy( offset_host, offset );

    // we expect the collision list to be diag(0, 1, ..., nx*ny*nz-1)
    for ( int i = 0; i < n; ++i )
    {
        TEST_EQUALITY( indices_host( i ), i );
        TEST_EQUALITY( offset_host( i ), i );
    }

    // (ii) use bounding boxes that overlap with first neighbors
    //
    // i-2  i-1  i  i+1
    //
    //  o    x---x---x   j+1
    //       |       |
    //  o    x   x   x   j
    //       |       |
    //  o    x---x---x   j-1
    //
    //  o    o   o   o   j-2
    //

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
    Kokkos::parallel_for(
        "fill_first_neighbors_queries",
        Kokkos::RangePolicy<ExecutionSpace>( 0, n ), KOKKOS_LAMBDA( int i ) {
            queries[i] = DataTransferKit::Details::Overlap( bounding_boxes[i] );
        } );
    Kokkos::fence();
    bvh.query( queries, indices, offset );
    Kokkos::deep_copy( indices_host, indices );
    Kokkos::deep_copy( offset_host, offset );

    for ( int i = 0; i < nx; ++i )
        for ( int j = 0; j < ny; ++j )
            for ( int k = 0; k < nz; ++k )
            {
                int index = ind( i, j, k );
                for ( int l = offset( index ); l < offset( index + 1 ); ++l )
                {
                    TEST_ASSERT( ref[index].count( indices( l ) ) != 0 );
                }
            }

    // (iii) use random points
    //
    // i-1      i      i+1
    //
    //  o       o       o   j+1
    //         -------
    //        |       |
    //        |   +   |
    //  o     | x     | o   j
    //         -------
    //
    //  o       o       o   j-1
    //
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution_x( 0.0, Lz );
    std::uniform_real_distribution<double> distribution_y( 0.0, Ly );
    std::uniform_real_distribution<double> distribution_z( 0.0, Lz );

    for ( int l = 0; l < n; ++l )
    {
        double x = distribution_x( generator );
        double y = distribution_y( generator );
        double z = distribution_z( generator );
        bounding_boxes_host( l ) = {
            x - 0.5 * Lx / ( nx - 1 ), x + 0.5 * Lx / ( nx - 1 ),
            y - 0.5 * Ly / ( ny - 1 ), y + 0.5 * Ly / ( ny - 1 ),
            z - 0.5 * Lz / ( nz - 1 ), z + 0.5 * Lz / ( nz - 1 ),
        };

        int i = std::round( x / Lx * ( nx - 1 ) );
        int j = std::round( y / Ly * ( ny - 1 ) );
        int k = std::round( z / Lz * ( nz - 1 ) );
        // Save the indices for the check
        ref[l] = {ind( i, j, k )};
    }

    Kokkos::deep_copy( bounding_boxes, bounding_boxes_host );
    Kokkos::parallel_for(
        "fill_first_neighbors_queries",
        Kokkos::RangePolicy<ExecutionSpace>( 0, n ), KOKKOS_LAMBDA( int i ) {
            queries[i] = DataTransferKit::Details::Overlap( bounding_boxes[i] );
        } );
    Kokkos::fence();
    bvh.query( queries, indices, offset );
    Kokkos::deep_copy( indices_host, indices );
    Kokkos::deep_copy( offset_host, offset );

    for ( int i = 0; i < n; ++i )
    {
        TEST_EQUALITY( offset( i ), i );
        TEST_ASSERT( ref[i].count( indices[i] ) != 0 );
    }
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
                cloud[ind( i, j, k )] = {{x, y, z}};
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
        cloud[i] = {{x, y, z}};
    }
    return cloud;
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( LinearBVH, rtree, DeviceType )
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

    DataTransferKit::BVH<DeviceType> bvh( bounding_boxes );

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

    Kokkos::View<details::Nearest *, DeviceType> nearest_queries(
        "neatest_queries", n_points );
    Kokkos::parallel_for( "register_nearest_queries",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n_points ),
                          KOKKOS_LAMBDA( int i ) {
                              nearest_queries( i ) = details::nearest(
                                  {{point_coords( i, 0 ), point_coords( i, 1 ),
                                    point_coords( i, 2 )}},
                                  k( i ) );
                          } );
    Kokkos::fence();

    Kokkos::View<details::Within *, DeviceType> within_queries(
        "within_queries", n_points );
    Kokkos::parallel_for( "register_within_queries",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n_points ),
                          KOKKOS_LAMBDA( int i ) {
                              within_queries( i ) = details::within(
                                  {{point_coords( i, 0 ), point_coords( i, 1 ),
                                    point_coords( i, 2 )}},
                                  radii( i ) );
                          } );
    Kokkos::fence();

    Kokkos::View<int *, DeviceType> offset_nearest( "offset_nearest" );
    Kokkos::View<int *, DeviceType> indices_nearest( "indices_nearest" );
    bvh.query( nearest_queries, indices_nearest, offset_nearest );

    Kokkos::View<int *, DeviceType> offset_within( "offset_within" );
    Kokkos::View<int *, DeviceType> indices_within( "indices_within" );
    bvh.query( within_queries, indices_within, offset_within );

    for ( auto data : {std::make_tuple( returned_values_nearest,
                                        indices_nearest, offset_nearest ),
                       std::make_tuple( returned_values_within, indices_within,
                                        offset_within )} )
    {
        auto returned_values = std::get<0>( data );
        auto indices = std::get<1>( data );
        auto offset = std::get<2>( data );

        auto indices_host = Kokkos::create_mirror_view( indices );
        auto offset_host = Kokkos::create_mirror_view( offset );
        Kokkos::deep_copy( indices_host, indices );
        Kokkos::deep_copy( offset_host, offset );

        for ( int i = 0; i < n_points; ++i )
        {
            auto const &ref = returned_values[i];
            std::set<int> ref_ids;
            for ( auto const &id : ref )
                ref_ids.emplace( id.second );
            for ( int j = offset_host( i ); j < offset_host( i + 1 ); ++j )
            {
                TEST_ASSERT( ref_ids.count( indices_host( j ) ) != 0 );
            }
        }
    }
}

// Include the test macros.
#include "DataTransferKitSearch_ETIHelperMacros.h"

// Create the test group
#define UNIT_TEST_GROUP( NODE )                                                \
    using DeviceType##NODE = typename NODE::device_type;                       \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( LinearBVH, tag_dispatching,          \
                                          DeviceType##NODE )                   \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( LinearBVH, bounds,                   \
                                          DeviceType##NODE )                   \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( LinearBVH, queries,                  \
                                          DeviceType##NODE )                   \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( LinearBVH, nearest_queries,          \
                                          DeviceType##NODE )                   \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( LinearBVH, empty, DeviceType##NODE ) \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( LinearBVH, structured_grid,          \
                                          DeviceType##NODE )                   \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( LinearBVH, rtree, DeviceType##NODE )

// Demangle the types
DTK_ETI_MANGLING_TYPEDEFS()

// Instantiate the tests
DTK_INSTANTIATE_N( UNIT_TEST_GROUP )
