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

#include <DTK_DetailsTreeVisualization.hpp>
#include <DTK_LinearBVH.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_DefaultNode.hpp>

#include <algorithm>
#include <fstream>
#include <random>

#include <point_clouds.hpp>

template <typename View>
void printPointCloud( View points, std::ostream &os )
{
    auto const n = points.extent_int( 0 );
    for ( int i = 0; i < n; ++i )
        os << "\\node[leaf] at (" << points( i )[0] << "," << points( i )[1]
           << ") {\\textbullet};\n";
}

template <typename TreeType>
void viz()
{
    using DeviceType = typename TreeType::device_type;
    using ExecutionSpace = typename DeviceType::execution_space;
    Kokkos::View<DataTransferKit::Point *, DeviceType> points( "points" );
    loadPointCloud( "leaf_cloud.txt", points );

    TreeType bvh( points );

    using TreeVisualization =
        typename DataTransferKit::Details::TreeVisualization<DeviceType>;
    using TikZVisitor = typename TreeVisualization::TikZVisitor;
    using GraphvizVisitor = typename TreeVisualization::GraphvizVisitor;

    std::string const prefix = "trash_";

    int const n_neighbors = 10;
    int const n_queries = bvh.size();
    Kokkos::View<DataTransferKit::Nearest<DataTransferKit::Point> *, DeviceType>
        queries( "queries", n_queries );
    Kokkos::parallel_for( Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
                          KOKKOS_LAMBDA( int i ) {
                              queries( i ) = DataTransferKit::nearest(
                                  points( i ), n_neighbors );
                          } );
    Kokkos::fence();

    auto performQueries = [&bvh, &queries]( std::string const &p,
                                            std::string const &s ) {
        std::ofstream fout;
        for ( int i = 0; i < queries.extent_int( 0 ); ++i )
        {
            std::string const fname = p + std::to_string( i ) + s;
            fout.open( fname, std::fstream::out );
            TreeVisualization::visit( bvh, queries( i ),
                                      GraphvizVisitor{fout} );
            fout.close();
        }
    };

    std::fstream fout;

    // Print the point cloud
    fout.open( prefix + "points.tex", std::fstream::out );
    printPointCloud( points, fout );
    fout.close();

    // Print the bounding volume hierarchy
    fout.open( prefix + "bounding_volumes.tex", std::fstream::out );
    TreeVisualization::visitAllIterative( bvh, TikZVisitor{fout} );
    fout.close();

    // Print the entire tree
    fout.open( prefix + "tree_all_nodes_and_edges.dot.m4", std::fstream::out );
    TreeVisualization::visitAllIterative( bvh, GraphvizVisitor{fout} );
    fout.close();

    std::string const suffix = "_nearest_traversal.dot.m4";
    performQueries( prefix + "untouched_", suffix );

    // Shuffle the queries
    std::random_device rd;
    std::mt19937 g( rd() );
    std::shuffle( queries.data(), queries.data() + queries.size(), g );
    performQueries( prefix + "shuffled_", suffix );

    // Sort them
    auto permute = DataTransferKit::Details::BatchedQueries<
        DeviceType>::sortQueriesAlongZOrderCurve( bvh.bounds(), queries );
    queries =
        DataTransferKit::Details::BatchedQueries<DeviceType>::applyPermutation(
            permute, queries );
    performQueries( prefix + "sorted_", suffix );
}

// FIXME version of Kokkos in the CI/dev base image is too old and does not have
// scope guards so we define our own...
struct KokkosScopeGuard
{
    KokkosScopeGuard( Kokkos::InitArguments const &args )
    {
        Kokkos::initialize( args );
    }
    ~KokkosScopeGuard() { Kokkos::finalize(); }
};

int main( int argc, char *argv[] )
{
    Kokkos::InitArguments args;
    args.disable_warnings = true;
    KokkosScopeGuard guard( args );

    using Serial = Kokkos::Compat::KokkosSerialWrapperNode::device_type;
    using Tree = DataTransferKit::BVH<Serial>;
    viz<Tree>();

    return 0;
}
