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

#include <fstream>

#include <point_clouds.hpp>

template <typename TreeType>
void viz()
{
    using DeviceType = typename TreeType::device_type;
    using ExecutionSpace = typename DeviceType::execution_space;
    Kokkos::View<DataTransferKit::Point *, DeviceType> points( "points" );
    loadPointCloud( "/scratch/source/trilinos/release/DataTransferKit/packages/"
                    "Search/examples/point_clouds/leaf_cloud.txt",
                    points );

    TreeType bvh( points );

    std::fstream fout;

    // Print the entire tree
    std::string const prefix = "trash_";
    fout.open( prefix + "tree_all_nodes_and_edges.dot.m4", std::fstream::out );
    using TreeVisualization =
        typename DataTransferKit::Details::TreeVisualization<DeviceType>;
    using GraphvizVisitor = typename TreeVisualization::GraphvizVisitor;
    TreeVisualization::visitAllIterative( bvh, GraphvizVisitor{fout} );
    fout.close();

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

    for ( int i = 0; i < n_queries; ++i )
    {
        fout.open( prefix + "untouched_" + std::to_string( i ) +
                       "_nearest_traversal.dot.m4",
                   std::fstream::out );
        TreeVisualization::visit( bvh, queries( i ), GraphvizVisitor{fout} );
        fout.close();
    }
}

int main( int argc, char *argv[] )
{
    Kokkos::initialize( argc, argv );

    using Serial = Kokkos::Compat::KokkosSerialWrapperNode::device_type;
    using Tree = DataTransferKit::BVH<Serial>;
    viz<Tree>();

    Kokkos::finalize();

    return 0;
}
