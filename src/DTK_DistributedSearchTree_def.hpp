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

#ifndef DTK_DISTRIBUTED_SEARCH_TREE_DEF_HPP
#define DTK_DISTRIBUTED_SEARCH_TREE_DEF_HPP

#include <DTK_Box.hpp>
#include <DTK_DetailsUtils.hpp> // accumulate

#include <Teuchos_Array.hpp>
#include <Teuchos_CommHelpers.hpp>

namespace DataTransferKit
{

template <typename DeviceType>
DistributedSearchTree<DeviceType>::DistributedSearchTree(
    Teuchos::RCP<Teuchos::Comm<int> const> comm,
    Kokkos::View<Box const *, DeviceType> bounding_boxes )
    : _comm( comm )
    , _bottom_tree( bounding_boxes )
{
    int const comm_rank = _comm->getRank();
    int const comm_size = _comm->getSize();

    Kokkos::View<Box *, DeviceType> boxes(
        Kokkos::ViewAllocateWithoutInitializing( "rank_bounding_boxes" ),
        comm_size );
    // FIXME: I am not sure how to do the MPI allgather with Teuchos for data
    // living on the device so I copied to the host.
    auto boxes_host = Kokkos::create_mirror_view( boxes );
    boxes_host( comm_rank ) = _bottom_tree.bounds();

    Teuchos::Array<double> bounds( 6 * comm_size );
    Teuchos::gatherAll( *_comm, 6,
                        reinterpret_cast<double *>( &boxes_host( comm_rank ) ),
                        6 * comm_size, bounds.getRawPtr() );
    for ( int i = 0; i < comm_size; ++i )
        boxes_host( i ) = reinterpret_cast<Box const &>( bounds[6 * i] );
    Kokkos::deep_copy( boxes, boxes_host );

    _top_tree = BVH<DeviceType>( boxes );

    _bottom_tree_sizes = Kokkos::View<size_type *, DeviceType>(
        Kokkos::ViewAllocateWithoutInitializing( "leave_count_in_local_trees" ),
        comm_size );
    auto bottom_tree_sizes_host =
        Kokkos::create_mirror_view( _bottom_tree_sizes );
    auto const bottom_tree_size = _bottom_tree.size();
    Teuchos::gatherAll( *comm, 1, &bottom_tree_size, comm_size,
                        bottom_tree_sizes_host.data() );
    Kokkos::deep_copy( _bottom_tree_sizes, bottom_tree_sizes_host );

    _top_tree_size = accumulate( _bottom_tree_sizes, 0 );
}

} // namespace DataTransferKit

// Explicit instantiation macro
#define DTK_DISTRIBUTED_SEARCH_TREE_INSTANT( NODE )                            \
    template class DistributedSearchTree<typename NODE::device_type>;

#endif
