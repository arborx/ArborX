/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#ifndef DTK_DISTRIBUTED_SEARCH_TREE_DEF_HPP
#define DTK_DISTRIBUTED_SEARCH_TREE_DEF_HPP

#include <details/DTK_DetailsBox.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_CommHelpers.hpp>

namespace DataTransferKit
{

template <typename DeviceType>
DistributedSearchTree<DeviceType>::DistributedSearchTree(
    Teuchos::RCP<Teuchos::Comm<int> const> comm,
    Kokkos::View<Box const *, DeviceType> bounding_boxes )
    : _comm( comm )
    , _local_tree( bounding_boxes )
{
    int const comm_rank = _comm->getRank();
    int const comm_size = _comm->getSize();

    Kokkos::View<Box *, DeviceType> boxes( "rank_bounding_boxes", comm_size );
    // FIXME: I am not sure how to do the MPI allgather with Teuchos for data
    // living on the device so I copied to the host.
    auto boxes_host = Kokkos::create_mirror_view( boxes );
    boxes_host( comm_rank ) = _local_tree.bounds();

    Teuchos::Array<double> bounds( 6 * comm_size );
    Teuchos::gatherAll( *_comm, 6,
                        reinterpret_cast<double *>( &boxes_host( comm_rank ) ),
                        6 * comm_size, bounds.getRawPtr() );
    for ( int i = 0; i < comm_size; ++i )
        boxes_host( i ) = Box( &( bounds[6 * i] ) );
    Kokkos::deep_copy( boxes, boxes_host );

    _distributed_tree = BVH<DeviceType>( boxes );
}

template <typename DeviceType>
typename DistributedSearchTree<DeviceType>::SizeType
DistributedSearchTree<DeviceType>::size() const
{
    // Below is an attempt to achieve lazy evaluation while preserving thread
    // safety.  `_size` is initialized to an invalid state upon construction
    // and is computed only once as the sum reduce of the local tree size over
    // all processes.
    if ( _size == std::numeric_limits<SizeType>::max() )
    {
        std::lock_guard<std::mutex> guard( _mutex );
        std::call_once(
            _once_flag,
            []( Teuchos::RCP<Teuchos::Comm<int> const> const &comm,
                SizeType const &local_size,
                Teuchos::Ptr<SizeType> const &global_size ) -> void {
                Teuchos::reduceAll( *comm, Teuchos::REDUCE_SUM, local_size,
                                    global_size );
            },
            _comm, _local_tree.size(), Teuchos::ptrFromRef( _size ) );
    }
    return _size;
}

} // end namespace DataTransferKit

// Explicit instantiation macro
#define DTK_DISTRIBUTED_SEARCH_TREE_INSTANT( NODE )                            \
    template class DistributedSearchTree<typename NODE::device_type>;

#endif
