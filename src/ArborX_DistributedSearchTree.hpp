/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DISTRIBUTED_SEARCH_TREE_HPP
#define ARBORX_DISTRIBUTED_SEARCH_TREE_HPP

#include <ArborX_Box.hpp>
#include <ArborX_DetailsDistributedSearchTreeImpl.hpp>
#include <ArborX_DetailsUtils.hpp> // accumulate
#include <ArborX_LinearBVH.hpp>

#include <Kokkos_View.hpp>

#include <mpi.h>

namespace ArborX
{

/** \brief Distributed search tree
 *
 *  \note query() must be called as collective over all processes in the
 *  communicator passed to the constructor.
 */
template <typename DeviceType>
class DistributedSearchTree
{
  public:
    using bounding_volume_type = typename BVH<DeviceType>::bounding_volume_type;
    using size_type = typename BVH<DeviceType>::size_type;

    template <typename Primitives>
    DistributedSearchTree( MPI_Comm comm, Primitives const &primitives );

    /** Returns the smallest axis-aligned box able to contain all the objects
     *  stored in the tree or an invalid box if the tree is empty.
     */
    bounding_volume_type bounds() const { return _top_tree.bounds(); }

    /** Returns the global number of objects stored in the tree.
     */
    size_type size() const { return _top_tree_size; }

    /** Indicates whether the tree is empty on all processes.
     */
    bool empty() const { return size() == 0; }

    /** \brief Finds object satisfying the passed predicates (e.g. nearest to
     *  some point or overlaping with some box)
     *
     *  This query function performs a batch of spatial or k-nearest neighbors
     *  searches.  The results give indices of the objects that satisfy
     *  predicates (as given to the constructor).  They are organized in a
     *  distributed compressed row storage format.
     *
     *  \c indices stores the indices of the objects that satisfy the
     *  predicates.  \c offset stores the locations in the \c indices view that
     *  start a predicate, that is, \c queries(q) is satisfied by \c indices(o)
     *  for <code>objects(q) <= o < objects(q+1)</code> that live on processes
     *  \c ranks(o) respectively.  Following the usual convention,
     *  <code>offset(n) = nnz</code>, where \c n is the number of queries that
     *  were performed and \c nnz is the total number of collisions.
     *
     *  \note The views \c indices, \c offset, and \c ranks are passed by
     *  reference because \c Kokkos::realloc() calls the assignment operator.
     *
     *  \param[in] predicates Collection of predicates of the same type.  These
     *  may be spatial predicates or nearest predicates.
     *  \param[out] args
     *     - \c indices Object local indices that satisfy the predicates.
     *     - \c offset Array of predicate offsets for one-dimensional
     *       storage.
     *     - \c ranks Process ranks that own objects.
     *     - \c distances Computed distances (optional and only for nearest
     *       predicates).
     */
    template <typename Predicates, typename... Args>
    void query( Predicates const &predicates, Args &&... args ) const
    {
        // FIXME lame placeholder for concept check
        static_assert( Kokkos::is_view<Predicates>::value, "must pass a view" );
        using Tag = typename Predicates::value_type::Tag;
        Details::DistributedSearchTreeImpl<DeviceType>::queryDispatch(
            Tag{}, *this, predicates, std::forward<Args>( args )... );
    }

  private:
    friend struct Details::DistributedSearchTreeImpl<DeviceType>;
    MPI_Comm _comm;
    BVH<DeviceType> _top_tree;    // replicated
    BVH<DeviceType> _bottom_tree; // local
    size_type _top_tree_size;
    Kokkos::View<size_type *, DeviceType> _bottom_tree_sizes;
};

template <typename DeviceType>
template <typename Primitives>
DistributedSearchTree<DeviceType>::DistributedSearchTree(
    MPI_Comm comm, Primitives const &primitives )
    : _comm( comm )
    , _bottom_tree( primitives )
{
    int comm_rank;
    MPI_Comm_rank( _comm, &comm_rank );
    int comm_size;
    MPI_Comm_size( _comm, &comm_size );

    Kokkos::View<Box *, DeviceType> boxes(
        Kokkos::ViewAllocateWithoutInitializing( "rank_bounding_boxes" ),
        comm_size );
    // FIXME when we move to MPI with CUDA-aware support, we will not need to
    // copy from the device to the host
    auto boxes_host = Kokkos::create_mirror_view( boxes );
    boxes_host( comm_rank ) = _bottom_tree.bounds();
    MPI_Allgather( MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   static_cast<void *>( boxes_host.data() ), sizeof( Box ),
                   MPI_BYTE, _comm );
    Kokkos::deep_copy( boxes, boxes_host );

    _top_tree = BVH<DeviceType>( boxes );

    _bottom_tree_sizes = Kokkos::View<size_type *, DeviceType>(
        Kokkos::ViewAllocateWithoutInitializing( "leave_count_in_local_trees" ),
        comm_size );
    auto bottom_tree_sizes_host =
        Kokkos::create_mirror_view( _bottom_tree_sizes );
    bottom_tree_sizes_host( comm_rank ) = _bottom_tree.size();
    MPI_Allgather( MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   static_cast<void *>( bottom_tree_sizes_host.data() ),
                   sizeof( size_type ), MPI_BYTE, _comm );
    Kokkos::deep_copy( _bottom_tree_sizes, bottom_tree_sizes_host );

    _top_tree_size = accumulate( _bottom_tree_sizes, 0 );
}

} // namespace ArborX

#endif
