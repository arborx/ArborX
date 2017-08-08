/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#ifndef DTK_DISTRIBUTED_SEARCH_TREE_DECL_HPP
#define DTK_DISTRIBUTED_SEARCH_TREE_DECL_HPP

#include <Kokkos_View.hpp>

#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>

#include <DTK_DBC.hpp>
#include <DTK_LinearBVH.hpp>
#include <details/DTK_DetailsDistributedSearchTreeImpl.hpp>

#include "DTK_ConfigDefs.hpp"

namespace DataTransferKit
{

template <typename DeviceType>
class DistributedSearchTree
{
  public:
    DistributedSearchTree(
        Teuchos::RCP<Teuchos::Comm<int> const> comm,
        Kokkos::View<Box const *, DeviceType> bounding_boxes );

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
     *  \param[in] queries Collection of predicates of the same type.  These
     *  may be spatial predicates or nearest predicates.
     *  \param[out] indices Object local indices that satisfy the predicates.
     *  \param[out] offset Array of predicate offsets for one-dimensional
     *  storage.
     *  \param[out] ranks Process ranks that own objects.
     */
    template <typename Query>
    void query( Kokkos::View<Query *, DeviceType> queries,
                Kokkos::View<int *, DeviceType> &indices,
                Kokkos::View<int *, DeviceType> &offset,
                Kokkos::View<int *, DeviceType> &ranks ) const;

    template <typename Query>
    typename std::enable_if<
        std::is_same<typename Query::Tag, Details::NearestPredicateTag>::value,
        void>::type
    query( Kokkos::View<Query *, DeviceType> queries,
           Kokkos::View<int *, DeviceType> &indices,
           Kokkos::View<int *, DeviceType> &offset,
           Kokkos::View<int *, DeviceType> &ranks,
           Kokkos::View<double *, DeviceType> &distances ) const;

  private:
    Teuchos::RCP<Teuchos::Comm<int> const> _comm;
    BVH<DeviceType> _local_tree;
    std::shared_ptr<BVH<DeviceType>> _distributed_tree;
};

template <typename DeviceType>
template <typename Query>
void DistributedSearchTree<DeviceType>::query(
    Kokkos::View<Query *, DeviceType> queries,
    Kokkos::View<int *, DeviceType> &indices,
    Kokkos::View<int *, DeviceType> &offset,
    Kokkos::View<int *, DeviceType> &ranks ) const
{
    using Tag = typename Query::Tag;
    DistributedSearchTreeImpl<DeviceType>::query_dispatch(
        _comm, *_distributed_tree, _local_tree, queries, indices, offset, ranks,
        Tag{} );
}

template <typename DeviceType>
template <typename Query>
typename std::enable_if<
    std::is_same<typename Query::Tag, Details::NearestPredicateTag>::value,
    void>::type
DistributedSearchTree<DeviceType>::query(
    Kokkos::View<Query *, DeviceType> queries,
    Kokkos::View<int *, DeviceType> &indices,
    Kokkos::View<int *, DeviceType> &offset,
    Kokkos::View<int *, DeviceType> &ranks,
    Kokkos::View<double *, DeviceType> &distances ) const
{
    using Tag = typename Query::Tag;
    DistributedSearchTreeImpl<DeviceType>::query_dispatch(
        _comm, *_distributed_tree, _local_tree, queries, indices, offset, ranks,
        Tag{}, &distances );
}

} // end namespace DataTransferKit

#endif
