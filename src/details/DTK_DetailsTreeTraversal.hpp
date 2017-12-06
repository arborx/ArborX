/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/
#ifndef DTK_DETAILS_TREE_TRAVERSAL_HPP
#define DTK_DETAILS_TREE_TRAVERSAL_HPP

#include <DTK_DBC.hpp>

#include <DTK_DetailsAlgorithms.hpp>
#include <DTK_DetailsNode.hpp>
#include <DTK_DetailsPredicate.hpp>
#include <DTK_DetailsPriorityQueue.hpp>
#include <DTK_DetailsStack.hpp>

namespace DataTransferKit
{

template <typename DeviceType>
class BVH;

namespace Details
{
template <typename DeviceType>
struct TreeTraversal
{
  public:
    using ExecutionSpace = typename DeviceType::execution_space;

    template <typename Predicate, typename Insert>
    KOKKOS_INLINE_FUNCTION static int query( BVH<DeviceType> const bvh,
                                             Predicate const &pred,
                                             Insert const &insert )
    {
        using Tag = typename Predicate::Tag;
        return queryDispatch( bvh, pred, insert, Tag{} );
    }

    /**
     * Return true if the node is a leaf.
     */
    KOKKOS_INLINE_FUNCTION
    static bool isLeaf( BVH<DeviceType> bvh, Node const *node )
    {
        // COMMENT: could also check that pointer is in the range [leaf_nodes,
        // leaf_nodes+n]
        (void)bvh;
        return ( node->children.first == nullptr ) &&
               ( node->children.second == nullptr );
    }

    /**
     * Return the index of the leaf node.
     */
    KOKKOS_INLINE_FUNCTION
    static int getIndex( BVH<DeviceType> bvh, Node const *leaf )
    {
        return bvh._indices[leaf - bvh._leaf_nodes.data()];
    }

    /**
     * Return the root node of the BVH.
     */
    KOKKOS_INLINE_FUNCTION
    static Node const *getRoot( BVH<DeviceType> bvh )
    {
        if ( bvh.empty() )
            return nullptr;
        return ( bvh.size() > 1 ? bvh._internal_nodes : bvh._leaf_nodes )
            .data();
    }
};

// There are two (related) families of search: one using a spatial predicate and
// one using nearest neighbours query (see boost::geometry::queries
// documentation).
template <typename DeviceType, typename Predicate, typename Insert>
KOKKOS_FUNCTION int spatialQuery( BVH<DeviceType> const bvh,
                                  Predicate const &predicate,
                                  Insert const &insert )
{
    if ( bvh.empty() )
        return 0;

    if ( bvh.size() == 1 )
    {
        Node const *leaf = TreeTraversal<DeviceType>::getRoot( bvh );
        if ( predicate( leaf ) )
        {
            int const leaf_index =
                TreeTraversal<DeviceType>::getIndex( bvh, leaf );
            insert( leaf_index );
            return 1;
        }
        else
            return 0;
    }

    Stack<Node const *> stack;

    Node const *root = TreeTraversal<DeviceType>::getRoot( bvh );
    stack.push( root );
    int count = 0;

    while ( !stack.empty() )
    {
        Node const *node = stack.top();
        stack.pop();

        if ( TreeTraversal<DeviceType>::isLeaf( bvh, node ) )
        {
            insert( TreeTraversal<DeviceType>::getIndex( bvh, node ) );
            count++;
        }
        else
        {
            for ( Node const *child :
                  {node->children.first, node->children.second} )
            {
                if ( predicate( child ) )
                {
                    stack.push( child );
                }
            }
        }
    }
    return count;
}

// query k nearest neighbours
template <typename DeviceType, typename Distance, typename Insert>
KOKKOS_FUNCTION int nearestQuery( BVH<DeviceType> const bvh,
                                  Distance const &distance, int k,
                                  Insert const &insert )
{
    if ( bvh.empty() || k < 1 )
        return 0;

    if ( bvh.size() == 1 )
    {
        Node const *leaf = TreeTraversal<DeviceType>::getRoot( bvh );
        int const leaf_index = TreeTraversal<DeviceType>::getIndex( bvh, leaf );
        double const leaf_distance = distance( leaf );
        insert( leaf_index, leaf_distance );
        return 1;
    }

    using PairNodePtrDistance = Kokkos::pair<Node const *, double>;

    struct CompareDistance
    {
        KOKKOS_INLINE_FUNCTION bool operator()( PairNodePtrDistance const &lhs,
                                                PairNodePtrDistance const &rhs )
        {
            // reverse order (larger distance means lower priority)
            return lhs.second > rhs.second;
        }
    };

    PriorityQueue<PairNodePtrDistance, CompareDistance> queue;
    // priority does not matter for the root since the node will be
    // processed directly and removed from the priority queue we don't even
    // bother computing the distance to it.
    Node const *root = TreeTraversal<DeviceType>::getRoot( bvh );
    queue.push( root, 0. );
    int count = 0;

    while ( !queue.empty() && count < k )
    {
        // get the node that is on top of the priority list (i.e. is the
        // closest to the query point)
        Node const *node = queue.top().first;
        double const node_distance = queue.top().second;
        // NOTE: it would be nice to be able to do something like
        // tie( node, node_distance = queue.top();
        queue.pop();
        if ( TreeTraversal<DeviceType>::isLeaf( bvh, node ) )
        {
            insert( TreeTraversal<DeviceType>::getIndex( bvh, node ),
                    node_distance );
            count++;
        }
        else
        {
            // insert children of the node in the priority list
            for ( Node const *child :
                  {node->children.first, node->children.second} )
            {
                double const child_distance = distance( child );
                queue.push( child, child_distance );
            }
        }
    }
    return count;
}

template <typename DeviceType, typename Predicate, typename Insert>
KOKKOS_INLINE_FUNCTION int
queryDispatch( BVH<DeviceType> const bvh, Predicate const &pred,
               Insert const &insert, SpatialPredicateTag )
{
    return spatialQuery( bvh, pred, insert );
}

template <typename DeviceType, typename Predicate, typename Insert>
KOKKOS_INLINE_FUNCTION int
queryDispatch( BVH<DeviceType> const bvh, Predicate const &pred,
               Insert const &insert, NearestPredicateTag )
{
    auto const geometry = pred._geometry;
    auto const k = pred._k;
    return nearestQuery( bvh,
                         [geometry]( Node const *node ) {
                             return distance( geometry, node->bounding_box );
                         },
                         k, insert );
}

} // end namespace Details
} // end namespace DataTransferKit

#endif
