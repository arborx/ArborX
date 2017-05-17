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
#include <DTK_Predicate.hpp>
#include <DTK_PriorityQueue.hpp>
#include <DTK_Stack.hpp>

#include <DTK_BVHQuery.hpp>
#include <DTK_LinearBVH.hpp>
#include <DTK_Node.hpp>

namespace DataTransferKit
{
namespace Details
{

// There are two (related) families of search: one using a spatial predicate and
// one using nearest neighbours query (see boost::geometry::queries
// documentation).
template <typename NO, typename Predicate>
KOKKOS_FUNCTION void
spatial_query( BVH<NO> const bvh, Predicate const &predicate, int *indices,
               unsigned int &n_indices, unsigned int max_n_indices )
{
    Stack<Node const *> stack;

    Node const *node = BVHQuery<NO>::getRoot( bvh );
    stack.push( node );
    n_indices = 0;

    while ( !stack.empty() )
    {
        node = stack.top();
        stack.pop();

        if ( BVHQuery<NO>::isLeaf( node ) )
        {
#if HAVE_DTK_DBC
            if ( n_indices > max_n_indices )
                printf( "Increase the size of indices array\n" );
#endif
            assert( n_indices < max_n_indices );
            // and just to make compilers happy if NDEBUG
            (void)max_n_indices;

            indices[n_indices++] = BVHQuery<NO>::getIndex( bvh, node );
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
}

template <typename NO, typename Predicate>
unsigned int query_dispatch( BVH<NO> const bvh, Predicate const &pred,
                             Kokkos::View<int *, typename NO::device_type> out,
                             SpatialPredicateTag )
{
    unsigned int constexpr max_n_indices = 1000;
    int indices[max_n_indices];
    unsigned int n_indices = 0;
    spatial_query( bvh, pred, indices, n_indices, max_n_indices );
    out = Kokkos::View<int *, typename NO::device_type>( "dummy", n_indices );
    for ( unsigned int i = 0; i < n_indices; ++i )
    {
        out[i] = indices[i];
    }
    return n_indices;
}

// query k nearest neighbours
template <typename NO>
KOKKOS_FUNCTION void
nearest_query( BVH<NO> const bvh, Point const &query_point, int k, int *indices,
               unsigned int &n_indices, unsigned int max_n_indices )
{
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
    // bother computing the distance to it
    Node const *node = BVHQuery<NO>::getRoot( bvh );
    double node_distance = 0.0;
    queue.push( node, node_distance );
    n_indices = 0;

    while ( !queue.empty() && static_cast<int>( n_indices ) < k )
    {
        // get the node that is on top of the priority list (i.e. is the
        // closest to the query point)
        node = queue.top().first; // std::tie(node, std::ignore) = ...
        queue.pop();
        if ( BVHQuery<NO>::isLeaf( node ) )
        {
#if HAVE_DTK_DBC
            if ( n_indices > max_n_indices )
                printf( "Increase the size of indices array\n" );
#endif
            assert( n_indices < max_n_indices );
            // and just to make compilers happy if NDEBUG
            (void)max_n_indices;

            indices[n_indices++] = BVHQuery<NO>::getIndex( bvh, node );
        }
        else
        {
            // insert children of the node in the priority list
            for ( Node const *child :
                  {node->children.first, node->children.second} )
            {
                double child_distance =
                    distance( query_point, child->bounding_box );
                queue.push( child, child_distance );
            }
        }
    }
}

template <typename NO, typename Predicate>
int query_dispatch( BVH<NO> const bvh, Predicate const &pred,
                    Kokkos::View<int *, typename NO::device_type> out,
                    NearestPredicateTag )
{
    unsigned int constexpr max_n_indices = 1000;
    int indices[max_n_indices];
    unsigned int n_indices = 0;
    nearest_query( bvh, pred._query_point, pred._k, indices, n_indices,
                   max_n_indices );
    int const n = n_indices;
    out = Kokkos::View<int *, typename NO::device_type>( "dummy", n );
    for ( unsigned int i = 0; i < n_indices; ++i )
        out[i] = indices[i];

    return n;
}

} // end namespace Details
} // end namespace DataTransferKit

#endif
