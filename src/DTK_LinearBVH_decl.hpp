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

#ifndef DTK_LINEAR_BVH_DECL_HPP
#define DTK_LINEAR_BVH_DECL_HPP

#include "DTK_ConfigDefs.hpp"

#include <DTK_Box.hpp>
#include <DTK_DetailsAlgorithms.hpp>
#include <DTK_DetailsBatchedQueries.hpp>
#include <DTK_DetailsNode.hpp>
#include <DTK_DetailsTreeTraversal.hpp>
#include <DTK_DetailsUtils.hpp>
#include <DTK_Predicates.hpp>

#include <Kokkos_ArithTraits.hpp>
#include <Kokkos_Array.hpp>
#include <Kokkos_View.hpp>

namespace DataTransferKit
{

template <typename DeviceType>
class BoundingVolumeHierarchy
{
  public:
    using TreeType = BoundingVolumeHierarchy;

    BoundingVolumeHierarchy() = default; // build an empty tree
    BoundingVolumeHierarchy(
        Kokkos::View<Box const *, DeviceType> bounding_boxes );

    // Views are passed by reference here because internally Kokkos::realloc()
    // is called.
    template <typename Query, typename... Args>
    void query( Kokkos::View<Query *, DeviceType> queries,
                Args &&... args ) const;

    KOKKOS_INLINE_FUNCTION
    Box bounds() const
    {
        if ( empty() )
            return Box();
        return ( size() > 1 ? _internal_nodes : _leaf_nodes )[0].bounding_box;
    }

    using SizeType = typename Kokkos::View<int *, DeviceType>::size_type;
    KOKKOS_INLINE_FUNCTION
    SizeType size() const { return _leaf_nodes.extent( 0 ); }

    KOKKOS_INLINE_FUNCTION
    bool empty() const { return size() == 0; }

  private:
    friend struct Details::TreeTraversal<DeviceType>;

    Kokkos::View<Node *, DeviceType> _leaf_nodes;
    Kokkos::View<Node *, DeviceType> _internal_nodes;
};

template <typename DeviceType>
using BVH = typename BoundingVolumeHierarchy<DeviceType>::TreeType;

template <typename DeviceType, typename Query>
void queryDispatch(
    Details::NearestPredicateTag, BoundingVolumeHierarchy<DeviceType> const bvh,
    Kokkos::View<Query *, DeviceType> queries,
    Kokkos::View<int *, DeviceType> &indices,
    Kokkos::View<int *, DeviceType> &offset,
    Kokkos::View<double *, DeviceType> *distances_ptr = nullptr )
{
    using ExecutionSpace = typename DeviceType::execution_space;

    auto const n_queries = queries.extent( 0 );

    auto const permute =
        Details::BatchedQueries<DeviceType>::sortQueriesAlongZOrderCurve(
            bvh.bounds(), queries );

    queries = Details::BatchedQueries<DeviceType>::applyPermutation( permute,
                                                                     queries );

    reallocWithoutInitializing( offset, n_queries + 1 );
    Kokkos::deep_copy( offset, 0 );

    Kokkos::parallel_for(
        DTK_MARK_REGION( "scan_queries_for_numbers_of_nearest_neighbors" ),
        Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
        KOKKOS_LAMBDA( int i ) { offset( permute( i ) ) = queries( i )._k; } );
    Kokkos::fence();

    exclusivePrefixSum( offset );
    int const n_results = lastElement( offset );

    reallocWithoutInitializing( indices, n_results );
    int const invalid_index = -1;
    Kokkos::deep_copy( indices, invalid_index );
    if ( distances_ptr )
    {
        Kokkos::View<double *, DeviceType> &distances = *distances_ptr;
        reallocWithoutInitializing( distances, n_results );
        double const invalid_distance = -Kokkos::ArithTraits<double>::max();
        Kokkos::deep_copy( distances, invalid_distance );

        Kokkos::parallel_for(
            DTK_MARK_REGION( "perform_nearest_queries_and_return_distances" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
            KOKKOS_LAMBDA( int i ) {
                int count = 0;
                Details::TreeTraversal<DeviceType>::query(
                    bvh, queries( i ),
                    [indices, offset, distances, permute, i,
                     &count]( int index, double distance ) {
                        indices( offset( permute( i ) ) + count ) = index;
                        distances( offset( permute( i ) ) + count ) = distance;
                        count++;
                    } );
            } );
        Kokkos::fence();
    }
    else
    {
        Kokkos::parallel_for(
            DTK_MARK_REGION( "perform_nearest_queries" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
            KOKKOS_LAMBDA( int i ) {
                int count = 0;
                Details::TreeTraversal<DeviceType>::query(
                    bvh, queries( i ),
                    [indices, offset, permute, i, &count]( int index, double ) {
                        indices( offset( permute( i ) ) + count++ ) = index;
                    } );
            } );
        Kokkos::fence();
    }
    // Find out if they are any invalid entries in the indices (i.e. at least
    // one query asked for more neighbors that they are leaves in the tree) and
    // eliminate them if necessary.
    auto tmp_offset = cloneWithoutInitializingNorCopying( offset );
    Kokkos::deep_copy( tmp_offset, 0 );
    Kokkos::parallel_for( DTK_MARK_REGION( "count_invalid_indices" ),
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
                          KOKKOS_LAMBDA( int q ) {
                              for ( int i = offset( q ); i < offset( q + 1 );
                                    ++i )
                                  if ( indices( i ) == invalid_index )
                                  {
                                      tmp_offset( q ) = offset( q + 1 ) - i;
                                      break;
                                  }
                          } );
    Kokkos::fence();
    exclusivePrefixSum( tmp_offset );
    int const n_invalid_indices = lastElement( tmp_offset );
    if ( n_invalid_indices > 0 )
    {
        Kokkos::parallel_for(
            DTK_MARK_REGION( "subtract_invalid_entries_from_offset" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries + 1 ),
            KOKKOS_LAMBDA( int q ) {
                tmp_offset( q ) = offset( q ) - tmp_offset( q );
            } );
        Kokkos::fence();

        int const n_valid_indices = n_results - n_invalid_indices;
        Kokkos::View<int *, DeviceType> tmp_indices(
            Kokkos::ViewAllocateWithoutInitializing( indices.label() ),
            n_valid_indices );

        Kokkos::parallel_for(
            DTK_MARK_REGION( "copy_valid_indices" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
            KOKKOS_LAMBDA( int q ) {
                for ( int i = 0; i < tmp_offset( q + 1 ) - tmp_offset( q );
                      ++i )
                {
                    tmp_indices( tmp_offset( q ) + i ) =
                        indices( offset( q ) + i );
                }
            } );
        Kokkos::fence();
        indices = tmp_indices;
        if ( distances_ptr )
        {
            Kokkos::View<double *, DeviceType> &distances = *distances_ptr;
            Kokkos::View<double *, DeviceType> tmp_distances(
                Kokkos::ViewAllocateWithoutInitializing( distances.label() ),
                n_valid_indices );
            Kokkos::parallel_for(
                DTK_MARK_REGION( "copy_valid_distances" ),
                Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
                KOKKOS_LAMBDA( int q ) {
                    for ( int i = 0; i < tmp_offset( q + 1 ) - tmp_offset( q );
                          ++i )
                    {
                        tmp_distances( tmp_offset( q ) + i ) =
                            distances( offset( q ) + i );
                    }
                } );
            Kokkos::fence();
            distances = tmp_distances;
        }
        offset = tmp_offset;
    }
}

template <typename DeviceType, typename Query>
void queryDispatch( Details::SpatialPredicateTag,
                    BoundingVolumeHierarchy<DeviceType> const bvh,
                    Kokkos::View<Query *, DeviceType> queries,
                    Kokkos::View<int *, DeviceType> &indices,
                    Kokkos::View<int *, DeviceType> &offset,
                    int buffer_size = 30 )
{
    using ExecutionSpace = typename DeviceType::execution_space;

    auto const n_queries = queries.extent( 0 );

    auto const permute =
        Details::BatchedQueries<DeviceType>::sortQueriesAlongZOrderCurve(
            bvh.bounds(), queries );

    queries = Details::BatchedQueries<DeviceType>::applyPermutation( permute,
                                                                     queries );

    // Initialize view
    // [ 0 0 0 .... 0 0 ]
    //                ^
    //                N
    reallocWithoutInitializing( offset, n_queries + 1 );
    Kokkos::deep_copy( offset, 0 );

    // Not proud of that one but that will do for now :/
    auto const throw_if_buffer_optimization_fails = [&buffer_size]() {
        if ( buffer_size < 0 )
        {
            buffer_size = -buffer_size;
            return true;
        }
        else
            return false;
    }();

    // Say we found exactly two object for each query:
    // [ 2 2 2 .... 2 0 ]
    //   ^            ^
    //   0th          Nth element in the view
    if ( buffer_size > 0 )
    {
        reallocWithoutInitializing( indices, n_queries * buffer_size );
        // NOTE I considered filling with invalid indices but it is unecessary
        // work

        Kokkos::parallel_for(
            DTK_MARK_REGION(
                "first_pass_at_the_search_with_buffer_optimization" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
            KOKKOS_LAMBDA( int i ) {
                int count = 0;
                offset( permute( i ) ) =
                    Details::TreeTraversal<DeviceType>::query(
                        bvh, queries( i ),
                        [indices, offset, permute, buffer_size, i,
                         &count]( int index ) {
                            if ( count < buffer_size )
                                indices( permute( i ) * buffer_size +
                                         count++ ) = index;
                        } );
            } );
    }
    else
        Kokkos::parallel_for(
            DTK_MARK_REGION(
                "first_pass_at_the_search_count_the_number_of_indices" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
            KOKKOS_LAMBDA( int i ) {
                offset( permute( i ) ) =
                    Details::TreeTraversal<DeviceType>::query(
                        bvh, queries( i ), []( int index ) {} );
            } );
    Kokkos::fence();

    // NOTE max() internally calls Kokkos::parallel_reduce.  Only pay for it if
    // actually trying buffer optimization.  In principle, any strictly
    // positive value can be assigned otherwise.
    auto const max_results_per_query =
        ( buffer_size > 0 )
            ? max( offset )
            : std::numeric_limits<typename std::remove_reference<decltype(
                  offset )>::type::value_type>::max();

    // Then we would get:
    // [ 0 2 4 .... 2N-2 2N ]
    //                    ^
    //                    N
    exclusivePrefixSum( offset );

    // Let us extract the last element in the view which is the total count of
    // objects which where found to meet the query predicates:
    //
    // [ 2N ]
    int const n_results = lastElement( offset );

    if ( max_results_per_query > buffer_size )
    {
        // FIXME can definitely do better about error message
        DTK_INSIST( !throw_if_buffer_optimization_fails );

        // We allocate the memory and fill
        //
        // [ A0 A1 B0 B1 C0 C1 ... X0 X1 ]
        //   ^     ^     ^         ^     ^
        //   0     2     4         2N-2  2N
        reallocWithoutInitializing( indices, n_results );
        Kokkos::parallel_for(
            DTK_MARK_REGION( "second_pass" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
            KOKKOS_LAMBDA( int i ) {
                int count = 0;
                Details::TreeTraversal<DeviceType>::query(
                    bvh, queries( i ),
                    [indices, offset, permute, i, &count]( int index ) {
                        indices( offset( permute( i ) ) + count++ ) = index;
                    } );
            } );
        Kokkos::fence();
    }
    else
    {
        Kokkos::View<int *, DeviceType> tmp_indices(
            Kokkos::ViewAllocateWithoutInitializing( indices.label() ),
            n_results );
        Kokkos::parallel_for(
            DTK_MARK_REGION( "copy_valid_indices" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
            KOKKOS_LAMBDA( int q ) {
                for ( int i = 0; i < offset( q + 1 ) - offset( q ); ++i )
                {
                    tmp_indices( offset( q ) + i ) =
                        indices( q * buffer_size + i );
                }
            } );
        Kokkos::fence();
        indices = tmp_indices;
    }
}

template <typename DeviceType, typename Query>
void queryDispatch( Details::NearestPredicateTag tag,
                    BoundingVolumeHierarchy<DeviceType> const bvh,
                    Kokkos::View<Query *, DeviceType> queries,
                    Kokkos::View<int *, DeviceType> &indices,
                    Kokkos::View<int *, DeviceType> &offset,
                    Kokkos::View<double *, DeviceType> &distances )
{
    queryDispatch( tag, bvh, queries, indices, offset, &distances );
}

template <typename DeviceType>
template <typename Query, typename... Args>
void BoundingVolumeHierarchy<DeviceType>::query(
    Kokkos::View<Query *, DeviceType> queries, Args &&... args ) const
{
    using Tag = typename Query::Tag;
    queryDispatch( Tag{}, *this, queries, std::forward<Args>( args )... );
}

} // namespace DataTransferKit

#endif
