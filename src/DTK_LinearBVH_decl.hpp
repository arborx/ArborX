/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#ifndef DTK_LINEAR_BVH_DECL_HPP
#define DTK_LINEAR_BVH_DECL_HPP

#include <Kokkos_ArithTraits.hpp>
#include <Kokkos_Array.hpp>
#include <Kokkos_View.hpp>

#include <DTK_DetailsAlgorithms.hpp>
#include <DTK_DetailsBox.hpp>
#include <DTK_DetailsNode.hpp>
#include <DTK_DetailsPredicate.hpp>

#include "DTK_ConfigDefs.hpp"

namespace DataTransferKit
{
namespace Details
{
template <typename DeviceType>
struct TreeTraversal;
}

/**
 * Bounding Volume Hierarchy.
 */
template <typename DeviceType>
class BVH
{
  public:
    BVH( Kokkos::View<Box const *, DeviceType> bounding_boxes );

    // Views are passed by reference here because Kokkos::resize() effectively
    // calls the assignment operator.
    template <typename Query>
    void query( Kokkos::View<Query *, DeviceType> queries,
                Kokkos::View<int *, DeviceType> &indices,
                Kokkos::View<int *, DeviceType> &offset ) const;

    Box bounds() const;

  private:
    friend struct Details::TreeTraversal<DeviceType>;

    Kokkos::View<Node *, DeviceType> _leaf_nodes;
    Kokkos::View<Node *, DeviceType> _internal_nodes;
    /**
     * Array of indices that sort the boxes used to construct the hierarchy.
     * The leaf nodes are ordered so we need these to identify objects that
     * meet a predicate.
     */
    Kokkos::View<int *, DeviceType> _indices;
};

template <typename DeviceType>
template <typename Query>
void BVH<DeviceType>::query( Kokkos::View<Query *, DeviceType> queries,
                             Kokkos::View<int *, DeviceType> &indices,
                             Kokkos::View<int *, DeviceType> &offset ) const
{
    using ExecutionSpace = typename DeviceType::execution_space;

    namespace details = DataTransferKit::Details;

    int const n_queries = queries.extent( 0 );

    // Initialize view
    // [ 0 0 0 .... 0 0 ]
    //                ^
    //                N
    Kokkos::resize( offset, n_queries + 1 );
    Kokkos::parallel_for(
        REGION_NAME( "initialize_offset_set_all_entries_to_zero)" ),
        Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries + 1 ),
        KOKKOS_LAMBDA( int i ) { offset[i] = 0; } );
    Kokkos::fence();

    // Make a copy of *this. We need this to put is on the device. Otherwise,
    // it will throw illegal address error in the parallel_for loops below.
    BVH<DeviceType> bvh = *this;

    // Say we found exactly two object for each query:
    // [ 2 2 2 .... 2 0 ]
    //   ^            ^
    //   0th          Nth element in the view
    Kokkos::parallel_for(
        REGION_NAME( "first_pass_at_the_search_count_the_number_of_indices" ),
        Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
        KOKKOS_LAMBDA( int i ) {
            offset( i ) = details::TreeTraversal<DeviceType>::query(
                bvh, queries( i ), []( int index ) {} );
        } );
    Kokkos::fence();

    // Then we would get:
    // [ 0 2 4 .... 2N-2 2N ]
    //                    ^
    //                    N
    Kokkos::parallel_scan(
        REGION_NAME( "compute_offset" ),
        Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries + 1 ),
        KOKKOS_LAMBDA( int i, int &update, bool final_pass ) {
            int const offset_i = offset( i );
            if ( final_pass )
                offset( i ) = update;
            update += offset_i;
        } );
    Kokkos::fence();

    // Let us extract the last element in the view which is the total count of
    // objects which where found to meet the query predicates:
    //
    // [ 2N ]
    auto total_count = Kokkos::subview( offset, n_queries );
    auto total_count_host = Kokkos::create_mirror_view( total_count );
    // We allocate the memory and fill
    //
    // [ A0 A1 B0 B1 C0 C1 ... X0 X1 ]
    //   ^     ^     ^         ^     ^
    //   0     2     4         2N-2  2N
    Kokkos::deep_copy( total_count_host, total_count );
    Kokkos::resize( indices, total_count( 0 ) );
    Kokkos::parallel_for( REGION_NAME( "second_pass" ),
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
                          KOKKOS_LAMBDA( int i ) {
                              int count = 0;
                              details::TreeTraversal<DeviceType>::query(
                                  bvh, queries( i ),
                                  [indices, offset, i, &count]( int index ) {
                                      indices( offset( i ) + count++ ) = index;
                                  } );
                          } );
    Kokkos::fence();
}

} // end namespace DataTransferKit

#endif
