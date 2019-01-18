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

#ifndef DTK_DETAILS_BATCHED_QUERIES_HPP
#define DTK_DETAILS_BATCHED_QUERIES_HPP

#include <DTK_Box.hpp>
#include <DTK_DetailsAlgorithms.hpp> // returnCentroid, translateAndScale
#include <DTK_DetailsMortonCode.hpp> // morton3D
#include <DTK_DetailsSortUtils.hpp>  // sortObjects
#include <DTK_DetailsUtils.hpp>      // iota, exclusivePrefixSum, lastElement

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#include <tuple>

namespace DataTransferKit
{

namespace Details
{
template <typename DeviceType>
struct BatchedQueries
{
  public:
    using ExecutionSpace = typename DeviceType::execution_space;

    // BatchedQueries defines functions for sorting queries along the Z-order
    // space-filling curve in order to minimize data divergence.  The goal is
    // to increase correlation between traversal decisions made by nearby
    // threads and thereby increase performance.
    //
    // NOTE: sortQueriesAlongZOrderCurve() does not actually apply the sorting
    // order, it returns the permutation indices.  applyPermutation() was added
    // in that purpose.  reversePermutation() is able to restore the initial
    // order on the results that are in "compressed row storage" format.  You
    // may notice it is not used any more in the code that performs the batched
    // queries.  We found that it was slighly more performant to add a level of
    // indirection when recording results rather than using that function at
    // the end.  We decided to keep reversePermutation around for now.

    template <typename Query>
    static Kokkos::View<size_t *, DeviceType>
    sortQueriesAlongZOrderCurve( Box const &scene_bounding_box,
                                 Kokkos::View<Query *, DeviceType> queries )
    {
        auto const n_queries = queries.extent( 0 );

        Kokkos::View<unsigned int *, DeviceType> morton_codes(
            Kokkos::ViewAllocateWithoutInitializing( "morton" ), n_queries );
        Kokkos::parallel_for(
            DTK_MARK_REGION( "assign_morton_codes_to_queries" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
            KOKKOS_LAMBDA( int i ) {
                Point xyz = Details::returnCentroid( queries( i )._geometry );
                translateAndScale( xyz, xyz, scene_bounding_box );
                morton_codes( i ) = morton3D( xyz[0], xyz[1], xyz[2] );
            } );
        Kokkos::fence();

        return sortObjects( morton_codes );
    }

    template <typename T>
    static Kokkos::View<T *, DeviceType>
    applyPermutation( Kokkos::View<size_t const *, DeviceType> permute,
                      Kokkos::View<T *, DeviceType> v )
    {
        auto const n = permute.extent( 0 );
        DTK_REQUIRE( v.extent( 0 ) == n );

        auto w = cloneWithoutInitializingNorCopying( v );
        Kokkos::parallel_for(
            DTK_MARK_REGION( "permute_entries" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
            KOKKOS_LAMBDA( int i ) { w( i ) = v( permute( i ) ); } );
        Kokkos::fence();

        return w;
    }

    static Kokkos::View<int *, DeviceType>
    permuteOffset( Kokkos::View<size_t const *, DeviceType> permute,
                   Kokkos::View<int const *, DeviceType> offset )
    {
        auto const n = permute.extent( 0 );
        DTK_REQUIRE( offset.extent( 0 ) == n + 1 );

        auto tmp_offset = cloneWithoutInitializingNorCopying( offset );
        Kokkos::parallel_for(
            DTK_MARK_REGION( "adjacent_difference_and_permutation" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
            KOKKOS_LAMBDA( int i ) {
                tmp_offset( permute( i ) ) = offset( i + 1 ) - offset( i );
            } );
        Kokkos::fence();

        exclusivePrefixSum( tmp_offset );

        return tmp_offset;
    }

    template <typename T>
    static Kokkos::View<T *, DeviceType>
    permuteIndices( Kokkos::View<size_t const *, DeviceType> permute,
                    Kokkos::View<T const *, DeviceType> indices,
                    Kokkos::View<int const *, DeviceType> offset,
                    Kokkos::View<int const *, DeviceType> tmp_offset )
    {
        auto const n = permute.extent( 0 );

        DTK_REQUIRE( offset.extent( 0 ) == n + 1 );
        DTK_REQUIRE( tmp_offset.extent( 0 ) == n + 1 );
        DTK_REQUIRE( lastElement( offset ) == indices.extent_int( 0 ) );
        DTK_REQUIRE( lastElement( tmp_offset ) == indices.extent_int( 0 ) );

        auto tmp_indices = cloneWithoutInitializingNorCopying( indices );
        Kokkos::parallel_for(
            DTK_MARK_REGION( "permute_indices" ),
            Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
            KOKKOS_LAMBDA( int q ) {
                for ( int i = 0; i < offset( q + 1 ) - offset( q ); ++i )
                {
                    tmp_indices( tmp_offset( permute( q ) ) + i ) =
                        indices( offset( q ) + i );
                }
            } );
        Kokkos::fence();
        return tmp_indices;
    }

    static std::tuple<Kokkos::View<int *, DeviceType>,
                      Kokkos::View<int *, DeviceType>>
    reversePermutation( Kokkos::View<size_t const *, DeviceType> permute,
                        Kokkos::View<int const *, DeviceType> offset,
                        Kokkos::View<int const *, DeviceType> indices )
    {
        auto const tmp_offset = permuteOffset( permute, offset );

        auto const tmp_indices =
            permuteIndices( permute, indices, offset, tmp_offset );
        return std::make_tuple( tmp_offset, tmp_indices );
    }

    static std::tuple<Kokkos::View<int *, DeviceType>,
                      Kokkos::View<int *, DeviceType>,
                      Kokkos::View<double *, DeviceType>>
    reversePermutation( Kokkos::View<size_t const *, DeviceType> permute,
                        Kokkos::View<int const *, DeviceType> offset,
                        Kokkos::View<int const *, DeviceType> indices,
                        Kokkos::View<double const *, DeviceType> distances )
    {
        auto const tmp_offset = permuteOffset( permute, offset );

        auto const tmp_indices =
            permuteIndices( permute, indices, offset, tmp_offset );

        auto const tmp_distances =
            permuteIndices( permute, distances, offset, tmp_offset );

        return std::make_tuple( tmp_offset, tmp_indices, tmp_distances );
    }
};

} // namespace Details
} // namespace DataTransferKit

#endif
