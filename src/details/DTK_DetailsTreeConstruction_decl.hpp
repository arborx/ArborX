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

#ifndef DTK_DETAILS_TREE_CONSTRUCTION_DECL_HPP
#define DTK_DETAILS_TREE_CONSTRUCTION_DECL_HPP

#include "DTK_ConfigDefs.hpp"

#include <DTK_Box.hpp>
#include <DTK_DetailsAlgorithms.hpp> // expand
#include <DTK_DetailsMortonCode.hpp> // morton3D
#include <DTK_DetailsNode.hpp>
#include <DTK_DetailsTags.hpp>
#include <DTK_KokkosHelpers.hpp> // clz

#include <Kokkos_Macros.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

namespace DataTransferKit
{
namespace Details
{
/**
 * This structure contains all the functions used to build the BVH. All the
 * functions are static.
 */
template <typename DeviceType>
struct TreeConstruction
{
  public:
    using ExecutionSpace = typename DeviceType::execution_space;

    template <typename Primitives>
    static void calculateBoundingBoxOfTheScene( Primitives const &primitives,
                                                Box &scene_bounding_box );

    // to assign the Morton code for a given object, we use the centroid point
    // of its bounding box, and express it relative to the bounding box of the
    // scene.
    template <typename Primitives>
    static void
    assignMortonCodes( Primitives const &primitives,
                       Kokkos::View<unsigned int *, DeviceType> morton_codes,
                       Box const &scene_bounding_box );

    template <typename Primitives>
    static void initializeLeafNodes(
        Primitives const &primitives,
        Kokkos::View<size_t const *, DeviceType> permutation_indices,
        Kokkos::View<Node *, DeviceType> leaf_nodes );

    static Node *generateHierarchy(
        Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes,
        Kokkos::View<Node *, DeviceType> leaf_nodes,
        Kokkos::View<Node *, DeviceType> internal_nodes,
        Kokkos::View<int *, DeviceType> parents );

    static void calculateInternalNodesBoundingVolumes(
        Kokkos::View<Node const *, DeviceType> leaf_nodes,
        Kokkos::View<Node *, DeviceType> internal_nodes,
        Kokkos::View<int const *, DeviceType> parents );

    KOKKOS_INLINE_FUNCTION
    static int
    commonPrefix( Kokkos::View<unsigned int *, DeviceType> morton_codes, int i,
                  int j )
    {
        using KokkosHelpers::clz;

        int const n = morton_codes.extent( 0 );
        if ( j < 0 || j > n - 1 )
            return -1;

        // our construction algorithm relies on keys being unique so we handle
        // explicitly case of duplicate Morton codes by augmenting each key by
        // a bit representation of its index.
        if ( morton_codes[i] == morton_codes[j] )
        {
            // clz( k[i] ^ k[j] ) == 32
            return 32 + clz( i ^ j );
        }
        return clz( morton_codes[i] ^ morton_codes[j] );
    }

    KOKKOS_FUNCTION
    static int
    findSplit( Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes,
               int first, int last );

    KOKKOS_FUNCTION
    static Kokkos::pair<int, int> determineRange(
        Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes, int i );
};

template <typename ViewType>
class CalculateBoundingBoxOfTheSceneFunctor
{
  public:
    CalculateBoundingBoxOfTheSceneFunctor(
        typename ViewType::const_type primitives )
        : _primitives( primitives )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void init( Box &box ) const { box = Box(); }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i, Box &box ) const
    {
        expand( box, _primitives( i ) );
    }

    KOKKOS_INLINE_FUNCTION
    void join( volatile Box &dst, volatile Box const &src ) const
    {
        expand( dst, src );
    }

  private:
    typename ViewType::const_type _primitives;
};

template <typename DeviceType>
template <typename Primitives>
inline void TreeConstruction<DeviceType>::calculateBoundingBoxOfTheScene(
    Primitives const &primitives, Box &scene_bounding_box )
{
    static_assert( Kokkos::is_view<Primitives>::value, "Must pass a view" );
    static_assert( std::is_same<typename Primitives::traits::device_type,
                                DeviceType>::value,
                   "Wrong device type" );
    // TODO static_assert( is_expandable_v<Box, typename
    // Primitives::value_type), "");
    auto const n = primitives.extent( 0 );
    Kokkos::parallel_reduce(
        DTK_MARK_REGION( "calculate_bounding_box_of_the_scene" ),
        Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
        CalculateBoundingBoxOfTheSceneFunctor<decltype( primitives )>(
            primitives ),
        scene_bounding_box );
    Kokkos::fence();
}

template <typename Primitives, typename MortonCodes>
inline void assignMortonCodesDispatch( BoxTag, Primitives const &primitives,
                                       MortonCodes morton_codes,
                                       Box const &scene_bounding_box )
{
    using ExecutionSpace =
        typename decltype( primitives )::traits::execution_space;
    auto const n = morton_codes.extent( 0 );
    Kokkos::parallel_for(
        DTK_MARK_REGION( "assign_morton_codes" ),
        Kokkos::RangePolicy<ExecutionSpace>( 0, n ), KOKKOS_LAMBDA( int i ) {
            Point xyz;
            centroid( primitives( i ), xyz );
            translateAndScale( xyz, xyz, scene_bounding_box );
            morton_codes( i ) = morton3D( xyz[0], xyz[1], xyz[2] );
        } );
    Kokkos::fence();
}

template <typename Primitives, typename MortonCodes>
inline void assignMortonCodesDispatch( PointTag, Primitives const &primitives,
                                       MortonCodes morton_codes,
                                       Box const &scene_bounding_box )
{
    using ExecutionSpace =
        typename decltype( primitives )::traits::execution_space;
    auto const n = morton_codes.extent( 0 );
    Kokkos::parallel_for(
        DTK_MARK_REGION( "assign_morton_codes" ),
        Kokkos::RangePolicy<ExecutionSpace>( 0, n ), KOKKOS_LAMBDA( int i ) {
            Point xyz;
            translateAndScale( primitives( i ), xyz, scene_bounding_box );
            morton_codes( i ) = morton3D( xyz[0], xyz[1], xyz[2] );
        } );
    Kokkos::fence();
}

template <typename DeviceType>
template <typename Primitives>
inline void TreeConstruction<DeviceType>::assignMortonCodes(
    Primitives const &primitives,
    Kokkos::View<unsigned int *, DeviceType> morton_codes,
    Box const &scene_bounding_box )
{
    auto const n = primitives.extent( 0 );
    DTK_REQUIRE( morton_codes.extent( 0 ) == n );

    using Tag = typename Tag<typename decltype(
        primitives )::traits::non_const_value_type>::type;
    assignMortonCodesDispatch( Tag{}, primitives, morton_codes,
                               scene_bounding_box );
}

template <typename Primitives, typename Indices, typename Nodes>
inline void initializeLeafNodesDispatch( BoxTag, Primitives const &primitives,
                                         Indices permutation_indices,
                                         Nodes leaf_nodes )
{
    using ExecutionSpace =
        typename decltype( primitives )::traits::execution_space;
    auto const n = leaf_nodes.extent( 0 );
    Kokkos::parallel_for(
        DTK_MARK_REGION( "initialize_leaf_nodes" ),
        Kokkos::RangePolicy<ExecutionSpace>( 0, n ), KOKKOS_LAMBDA( int i ) {
            leaf_nodes( i ) = {
                {nullptr, reinterpret_cast<Node *>( permutation_indices( i ) )},
                primitives( permutation_indices( i ) )};
        } );
    Kokkos::fence();
}

template <typename Primitives, typename Indices, typename Nodes>
inline void initializeLeafNodesDispatch( PointTag, Primitives const &primitives,
                                         Indices permutation_indices,
                                         Nodes leaf_nodes )
{
    using ExecutionSpace =
        typename decltype( primitives )::traits::execution_space;
    auto const n = leaf_nodes.extent( 0 );
    Kokkos::parallel_for(
        DTK_MARK_REGION( "initialize_leaf_nodes" ),
        Kokkos::RangePolicy<ExecutionSpace>( 0, n ), KOKKOS_LAMBDA( int i ) {
            leaf_nodes( i ) = {
                {nullptr, reinterpret_cast<Node *>( permutation_indices( i ) )},
                {primitives( permutation_indices( i ) ),
                 primitives( permutation_indices( i ) )}};
        } );
    Kokkos::fence();
}

template <typename DeviceType>
template <typename Primitives>
inline void TreeConstruction<DeviceType>::initializeLeafNodes(
    Primitives const &primitives,
    Kokkos::View<size_t const *, DeviceType> permutation_indices,
    Kokkos::View<Node *, DeviceType> leaf_nodes )
{
    auto const n = leaf_nodes.extent( 0 );
    DTK_REQUIRE( permutation_indices.extent( 0 ) == n );
    DTK_REQUIRE( primitives.extent( 0 ) == n );

    static_assert( sizeof( typename decltype(
                       permutation_indices )::value_type ) == sizeof( Node * ),
                   "Encoding leaf index in pointer to child is not safe if the "
                   "index and pointer types do not have the same size" );

    using Tag = typename Tag<typename decltype(
        primitives )::traits::non_const_value_type>::type;
    initializeLeafNodesDispatch( Tag{}, primitives, permutation_indices,
                                 leaf_nodes );
}

} // namespace Details
} // namespace DataTransferKit

#endif
