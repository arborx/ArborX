/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/
#include <DTK_KokkosHelpers.hpp>
#include <DTK_TreeConstruction.hpp>
#include <details/DTK_DetailsAlgorithms.hpp>

#include <Kokkos_ArithTraits.hpp>
#include <Teuchos_UnitTestHarness.hpp>

#include <algorithm>
#include <bitset>
#include <sstream>
#include <vector>

namespace dtk = DataTransferKit::Details;

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DetailsBVH, morton_codes, NO )
{
    std::vector<DataTransferKit::Point> points = {
        {0.0, 0.0, 0.0},          {0.25, 0.75, 0.25}, {0.75, 0.25, 0.25},
        {0.75, 0.75, 0.25},       {1.33, 2.33, 3.33}, {1.66, 2.66, 3.66},
        {1024.0, 1024.0, 1024.0},
    };
    int const n = points.size();
    // lower left front corner corner of the octant the points fall in
    std::vector<std::array<unsigned int, 3>> anchors = {
        {0, 0, 0}, {0, 0, 0}, {0, 0, 0},         {0, 0, 0},
        {1, 2, 3}, {1, 2, 3}, {1023, 1023, 1023}};
    auto fun = []( std::array<unsigned int, 3> const &anchor ) {
        unsigned int i = std::get<0>( anchor );
        unsigned int j = std::get<1>( anchor );
        unsigned int k = std::get<2>( anchor );
        return 4 * dtk::expandBits( i ) + 2 * dtk::expandBits( j ) +
               dtk::expandBits( k );
    };
    std::vector<unsigned int> ref( n,
                                   Kokkos::ArithTraits<unsigned int>::max() );
    for ( int i = 0; i < n; ++i )
        ref[i] = fun( anchors[i] );
    // using points rather than boxes for convenience here but still have to
    // build the axis-aligned bounding boxes around them
    using DeviceType = typename NO::device_type;
    Kokkos::View<DataTransferKit::Box *, DeviceType> boxes( "boxes", n );
    for ( int i = 0; i < n; ++i )
        dtk::expand( boxes[i], points[i] );

    Kokkos::View<DataTransferKit::Box *, DeviceType> scene( "scene", 1 );
    dtk::TreeConstruction<NO> tc;
    tc.calculateBoundingBoxOfTheScene( boxes, scene[0] );

    // Copy the result on the host
    auto scene_host = Kokkos::create_mirror_view( scene );
    Kokkos::deep_copy( scene_host, scene );

    for ( int d = 0; d < 3; ++d )
    {
        TEST_EQUALITY( scene_host[0][2 * d + 0], 0.0 );
        TEST_EQUALITY( scene_host[0][2 * d + 1], 1024.0 );
    }

    Kokkos::View<unsigned int *, DeviceType> morton_codes( "morton_codes", n );
    tc.assignMortonCodes( boxes, morton_codes, scene[0] );
    auto morton_codes_host = Kokkos::create_mirror_view( morton_codes );
    Kokkos::deep_copy( morton_codes_host, morton_codes );
    TEST_COMPARE_ARRAYS( morton_codes_host, ref );
}

template <typename DeviceType>
class FillK
{
  public:
    KOKKOS_INLINE_FUNCTION
    FillK( Kokkos::View<unsigned int *, DeviceType> k )
        : _k( k )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i ) const { _k[i] = 4 - i; }

  private:
    Kokkos::View<unsigned int *, DeviceType> _k;
};

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DetailsBVH, indirect_sort, NO )
{
    // need a functionality that sort objects based on their Morton code and
    // also returns the indices in the original configuration

    // dummy unsorted Morton codes and corresponding sorted indices as reference
    // solution
    //
    using DeviceType = typename NO::device_type;
    using ExecutionSpace = typename DeviceType::execution_space;
    unsigned int const n = 4;
    Kokkos::View<unsigned int *, DeviceType> k( "k", n );
    // Fill K with 4, 3, 2, 1
    FillK<DeviceType> fill_k_functor( k );
    Kokkos::parallel_for( "fill_k", Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                          fill_k_functor );

    std::vector<int> ref = {3, 2, 1, 0};
    // distribute ids to unsorted objects
    Kokkos::View<int *, DeviceType> ids( "ids", n );
    DataTransferKit::Iota<DeviceType> fill_ids_functor( ids );
    Kokkos::parallel_for( "fill_ids",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                          fill_ids_functor );

    // sort morton codes and object ids
    dtk::TreeConstruction<NO> tc;
    tc.sortObjects( k, ids );

    auto k_host = Kokkos::create_mirror_view( k );
    Kokkos::deep_copy( k_host, k );
    auto ids_host = Kokkos::create_mirror_view( ids );
    Kokkos::deep_copy( ids_host, ids );

    // check that they are sorted
    for ( unsigned int i = 0; i < n; ++i )
        TEST_EQUALITY( k_host[i], i + 1 );
    // check that ids are properly ordered
    TEST_COMPARE_ARRAYS( ids_host, ref );
}

TEUCHOS_UNIT_TEST( DetailsBVH, number_of_leading_zero_bits )
{
    TEST_EQUALITY( dtk::countLeadingZeros( 0 ), 32 );
    TEST_EQUALITY( dtk::countLeadingZeros( 1 ), 31 );
    TEST_EQUALITY( dtk::countLeadingZeros( 2 ), 30 );
    TEST_EQUALITY( dtk::countLeadingZeros( 3 ), 30 );
    TEST_EQUALITY( dtk::countLeadingZeros( 4 ), 29 );
    TEST_EQUALITY( dtk::countLeadingZeros( 5 ), 29 );
    TEST_EQUALITY( dtk::countLeadingZeros( 6 ), 29 );
    TEST_EQUALITY( dtk::countLeadingZeros( 7 ), 29 );
    TEST_EQUALITY( dtk::countLeadingZeros( 8 ), 28 );
    TEST_EQUALITY( dtk::countLeadingZeros( 9 ), 28 );
    // bitwise exclusive OR operator to compare bits
    TEST_EQUALITY( dtk::countLeadingZeros( 1 ^ 0 ), 31 );
    TEST_EQUALITY( dtk::countLeadingZeros( 2 ^ 0 ), 30 );
    TEST_EQUALITY( dtk::countLeadingZeros( 2 ^ 1 ), 30 );
    TEST_EQUALITY( dtk::countLeadingZeros( 3 ^ 0 ), 30 );
    TEST_EQUALITY( dtk::countLeadingZeros( 3 ^ 1 ), 30 );
    TEST_EQUALITY( dtk::countLeadingZeros( 3 ^ 2 ), 31 );
    TEST_EQUALITY( dtk::countLeadingZeros( 4 ^ 0 ), 29 );
    TEST_EQUALITY( dtk::countLeadingZeros( 4 ^ 1 ), 29 );
    TEST_EQUALITY( dtk::countLeadingZeros( 4 ^ 2 ), 29 );
    TEST_EQUALITY( dtk::countLeadingZeros( 4 ^ 3 ), 29 );
}

template <typename DeviceType>
class FillFi
{
  public:
    KOKKOS_INLINE_FUNCTION
    FillFi( Kokkos::View<unsigned int *, DeviceType> fi )
        : _fi( fi )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i ) const
    {
        // NOTE: Morton codes below are **not** unique
        unsigned int fi_array[] = {0,  1,  1,  2,  3,  5,  8,
                                   13, 21, 34, 55, 89, 144};

        _fi[i] = fi_array[i];
    }

  private:
    Kokkos::View<unsigned int *, DeviceType> _fi;
};

template <typename NO>
class ComputeResults
{
  public:
    using DeviceType = typename NO::device_type;
    KOKKOS_INLINE_FUNCTION
    ComputeResults( Kokkos::View<unsigned int *, DeviceType> fi,
                    Kokkos::View<int *, DeviceType> results, int n )
        : _fi( fi )
        , _results( results )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( int const i ) const
    {
        int index_1[] = {0, 0, 1, 1, 1, 2, 2, 0, 12, 12};
        int index_2[] = {0, 1, 0, 1, 2, 1, 2, -1, 12, 13};

        _results[i] =
            DataTransferKit::Details::TreeConstruction<NO>::commonPrefix(
                _fi, index_1[i], index_2[i] );
    }

  private:
    Kokkos::View<unsigned int *, DeviceType> _fi;
    Kokkos::View<int *, DeviceType> _results;
};

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DetailsBVH, common_prefix, NO )
{
    using DeviceType = typename NO::device_type;
    using ExecutionSpace = typename DeviceType::execution_space;
    int const n = 13;
    Kokkos::View<unsigned int *, DeviceType> fi( "fi", n );
    FillFi<DeviceType> fill_fi_functor( fi );
    Kokkos::parallel_for( "fill_fi",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
                          fill_fi_functor );
    Kokkos::fence();

    int const n_tests = 10;
    Kokkos::View<int *, DeviceType> results( "results", n_tests );

    ComputeResults<NO> compute_results_functor( fi, results, n );
    Kokkos::parallel_for( "compute_results",
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n_tests ),
                          compute_results_functor );
    Kokkos::fence();

    auto results_host = Kokkos::create_mirror_view( results );
    Kokkos::deep_copy( results_host, results );

    auto fi_host = Kokkos::create_mirror_view( fi );
    Kokkos::deep_copy( fi_host, fi );

    TEST_EQUALITY( results_host[0], 32 + 32 );
    TEST_EQUALITY( results_host[1], 31 );
    TEST_EQUALITY( results_host[2], 31 );
    // duplicate Morton codes
    TEST_EQUALITY( fi_host[1], 1 );
    TEST_EQUALITY( fi_host[1], fi_host[2] );
    TEST_EQUALITY( results_host[3], 64 );
    TEST_EQUALITY( results_host[4], 32 + 30 );
    TEST_EQUALITY( results_host[5], 62 );
    TEST_EQUALITY( results_host[6], 64 );
    // by definition \delta(i, j) = -1 when j \notin [0, n-1]
    TEST_EQUALITY( results_host[7], -1 );
    TEST_EQUALITY( results_host[8], 64 );
    TEST_EQUALITY( results_host[9], -1 );
}

TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( DetailsBVH, example_tree_construction, NO )
{
    // This is the example from the articles by Karras.
    // See
    // https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
    using DeviceType = typename NO::device_type;
    int const n = 8;
    Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes(
        "sorted_morton_codes", n );
    std::vector<std::string> s{
        "00001", "00010", "00100", "00101", "10011", "11000", "11001", "11110",
    };
    for ( int i = 0; i < n; ++i )
    {
        std::bitset<6> b( s[i] );
        std::cout << b << "  " << b.to_ulong() << "\n";
        sorted_morton_codes[i] = b.to_ulong();
    }

    // reference solution for a recursive traversal from top to bottom
    // starting from root, visiting first the left child and then the right one
    std::ostringstream ref;
    ref << "I0"
        << "I3"
        << "I1"
        << "L0"
        << "L1"
        << "I2"
        << "L2"
        << "L3"
        << "I4"
        << "L4"
        << "I5"
        << "I6"
        << "L5"
        << "L6"
        << "L7";
    std::cout << "ref=" << ref.str() << "\n";

    // hierarchy generation
    Kokkos::View<DataTransferKit::Node *, DeviceType> leaf_nodes( "leaf_nodes",
                                                                  n );
    Kokkos::View<DataTransferKit::Node *, DeviceType> internal_nodes(
        "internal_nodes", n - 1 );
    std::function<void( DataTransferKit::Node *, std::ostream & )>
        traverseRecursive;
    traverseRecursive = [&leaf_nodes, &internal_nodes, &traverseRecursive](
        DataTransferKit::Node *node, std::ostream &os ) {
        if ( std::any_of( leaf_nodes.data(), leaf_nodes.data() + n,
                          [node]( DataTransferKit::Node const &leaf_node ) {
                              return std::addressof( leaf_node ) == node;
                          } ) )
        {
            os << "L" << node - leaf_nodes.data();
        }
        else
        {
            os << "I" << node - internal_nodes.data();
            for ( DataTransferKit::Node *child :
                  {node->children.first, node->children.second} )
                traverseRecursive( child, os );
        }
    };

    dtk::TreeConstruction<NO> tc;
    tc.generateHierarchy( sorted_morton_codes, leaf_nodes, internal_nodes );

    DataTransferKit::Node *root = internal_nodes.data();
    TEST_ASSERT( root->parent == nullptr );

    std::ostringstream sol;
    traverseRecursive( root, sol );
    std::cout << "sol=" << sol.str() << "\n";

    TEST_EQUALITY( sol.str().compare( ref.str() ), 0 );
}

// Include the test macros.
#include "DataTransferKitSearch_ETIHelperMacros.h"

// Create the test group
#define UNIT_TEST_GROUP( NODE )                                                \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DetailsBVH, morton_codes, NODE )     \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DetailsBVH, indirect_sort, NODE )    \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DetailsBVH, common_prefix, NODE )    \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( DetailsBVH,                          \
                                          example_tree_construction, NODE )
// Demangle the types
DTK_ETI_MANGLING_TYPEDEFS()

// Instantiate the tests
DTK_INSTANTIATE_N( UNIT_TEST_GROUP )
