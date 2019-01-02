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
#ifndef DTK_DETAILS_DISTRIBUTOR_HPP
#define DTK_DETAILS_DISTRIBUTOR_HPP

#include <DTK_DBC.hpp>

#include <Kokkos_Core.hpp> // FIXME

#include <mpi.h>

#include <algorithm> // max_element
#include <numeric>   // iota
#include <vector>

#define REORDER_RECV YES

namespace DataTransferKit
{
namespace Details
{

// Computes the array of indices that sort the input array (in reverse order)
// but also returns the sorted unique elements in that array with the
// corresponding element counts and displacement (offsets)
template <typename InputView, typename OutputView>
static void sortAndDetermineBufferLayout( InputView ranks,
                                          OutputView permutation_indices,
                                          std::vector<int> &unique_ranks,
                                          std::vector<int> &counts,
                                          std::vector<int> &offsets )
{
    DTK_REQUIRE( unique_ranks.empty() );
    DTK_REQUIRE( offsets.empty() );
    DTK_REQUIRE( counts.empty() );
    DTK_REQUIRE( permutation_indices.extent( 0 ) == ranks.extent( 0 ) );
    static_assert(
        std::is_same<typename InputView::non_const_value_type, int>::value,
        "" );
    static_assert( std::is_same<typename OutputView::value_type, int>::value,
                   "" );
    static_assert(
        Kokkos::Impl::MemorySpaceAccess<typename OutputView::memory_space,
                                        Kokkos::HostSpace>::accessible,
        "" );

    offsets.push_back( 0 );

    auto const n = ranks.extent_int( 0 );
    if ( n == 0 )
        return;

    Kokkos::View<int *, Kokkos::HostSpace> ranks_duplicate(
        Kokkos::ViewAllocateWithoutInitializing( ranks.label() ),
        ranks.size() );
    Kokkos::deep_copy( ranks_duplicate, ranks );

    while ( true )
    {
        // TODO consider replacing with parallel reduce
        int const largest_rank = *std::max_element(
            ranks_duplicate.data(), ranks_duplicate.data() + n );
        if ( largest_rank == -1 )
            break;
        unique_ranks.push_back( largest_rank );
        counts.push_back( 0 );
        // TODO consider replacing with parallel scan
        for ( int i = 0; i < n; ++i )
        {
            if ( ranks_duplicate( i ) == largest_rank )
            {
                ranks_duplicate( i ) = -1;
                permutation_indices( i ) = offsets.back() + counts.back();
                ++counts.back();
            }
        }
        offsets.push_back( offsets.back() + counts.back() );
    }
}

// A number of MPI routines (in the current OpenMPI implementation at least) are
// checking at runtime that all pointers passed to them as argument are not null
// and call program termination even when no data would actually be
// communicated. This is problematic when using std::vector to store buffers and
// other arrays because std::vector::data() may return a null pointer if the
// vector is empty. This function "swaps" the pointer with some other pointer
// that is not dereferenceable if it is the null pointer.
template <typename T>
static T *notNullPtr( T *p )
{
    return p ? p : reinterpret_cast<T *>( -1 );
}

class Distributor
{
  public:
    Distributor( MPI_Comm comm )
        : _comm( comm )
    {
    }

    template <typename View>
    size_t createFromSends( View const &destination_ranks )
    {
        static_assert( View::rank == 1, "" );
        static_assert(
            std::is_same<typename View::non_const_value_type, int>::value, "" );
        int comm_rank;
        MPI_Comm_rank( _comm, &comm_rank );

        _permute = Kokkos::View<int *, Kokkos::HostSpace>(
            Kokkos::ViewAllocateWithoutInitializing( "permute" ),
            destination_ranks.size() );
        std::vector<int> destinations;
        sortAndDetermineBufferLayout( destination_ranks, _permute, destinations,
                                      _dest_counts, _dest_offsets );

        {
            int const reorder = 0; // Ranking may *not* be reordered
            int const sources[1] = {comm_rank};
            int const degrees[1] = {static_cast<int>( destinations.size() )};

            MPI_Dist_graph_create(
                _comm, 1, sources, degrees, notNullPtr( destinations.data() ),
                MPI_UNWEIGHTED, MPI_INFO_NULL, reorder, &_comm_dist_graph );
        }

        int indegrees;
        int outdegrees;
        int weighted;
        MPI_Dist_graph_neighbors_count( _comm_dist_graph, &indegrees,
                                        &outdegrees, &weighted );
        DTK_ENSURE( weighted == 0 );
        DTK_ENSURE( outdegrees == static_cast<int>( destinations.size() ) );

        std::vector<int> sources( indegrees );
        MPI_Dist_graph_neighbors( _comm_dist_graph, indegrees,
                                  notNullPtr( sources.data() ), MPI_UNWEIGHTED,
                                  outdegrees, notNullPtr( destinations.data() ),
                                  MPI_UNWEIGHTED );

        _src_counts.resize( indegrees );
        MPI_Neighbor_alltoall( _dest_counts.data(), 1, MPI_INT,
                               _src_counts.data(), 1, MPI_INT,
                               _comm_dist_graph );

        _src_offsets.resize( indegrees + 1 );
        // exclusive scan
        _src_offsets[0] = 0;
        for ( int i = 0; i < indegrees; ++i )
            _src_offsets[i + 1] = _src_offsets[i] + _src_counts[i];

#if defined( REORDER_RECV )
        int comm_size = -1;
        MPI_Comm_size( _comm, &comm_size );
        _permute_recv.resize( _src_offsets.back() );
        int offset = 0;
        for ( int i = 0; i < comm_size; ++i )
        {
            auto const it = std::find( sources.begin(), sources.end(), i );
            if ( it != sources.end() )
            {
                int const j = std::distance( sources.begin(), it );
                std::iota( &_permute_recv[offset],
                           &_permute_recv[offset] + _src_counts[j],
                           _src_offsets[j] );
                offset += _src_counts[j];
            }
        }
        DTK_ENSURE( offset == static_cast<int>( _permute_recv.size() ) );
#endif

        return _src_offsets.back();
    }
    template <typename View>
    void doPostsAndWaits( typename View::const_type const &exports,
                          size_t num_packets, View const &imports ) const
    {
        DTK_REQUIRE( num_packets * _src_offsets.back() == imports.size() );
        DTK_REQUIRE( num_packets * _dest_offsets.back() == exports.size() );

        using ValueType = typename View::value_type;
        static_assert( View::rank == 1, "" );
        static_assert(
            Kokkos::Impl::MemorySpaceAccess<typename View::memory_space,
                                            Kokkos::HostSpace>::accessible,
            "" );

        std::vector<int> dest_counts = _dest_counts;
        std::vector<int> dest_offsets = _dest_offsets;
        std::vector<int> src_counts = _src_counts;
        std::vector<int> src_offsets = _src_offsets;
        for ( auto pv :
              {&dest_counts, &dest_offsets, &src_counts, &src_offsets} )
            for ( auto &x : *pv )
                x *= num_packets * sizeof( ValueType );

        std::vector<ValueType> dest_buffer( exports.size() );
        std::vector<ValueType> src_buffer( imports.size() );

        // TODO
        // * apply permutation on the device in a parallel for
        // * switch to MPI with CUDA support (do not copy to host)
        for ( int i = 0; i < _dest_offsets.back(); ++i )
            std::copy( &exports[num_packets * i],
                       &exports[num_packets * i] + num_packets,
                       &dest_buffer[num_packets * _permute[i]] );

        MPI_Neighbor_alltoallv(
            notNullPtr( dest_buffer.data() ), notNullPtr( dest_counts.data() ),
            dest_offsets.data(), MPI_BYTE, notNullPtr( src_buffer.data() ),
            notNullPtr( src_counts.data() ), src_offsets.data(), MPI_BYTE,
            _comm_dist_graph );

#if defined( REORDER_RECV )
        for ( int i = 0; i < _src_offsets.back(); ++i )
            std::copy( &src_buffer[num_packets * _permute_recv[i]],
                       &src_buffer[num_packets * _permute_recv[i]] +
                           num_packets,
                       &imports[num_packets * i] );
#else
        std::copy( src_buffer.begin(), src_buffer.end(), imports.data() );
#endif
    }
    size_t getTotalReceiveLength() const { return _src_offsets.back(); }
    size_t getTotalSendLength() const { return _dest_offsets.back(); }

  private:
    MPI_Comm _comm;
    MPI_Comm _comm_dist_graph;
    Kokkos::View<int *, Kokkos::HostSpace> _permute;
    std::vector<int> _dest_offsets;
    std::vector<int> _dest_counts;
    std::vector<int> _src_offsets;
    std::vector<int> _src_counts;
#if defined( REORDER_RECV )
    std::vector<int> _permute_recv;
#endif
};

} // namespace Details
} // namespace DataTransferKit

#endif
