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

#include <Teuchos_ArrayView.hpp>
#include <Tpetra_Distributor.hpp>

#include <Kokkos_Parallel.hpp>

#include <mpi.h>

#include <algorithm>

namespace DataTransferKit
{
namespace Details
{

// clang-format off
// copy/paste from https://github.com/kokkos/kokkos-tutorials/blob/ee619447cee9e57b831e34b151d0868958b6b1e5/Intro-Full/Exercises/mpi_exch/mpi_exch.cpp
// * added Experimental nested namespace qualifier for Max
// * declared function static to avoid multiple definition error at link time
// * return early if input argument is empty
// * replace parallel_reduce with std::max_element because I couldn't get it to compile
// * append size to offsets in main driver but might want to handle it here
static void extract_and_sort_ranks(
    Kokkos::View<int*, Kokkos::HostSpace> destination_ranks,
    Kokkos::View<int*, Kokkos::HostSpace> permutation,
    std::vector<int>& unique_ranks,
    std::vector<int>& offsets,
    std::vector<int>& counts) {
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  auto n = destination_ranks.extent(0);
if ( n == 0 ) return;
  using ST = decltype(n);
  Kokkos::View<int*, Kokkos::HostSpace> tmp_ranks("tmp ranks", destination_ranks.extent(0));
  Kokkos::deep_copy(tmp_ranks, destination_ranks);
  int offset = 0;
  // this implements a "sort" which is O(N * R) where (R) is
  // the total number of unique destination ranks.
  // it performs better than other algorithms in
  // the case when (R) is small, but results may vary
  while (true) {
    int next_biggest_rank;
    //Kokkos::parallel_reduce("find next biggest rank", n, KOKKOS_LAMBDA(ST i, int& local_max) {
    //  auto r = tmp_ranks(i);
    //  local_max = (r > local_max) ? r : local_max;
    //}, Kokkos::Experimental::Max<int, Kokkos::HostSpace>(next_biggest_rank));
    next_biggest_rank = *std::max_element(tmp_ranks.data(), tmp_ranks.data() + n);
    if (next_biggest_rank == -1) break;
    unique_ranks.push_back(next_biggest_rank);
    offsets.push_back(offset);
    Kokkos::View<int, Kokkos::HostSpace> total("total");
    Kokkos::parallel_scan("process biggest rank items", Kokkos::RangePolicy<Kokkos::Serial>(0, n),
    KOKKOS_LAMBDA(ST i, int& index, const bool last_pass) {
      if (last_pass && (tmp_ranks(i) == next_biggest_rank)) {
        permutation(i) = index + offset;
      }
      if (tmp_ranks(i) == next_biggest_rank) ++index;
      if (last_pass) {
        if (i + 1 == tmp_ranks.extent(0)) {
          total() = index;
        }
        if (tmp_ranks(i) == next_biggest_rank) {
          tmp_ranks(i) = -1;
        }
      }
    });
    auto host_total = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), total);
    auto count = host_total();
    counts.push_back(count);
    offset += count;
  }
}
// clang-format on

class Distributor
{
  public:
    Distributor( Teuchos::RCP<Teuchos::Comm<int> const> comm )
        : _distributor( comm )
        , _comm(
              *( Teuchos::rcp_dynamic_cast<Teuchos::MpiComm<int> const>( comm )
                     ->getRawMpiComm() ) )
    {
    }

    template <typename T>
    static T *notNullPtr( T *p )
    {
        return p ? p : reinterpret_cast<T *>( -1 );
    }

    size_t
    createFromSends( Teuchos::ArrayView<int const> const &export_proc_ids )
    {
        static_assert(
            std::is_same<typename View::non_const_value_type, int>::value, "" );
        int comm_rank;
        MPI_Comm_rank( _comm, &comm_rank );

        auto const n = export_proc_ids.size();
        Kokkos::View<int const *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>
            tmp( export_proc_ids.getRawPtr(), n );
        Kokkos::View<int *, Kokkos::HostSpace> destination_ranks(
            "destination_ranks", n );
        Kokkos::deep_copy( destination_ranks, tmp );

        _permute = Kokkos::View<int *, Kokkos::HostSpace>( "permute", n );
        std::vector<int> destinations;
        extract_and_sort_ranks( destination_ranks, _permute, destinations,
                                _dest_offsets, _dest_counts );

        if ( _dest_counts.empty() )
            _dest_offsets.push_back( 0 );
        else
            _dest_offsets.push_back( _dest_offsets.back() +
                                     _dest_counts.back() );

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
        assert( weighted == 0 );
        assert( outdegrees == (int)destinations.size() );

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

        int const nn = _distributor.createFromSends( export_proc_ids );
        assert( _src_offsets.back() == nn ); // check return value

        for ( int i = 0; i < indegrees; ++i )
        {
            auto it = std::find( std::begin( _distributor.getProcsFrom() ),
                                 std::end( _distributor.getProcsFrom() ),
                                 sources[i] );
            assert( it != std::end( _distributor.getProcsFrom() ) );
            int j =
                std::distance( std::begin( _distributor.getProcsFrom() ), it );
            assert( _src_counts[i] == (int)_distributor.getLengthsFrom()[j] );
        }

        for ( int i = 0; i < outdegrees; ++i )
        {
            auto it = std::find( std::begin( _distributor.getProcsTo() ),
                                 std::end( _distributor.getProcsTo() ),
                                 destinations[i] );
            assert( it != std::end( _distributor.getProcsTo() ) );
            int j =
                std::distance( std::begin( _distributor.getProcsTo() ), it );
            assert( _dest_counts[i] == (int)_distributor.getLengthsTo()[j] );
        }

        return _src_offsets.back();
    }
    template <typename Packet>
    void doPostsAndWaits( Teuchos::ArrayView<Packet const> const &exports,
                          size_t numPackets,
                          Teuchos::ArrayView<Packet> const &imports )
    {

        std::vector<int> dest_counts = _dest_counts;
        std::vector<int> dest_offsets = _dest_offsets;
        std::vector<int> src_counts = _src_counts;
        std::vector<int> src_offsets = _src_offsets;
        for ( auto pv :
              {&dest_counts, &dest_offsets, &src_counts, &src_offsets} )
            for ( auto &x : *pv )
                x *= numPackets * sizeof( Packet );

        std::vector<Packet> dest_buffer( exports.size() );
        std::vector<Packet> src_buffer( imports.size() );

        assert( (size_t)src_offsets.back() ==
                imports.size() * sizeof( Packet ) );
        assert( (size_t)dest_offsets.back() ==
                exports.size() * sizeof( Packet ) );

        for ( int i = 0; i < _dest_offsets.back(); ++i )
            std::copy( &exports[numPackets * i],
                       &exports[numPackets * i] + numPackets,
                       &dest_buffer[numPackets * _permute[i]] );

        MPI_Neighbor_alltoallv(
            notNullPtr( dest_buffer.data() ), notNullPtr( dest_counts.data() ),
            dest_offsets.data(), MPI_BYTE, notNullPtr( src_buffer.data() ),
            notNullPtr( src_counts.data() ), src_offsets.data(), MPI_BYTE,
            _comm_dist_graph );

        std::copy( src_buffer.begin(), src_buffer.end(), imports.getRawPtr() );

        size_t num_receives = _distributor.getNumReceives();
        size_t num_sends = _distributor.getNumSends();
        if ( _distributor.hasSelfMessage() )
        {
            ++num_receives;
            ++num_sends;
        }
        assert( num_receives == _src_counts.size() );
        assert( num_sends == _dest_counts.size() );

        //_distributor.doPostsAndWaits( exports, numPackets, imports );
    }
    size_t getTotalReceiveLength() const
    {
        auto n = _distributor.getTotalReceiveLength();
        assert( n ==
                std::accumulate( std::begin( _distributor.getLengthsFrom() ),
                                 std::end( _distributor.getLengthsFrom() ),
                                 size_t{0} ) );
        assert( n == static_cast<size_t>( _src_offsets.back() ) );
        return n;
    }
    size_t getTotalSendLength() const
    {
        auto n = std::accumulate( std::begin( _distributor.getLengthsTo() ),
                                  std::end( _distributor.getLengthsTo() ),
                                  size_t{0} );
        assert( n == static_cast<size_t>( _dest_offsets.back() ) );
        return n;
    }

  private:
    Tpetra::Distributor _distributor;
    MPI_Comm _comm;
    MPI_Comm _comm_dist_graph;
    Kokkos::View<int *, Kokkos::HostSpace> _permute;
    std::vector<int> _dest_offsets;
    std::vector<int> _dest_counts;
    std::vector<int> _src_offsets;
    std::vector<int> _src_counts;
};

} // namespace Details
} // namespace DataTransferKit

#endif
