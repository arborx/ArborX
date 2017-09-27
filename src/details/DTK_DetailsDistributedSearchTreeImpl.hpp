/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/
#ifndef DTK_DETAILS_DISTRIBUTED_SEARCH_TREE_IMPL_HPP
#define DTK_DETAILS_DISTRIBUTED_SEARCH_TREE_IMPL_HPP

#include <DTK_LinearBVH.hpp>
#include <details/DTK_DetailsPredicate.hpp>
#include <details/DTK_DetailsPriorityQueue.hpp>
#include <details/DTK_DetailsUtils.hpp>

#include <Kokkos_Atomic.hpp>
#include <Kokkos_Sort.hpp>
#include <Teuchos_SerializationTraits.hpp>
#include <Tpetra_Distributor.hpp>

namespace Teuchos
{

template <typename Ordinal>
class SerializationTraits<Ordinal, DataTransferKit::Details::Within>
    : public DirectSerializationTraits<Ordinal,
                                       DataTransferKit::Details::Within>
{
};
template <typename Ordinal>
class SerializationTraits<Ordinal, DataTransferKit::Details::Nearest>
    : public DirectSerializationTraits<Ordinal,
                                       DataTransferKit::Details::Nearest>
{
};
template <typename Ordinal>
class SerializationTraits<Ordinal, DataTransferKit::Details::Overlap>
    : public DirectSerializationTraits<Ordinal,
                                       DataTransferKit::Details::Overlap>
{
};

} // end namespace Teuchos

namespace DataTransferKit
{

template <typename DeviceType>
struct DistributedSearchTreeImpl
{
    using ExecutionSpace = typename DeviceType::execution_space;

    // spatial queries
    template <typename Query>
    static void queryDispatch( Teuchos::RCP<Teuchos::Comm<int> const> comm,
                               BVH<DeviceType> const &distributed_tree,
                               BVH<DeviceType> const &local_tree,
                               Kokkos::View<Query *, DeviceType> queries,
                               Kokkos::View<int *, DeviceType> &indices,
                               Kokkos::View<int *, DeviceType> &offset,
                               Kokkos::View<int *, DeviceType> &ranks,
                               Details::SpatialPredicateTag );

    // nearest neighbors queries
    template <typename Query>
    static void queryDispatch(
        Teuchos::RCP<Teuchos::Comm<int> const> comm,
        BVH<DeviceType> const &distributed_tree,
        BVH<DeviceType> const &local_tree,
        Kokkos::View<Query *, DeviceType> queries,
        Kokkos::View<int *, DeviceType> &indices,
        Kokkos::View<int *, DeviceType> &offset,
        Kokkos::View<int *, DeviceType> &ranks, Details::NearestPredicateTag,
        Kokkos::View<double *, DeviceType> *distances_ptr = nullptr );

    template <typename Query>
    static void deviseStrategy( Point epsilon,
                                Kokkos::View<Query *, DeviceType> queries,
                                BVH<DeviceType> const &bvh,
                                Kokkos::View<int *, DeviceType> &indices,
                                Kokkos::View<int *, DeviceType> &offset );

    template <typename Query>
    static void forwardQueries( Teuchos::RCP<Teuchos::Comm<int> const> comm,
                                Kokkos::View<Query *, DeviceType> queries,
                                Kokkos::View<int *, DeviceType> indices,
                                Kokkos::View<int *, DeviceType> offset,
                                Kokkos::View<Query *, DeviceType> &fwd_queries,
                                Kokkos::View<int *, DeviceType> &fwd_ids,
                                Kokkos::View<int *, DeviceType> &fwd_ranks );

    static void communicateResultsBack(
        Teuchos::RCP<Teuchos::Comm<int> const> comm,
        Kokkos::View<int *, DeviceType> &indices,
        Kokkos::View<int *, DeviceType> offset,
        Kokkos::View<int *, DeviceType> &ranks,
        Kokkos::View<int *, DeviceType> &ids,
        Kokkos::View<double *, DeviceType> *distances_ptr = nullptr );

    template <typename Query>
    static void filterResults( Kokkos::View<Query *, DeviceType> queries,
                               Kokkos::View<double *, DeviceType> distances,
                               Kokkos::View<int *, DeviceType> &indices,
                               Kokkos::View<int *, DeviceType> &offset,
                               Kokkos::View<int *, DeviceType> &ranks );
    static void
    sortResults( Kokkos::View<int *, DeviceType> query_ids,
                 Kokkos::View<int *, DeviceType> results,
                 Kokkos::View<int *, DeviceType> ranks,
                 Kokkos::View<double *, DeviceType> *distances_ptr = nullptr );

    static void countResults( int n_queries,
                              Kokkos::View<int *, DeviceType> query_ids,
                              Kokkos::View<int *, DeviceType> &offset );

    // NOTE: Would love to pass the distributor as a const reference but
    // unfortunately the methods for executing the communication plan (e.g.
    // doPostsAndWaits() in this case) are not declared with the const
    // qualifier in Tpetra.
    template <typename T>
    static void sendAcrossNetwork( Tpetra::Distributor &distributor,
                                   Kokkos::View<T *, DeviceType> exports,
                                   Kokkos::View<T *, DeviceType> imports );

    static double epsilon;
};

// Default value for epsilon matches the inclusion tolerance in DTK-2.0 which
// is arbitrary and might need adjustement in client code.  See
// https://github.com/ORNL-CEES/DataTransferKit/blob/dtk-2.0/packages/Operators/src/Search/DTK_CoarseGlobalSearch.cpp#L61
template <typename DeviceType>
double DistributedSearchTreeImpl<DeviceType>::epsilon = 1.0e-6;

template <typename DeviceType>
template <typename T>
void DistributedSearchTreeImpl<DeviceType>::sendAcrossNetwork(
    Tpetra::Distributor &distributor, Kokkos::View<T *, DeviceType> exports,
    Kokkos::View<T *, DeviceType> imports )
{
    // NOTE: this function encapsulates the communication from and to views
    // with data living on the device.  This is a workaround we will
    // (hopefully) get rid of in the future when we upgrade Trilinos.  We
    // should be able to directly call doPostsAndWaits() on the views passed as
    // argument.  See https://github.com/trilinos/Trilinos/issues/1454
    auto exports_host = Kokkos::create_mirror_view( exports );
    Kokkos::deep_copy( exports_host, exports );
    auto imports_host = Kokkos::create_mirror_view( imports );
    distributor.doPostsAndWaits(
        Teuchos::ArrayView<T const>( exports_host.data(),
                                     exports_host.extent( 0 ) ),
        1,
        Teuchos::ArrayView<T>( imports_host.data(),
                               imports_host.extent( 0 ) ) );
    Kokkos::deep_copy( imports, imports_host );
}

template <typename DeviceType>
template <typename Query>
void DistributedSearchTreeImpl<DeviceType>::deviseStrategy(
    Point epsilon, Kokkos::View<Query *, DeviceType> queries,
    BVH<DeviceType> const &bvh, Kokkos::View<int *, DeviceType> &indices,
    Kokkos::View<int *, DeviceType> &offset )
{
    int const n_queries = queries.extent( 0 );
    Kokkos::View<Details::Overlap *, DeviceType> overlap_queries(
        "overlap_queries", n_queries );

    Kokkos::parallel_for( REGION_NAME( "fill_overlap_queries" ),
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
                          KOKKOS_LAMBDA( int q ) {
                              Point point = queries( q )._query_point;
                              Box box( {
                                  point[0] - epsilon[0],
                                  point[0] + epsilon[0],
                                  point[1] - epsilon[1],
                                  point[1] + epsilon[1],
                                  point[2] - epsilon[2],
                                  point[2] + epsilon[2],
                              } );
                              overlap_queries( q ) = Details::Overlap( box );
                          } );
    Kokkos::fence();

    bvh.query( overlap_queries, indices, offset );
}

template <typename DeviceType>
template <typename Query>
void DistributedSearchTreeImpl<DeviceType>::queryDispatch(
    Teuchos::RCP<Teuchos::Comm<int> const> comm,
    BVH<DeviceType> const &distributed_tree, BVH<DeviceType> const &local_tree,
    Kokkos::View<Query *, DeviceType> queries,
    Kokkos::View<int *, DeviceType> &indices,
    Kokkos::View<int *, DeviceType> &offset,
    Kokkos::View<int *, DeviceType> &ranks, Details::NearestPredicateTag,
    Kokkos::View<double *, DeviceType> *distances_ptr )
{
    // Determine what ranks have local trees that the objects associated with
    // the nearest neighbors queries oroverlap with when expanded in all
    // direction by epsilon.
    // NOTE: epsilon is a static member for now which is far from ideal.
    deviseStrategy( {{epsilon, epsilon, epsilon}}, queries, distributed_tree,
                    indices, offset );

    ////////////////////////////////////////////////////////////////////////////
    // Forward queries
    ////////////////////////////////////////////////////////////////////////////
    Kokkos::View<int *, DeviceType> ids( "query_ids" );
    Kokkos::View<Query *, DeviceType> fwd_queries( "fwd_queries" );
    forwardQueries( comm, queries, indices, offset, fwd_queries, ids, ranks );
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // Perform queries that have been received
    ////////////////////////////////////////////////////////////////////////////
    Kokkos::View<double *, DeviceType> distances( "distances" );
    if ( distances_ptr )
        distances = *distances_ptr;
    local_tree.query( fwd_queries, indices, offset, distances );
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // Communicate results back
    ////////////////////////////////////////////////////////////////////////////
    communicateResultsBack( comm, indices, offset, ranks, ids, &distances );
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // Merge results
    ////////////////////////////////////////////////////////////////////////////
    int const n_queries = queries.extent_int( 0 );
    countResults( n_queries, ids, offset );
    sortResults( ids, indices, ranks, &distances );
    filterResults( queries, distances, indices, offset, ranks );
    ////////////////////////////////////////////////////////////////////////////
}

template <typename DeviceType>
template <typename Query>
void DistributedSearchTreeImpl<DeviceType>::queryDispatch(
    Teuchos::RCP<Teuchos::Comm<int> const> comm,
    BVH<DeviceType> const &distributed_tree, BVH<DeviceType> const &local_tree,
    Kokkos::View<Query *, DeviceType> queries,
    Kokkos::View<int *, DeviceType> &indices,
    Kokkos::View<int *, DeviceType> &offset,
    Kokkos::View<int *, DeviceType> &ranks, Details::SpatialPredicateTag )
{
    ////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    distributed_tree.query( queries, indices, offset );
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // Forward queries
    ////////////////////////////////////////////////////////////////////////////
    Kokkos::View<int *, DeviceType> ids( "query_ids" );
    Kokkos::View<Query *, DeviceType> fwd_queries( "fwd_queries" );
    forwardQueries( comm, queries, indices, offset, fwd_queries, ids, ranks );
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // Perform queries that have been received
    ////////////////////////////////////////////////////////////////////////////
    local_tree.query( fwd_queries, indices, offset );
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // Communicate results back
    ////////////////////////////////////////////////////////////////////////////
    communicateResultsBack( comm, indices, offset, ranks, ids );
    ////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////
    // Merge results
    ////////////////////////////////////////////////////////////////////////////
    int const n_queries = queries.extent_int( 0 );
    countResults( n_queries, ids, offset );
    sortResults( ids, indices, ranks );
    ////////////////////////////////////////////////////////////////////////////
}

template <typename DeviceType>
void DistributedSearchTreeImpl<DeviceType>::sortResults(
    Kokkos::View<int *, DeviceType> query_ids,
    Kokkos::View<int *, DeviceType> results,
    Kokkos::View<int *, DeviceType> ranks,
    Kokkos::View<double *, DeviceType> *distances_ptr )
{
    int const n = query_ids.extent( 0 );
    // If they were no queries, min_val and max_val values won't change after
    // the parallel reduce (they are initialized to +infty and -infty
    // respectively) and the sort will hang.
    if ( n == 0 )
        return;

    typedef Kokkos::BinOp1D<Kokkos::View<int *, DeviceType>> CompType;

    Kokkos::Experimental::MinMaxScalar<int> result;
    Kokkos::Experimental::MinMax<int> reducer( result );
    parallel_reduce(
        Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
        Kokkos::Impl::min_max_functor<Kokkos::View<int *, DeviceType>>(
            query_ids ),
        reducer );
    if ( result.min_val == result.max_val )
        return;
    Kokkos::BinSort<Kokkos::View<int *, DeviceType>, CompType> bin_sort(
        query_ids, CompType( n / 2, result.min_val, result.max_val ), true );
    bin_sort.create_permute_vector();
    bin_sort.sort( results );
    bin_sort.sort( ranks );
    if ( distances_ptr )
        bin_sort.sort( *distances_ptr );
    Kokkos::fence();
}

template <typename DeviceType>
void DistributedSearchTreeImpl<DeviceType>::countResults(
    int n_queries, Kokkos::View<int *, DeviceType> query_ids,
    Kokkos::View<int *, DeviceType> &offset )
{
    int const nnz = query_ids.extent( 0 );

    Kokkos::realloc( offset, n_queries + 1 );
    fill( offset, 0 );

    Kokkos::parallel_for(
        REGION_NAME( "count_results_per_query" ),
        Kokkos::RangePolicy<ExecutionSpace>( 0, nnz ), KOKKOS_LAMBDA( int i ) {
            Kokkos::atomic_increment( &offset( query_ids( i ) ) );
        } );
    Kokkos::fence();

    exclusivePrefixSum( offset );
}

template <typename DeviceType>
template <typename Query>
void DistributedSearchTreeImpl<DeviceType>::forwardQueries(
    Teuchos::RCP<Teuchos::Comm<int> const> comm,
    Kokkos::View<Query *, DeviceType> queries,
    Kokkos::View<int *, DeviceType> indices,
    Kokkos::View<int *, DeviceType> offset,
    Kokkos::View<Query *, DeviceType> &fwd_queries,
    Kokkos::View<int *, DeviceType> &fwd_ids,
    Kokkos::View<int *, DeviceType> &fwd_ranks )
{
    int const comm_rank = comm->getRank();

    Tpetra::Distributor distributor( comm );

    int const n_queries = queries.extent( 0 );
    int const n_exports = offset( n_queries );
    int const n_imports = distributor.createFromSends(
        Teuchos::ArrayView<int>( indices.data(), n_exports ) );

    Kokkos::View<Query *, DeviceType> exports( queries.label(), n_exports );
    Kokkos::parallel_for( REGION_NAME( "forward_queries_fill_buffer" ),
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
                          KOKKOS_LAMBDA( int q ) {
                              for ( int i = offset( q ); i < offset( q + 1 );
                                    ++i )
                              {
                                  exports( i ) = queries( q );
                              }
                          } );
    Kokkos::fence();

    Kokkos::View<int *, DeviceType> export_ranks( "export_ranks", n_exports );
    fill( export_ranks, comm_rank );

    Kokkos::View<int *, DeviceType> import_ranks( "import_ranks", n_imports );
    sendAcrossNetwork( distributor, export_ranks, import_ranks );

    Kokkos::View<int *, DeviceType> export_ids( "export_ids", n_exports );
    Kokkos::parallel_for( REGION_NAME( "forward_queries_fill_ids" ),
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
                          KOKKOS_LAMBDA( int q ) {
                              for ( int i = offset( q ); i < offset( q + 1 );
                                    ++i )
                              {
                                  export_ids( i ) = q;
                              }
                          } );
    Kokkos::fence();
    Kokkos::View<int *, DeviceType> import_ids( "import_ids", n_imports );
    sendAcrossNetwork( distributor, export_ids, import_ids );

    // Send queries across the network
    Kokkos::View<Query *, DeviceType> imports( queries.label(), n_imports );
    sendAcrossNetwork( distributor, exports, imports );

    fwd_queries = imports;
    fwd_ids = import_ids;
    fwd_ranks = import_ranks;
}

template <typename DeviceType>
void DistributedSearchTreeImpl<DeviceType>::communicateResultsBack(
    Teuchos::RCP<Teuchos::Comm<int> const> comm,
    Kokkos::View<int *, DeviceType> &indices,
    Kokkos::View<int *, DeviceType> offset,
    Kokkos::View<int *, DeviceType> &ranks,
    Kokkos::View<int *, DeviceType> &ids,
    Kokkos::View<double *, DeviceType> *distances_ptr )
{
    int const comm_rank = comm->getRank();

    int const n_fwd_queries = offset.extent_int( 0 ) - 1;
    int const n_exports = offset( n_fwd_queries );
    Kokkos::View<int *, DeviceType> export_ranks( ranks.label(), n_exports );
    Kokkos::parallel_for(
        REGION_NAME( "setup_communication_plan" ),
        Kokkos::RangePolicy<ExecutionSpace>( 0, n_fwd_queries ),
        KOKKOS_LAMBDA( int q ) {
            for ( int i = offset( q ); i < offset( q + 1 ); ++i )
            {
                export_ranks( i ) = ranks( q );
            }
        } );
    Kokkos::fence();

    Tpetra::Distributor distributor( comm );
    int const n_imports = distributor.createFromSends(
        Teuchos::ArrayView<int>( export_ranks.data(), n_exports ) );

    // export_ranks already has adequate size since it was used as a buffer to
    // make the new communication plan.
    fill( export_ranks, comm_rank );

    Kokkos::View<int *, DeviceType> export_ids( ids.label(), n_exports );
    Kokkos::parallel_for(
        REGION_NAME( "fill_buffer" ),
        Kokkos::RangePolicy<ExecutionSpace>( 0, n_fwd_queries ),
        KOKKOS_LAMBDA( int q ) {
            for ( int i = offset( q ); i < offset( q + 1 ); ++i )
            {
                export_ids( i ) = ids( q );
            }
        } );
    Kokkos::fence();
    Kokkos::View<int *, DeviceType> export_indices = indices;

    Kokkos::View<int *, DeviceType> import_indices( indices.label(),
                                                    n_imports );
    Kokkos::View<int *, DeviceType> import_ranks( ranks.label(), n_imports );
    Kokkos::View<int *, DeviceType> import_ids( ids.label(), n_imports );
    sendAcrossNetwork( distributor, export_indices, import_indices );
    sendAcrossNetwork( distributor, export_ranks, import_ranks );
    sendAcrossNetwork( distributor, export_ids, import_ids );

    ids = import_ids;
    ranks = import_ranks;
    indices = import_indices;

    if ( distances_ptr )
    {
        Kokkos::View<double *, DeviceType> &distances = *distances_ptr;
        Kokkos::View<double *, DeviceType> export_distances = distances;
        Kokkos::View<double *, DeviceType> import_distances( distances.label(),
                                                             n_imports );
        sendAcrossNetwork( distributor, export_distances, import_distances );
        distances = import_distances;
    }
}

template <typename DeviceType>
template <typename Query>
void DistributedSearchTreeImpl<DeviceType>::filterResults(
    Kokkos::View<Query *, DeviceType> queries,
    Kokkos::View<double *, DeviceType> distances,
    Kokkos::View<int *, DeviceType> &indices,
    Kokkos::View<int *, DeviceType> &offset,
    Kokkos::View<int *, DeviceType> &ranks )
{
    int const n_queries = queries.extent_int( 0 );
    // truncated views are prefixed with an underscore
    Kokkos::View<int *, DeviceType> _offset( offset.label(), n_queries + 1 );
    fill( _offset, 0 );

    Kokkos::parallel_for( REGION_NAME( "discard_results" ),
                          Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
                          KOKKOS_LAMBDA( int q ) {
                              _offset( q ) = KokkosHelpers::min(
                                  offset( q + 1 ) - offset( q ),
                                  queries( q )._k );
                          } );
    Kokkos::fence();

    exclusivePrefixSum( _offset );

    int const n_truncated_results = _offset( n_queries );
    Kokkos::View<int *, DeviceType> _indices( indices.label(),
                                              n_truncated_results );
    Kokkos::View<int *, DeviceType> _ranks( ranks.label(),
                                            n_truncated_results );

    using PairIndexDistance = Kokkos::pair<Kokkos::Array<int, 2>, double>;
    struct CompareDistance
    {
        KOKKOS_INLINE_FUNCTION bool operator()( PairIndexDistance const &lhs,
                                                PairIndexDistance const &rhs )
        {
            // reverse order (larger distance means lower priority)
            return lhs.second > rhs.second;
        }
    };
    using PriorityQueue =
        Details::PriorityQueue<PairIndexDistance, CompareDistance>;

    Kokkos::parallel_for(
        REGION_NAME( "truncate_results" ),
        Kokkos::RangePolicy<ExecutionSpace>( 0, n_queries ),
        KOKKOS_LAMBDA( int q ) {
            PriorityQueue queue;
            for ( int i = offset( q ); i < offset( q + 1 ); ++i )
                queue.push( Kokkos::Array<int, 2>{{indices( i ), ranks( i )}},
                            distances( i ) );

            int count = 0;
            while ( !queue.empty() && count < queries( q )._k )
            {
                _indices( _offset( q ) + count ) = queue.top().first[0];
                _ranks( _offset( q ) + count ) = queue.top().first[1];
                queue.pop();
                ++count;
            }

        } );
    Kokkos::fence();
    indices = _indices;
    ranks = _ranks;
    offset = _offset;
}

} // end namespace DataTransferKit

#endif
