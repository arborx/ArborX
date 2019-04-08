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

#include <ArborX_DetailsContainers.hpp>
#include <ArborX_DetailsPriorityQueue.hpp>
#include <ArborX_DetailsStack.hpp>

// FIXME Some versions of kokkos have this header missing before the definition
// of Kokkos::Array.
#include <impl/Kokkos_Error.hpp>

#include <Kokkos_Array.hpp>

#include <random>

#include "ArborX_EnableViewComparison.hpp"

#include <boost/test/unit_test.hpp>

namespace dtk = DataTransferKit::Details;

namespace tt = boost::test_tools;

#define BOOST_TEST_MODULE Containers

BOOST_AUTO_TEST_CASE( dynamic_array_with_fixed_maximum_size )
{
    dtk::StaticVector<int, 4> a;

    BOOST_TEST( a.empty() );
    BOOST_TEST( a.size() == 0 );
    BOOST_TEST( a.maxSize() == 4 );
    BOOST_TEST( a.capacity() == 4 );

    a.pushBack( 255 );
    BOOST_TEST( !a.empty() );
    BOOST_TEST( a.size() == 1 );
    BOOST_TEST( a.maxSize() == 4 );
    BOOST_TEST( a.capacity() == 4 );
    BOOST_TEST( a.front() == 255 );
    BOOST_TEST( a[0] == 255 );
    BOOST_TEST( a.back() == 255 );

    a.pushBack( -1 );
    BOOST_TEST( !a.empty() );
    BOOST_TEST( a.size() == 2 );
    BOOST_TEST( a.maxSize() == 4 );
    BOOST_TEST( a.capacity() == 4 );
    BOOST_TEST( a.front() == 255 );
    BOOST_TEST( a[0] == 255 );
    BOOST_TEST( a.back() == -1 );
    BOOST_TEST( a[1] == -1 );

    a.pushBack( 33 );
    BOOST_TEST( !a.empty() );
    BOOST_TEST( a.size() == 3 );
    BOOST_TEST( a.maxSize() == 4 );
    BOOST_TEST( a.capacity() == 4 );
    BOOST_TEST( a.front() == 255 );
    BOOST_TEST( a[0] == 255 );
    BOOST_TEST( a[1] == -1 );
    BOOST_TEST( a.back() == 33 );
    BOOST_TEST( a[2] == 33 );

    a.popBack();
    BOOST_TEST( !a.empty() );
    BOOST_TEST( a.size() == 2 );
    BOOST_TEST( a.maxSize() == 4 );
    BOOST_TEST( a.capacity() == 4 );
    BOOST_TEST( a.front() == 255 );
    BOOST_TEST( a.back() == -1 );

    a.popBack();
    BOOST_TEST( !a.empty() );
    BOOST_TEST( a.size() == 1 );
    BOOST_TEST( a.maxSize() == 4 );
    BOOST_TEST( a.capacity() == 4 );
    BOOST_TEST( a.front() == 255 );
    BOOST_TEST( a.back() == 255 );

    a.popBack();
    BOOST_TEST( a.empty() );
    BOOST_TEST( a.size() == 0 );
    BOOST_TEST( a.maxSize() == 4 );
    BOOST_TEST( a.capacity() == 4 );
}

BOOST_AUTO_TEST_CASE( non_owning_view_over_dynamic_array )
{
    float data[6] = {255, 255, 255, 255, 255, 255};
    //                        ^^^^ ^^^^ ^^^^
    dtk::UnmanagedStaticVector<float> a( data + 2, 3 );

    BOOST_TEST( !std::is_default_constructible<decltype( a )>::value );
    BOOST_TEST( a.data() == data + 2 );

    BOOST_TEST( a.empty() );
    BOOST_TEST( a.size() == 0 );
    BOOST_TEST( a.maxSize() == 3 );
    BOOST_TEST( a.capacity() == 3 );

    a.pushBack( 0 );
    BOOST_TEST( !a.empty() );
    BOOST_TEST( a.size() == 1 );
    BOOST_TEST( a.maxSize() == 3 );
    BOOST_TEST( a.capacity() == 3 );
    BOOST_TEST( a.front() == 0 );
    BOOST_TEST( a.back() == 0 );
    BOOST_TEST( a[0] == 0 );

    a.pushBack( 1 );
    BOOST_TEST( !a.empty() );
    BOOST_TEST( a.size() == 2 );
    BOOST_TEST( a.maxSize() == 3 );
    BOOST_TEST( a.capacity() == 3 );
    BOOST_TEST( a.front() == 0 );
    BOOST_TEST( a[0] == 0 );
    BOOST_TEST( a.back() == 1 );
    BOOST_TEST( a[1] == 1 );

    a.pushBack( 2 );
    BOOST_TEST( !a.empty() );
    BOOST_TEST( a.size() == 3 );
    BOOST_TEST( a.maxSize() == 3 );
    BOOST_TEST( a.capacity() == 3 );
    BOOST_TEST( a.front() == 0 );
    BOOST_TEST( a[0] == 0 );
    BOOST_TEST( a[1] == 1 );
    BOOST_TEST( a.back() == 2 );
    BOOST_TEST( a[2] == 2 );

    a.popBack();
    BOOST_TEST( !a.empty() );
    BOOST_TEST( a.size() == 2 );
    BOOST_TEST( a.maxSize() == 3 );
    BOOST_TEST( a.capacity() == 3 );
    BOOST_TEST( a.front() == 0 );
    BOOST_TEST( a[0] == 0 );
    BOOST_TEST( a.back() == 1 );
    BOOST_TEST( a[1] == 1 );

    BOOST_TEST( data[0] == 255 );
    BOOST_TEST( data[1] == 255 );
    BOOST_TEST( data[2] == 0 );
    BOOST_TEST( data[3] == 1 );
    BOOST_TEST( data[4] == 2 );
    BOOST_TEST( data[5] == 255 );
}

#define BOOST_TEST_MODULE ContainerAdaptors

BOOST_AUTO_TEST_CASE( stack )
{
    // stack is empty at construction
    dtk::Stack<int> stack;
    BOOST_TEST( stack.empty() );
    BOOST_TEST( stack.size() == 0 );
    // insert element
    stack.push( 2 );
    BOOST_TEST( !stack.empty() );
    BOOST_TEST( stack.size() == 1 );
    BOOST_TEST( stack.top() == 2 );
    // insert another element
    stack.push( 5 );
    BOOST_TEST( !stack.empty() );
    BOOST_TEST( stack.size() == 2 );
    BOOST_TEST( stack.top() == 5 );
    // remove it
    stack.pop();
    BOOST_TEST( !stack.empty() );
    BOOST_TEST( stack.size() == 1 );
    BOOST_TEST( stack.top() == 2 );
    // empty the stack
    stack.pop();
    BOOST_TEST( stack.empty() );
    BOOST_TEST( stack.size() == 0 );
}

BOOST_AUTO_TEST_CASE( priority_queue )
{
    dtk::PriorityQueue<int> queue;
    // queue is empty at construction
    BOOST_TEST( queue.empty() );
    // insert element
    queue.push( 33 );
    BOOST_TEST( !queue.empty() );
    BOOST_TEST( queue.top() == 33 );
    // smaller distance stays on top of the priority queue
    queue.push( 24 );
    BOOST_TEST( queue.top() == 33 );
    // remove highest priority element
    queue.pop();
    BOOST_TEST( queue.top() == 24 );
    // insert element with higher priority and check it shows up on top
    queue.push( 33 );
    BOOST_TEST( queue.top() == 33 );
    // empty the queue
    queue.pop();
    queue.pop();
    BOOST_TEST( queue.empty() );
}

template <typename PriorityQueue>
void check_heap(
    PriorityQueue const &queue,
    std::vector<typename PriorityQueue::value_type> const &heap_ref )
{
    auto const size = queue.size();
    BOOST_TEST( size == static_cast<decltype( size )>( heap_ref.size() ) );

    // NOTE Shameless hack to inspect the private data of the priority queue.
    // Will break if data is reordered in the PriorityQueue class declaration.
    auto heap =
        reinterpret_cast<typename PriorityQueue::value_type const *>( &queue );
    for ( typename PriorityQueue::size_type i = 0; i < size; ++i )
        BOOST_TEST( heap[i] == heap_ref[i] );
}

#define BOOST_TEST_MODULE HeapOperations

BOOST_AUTO_TEST_CASE( push_heap )
{
    // Here checking against the example of binary heap insertion from
    // https://en.wikipedia.org/wiki/Binary_heap#Insert
    Kokkos::Array<int, 6> a = {11, 5, 8, 3, 4, 15};
    BOOST_TEST( dtk::isHeap( a.data(), a.data() + 5, dtk::Less<int>() ) );
    BOOST_TEST( !dtk::isHeap( a.data(), a.data() + 6, dtk::Less<int>() ) );
    dtk::pushHeap( a.data(), a.data() + 6, dtk::Less<int>() );
    BOOST_TEST( dtk::isHeap( a.data(), a.data() + 6, dtk::Less<int>() ) );
    Kokkos::Array<int, 6> ref = {15, 5, 11, 3, 4, 8};
    BOOST_TEST( a == ref, tt::per_element() );
}

BOOST_AUTO_TEST_CASE( pop_heap )
{
    // See https://en.wikipedia.org/wiki/Binary_heap#Extract
    Kokkos::Array<int, 5> a = {11, 5, 8, 3, 4};
    BOOST_TEST( dtk::isHeap( a.data(), a.data() + 5, dtk::Less<int>() ) );
    dtk::popHeap( a.data(), a.data() + 5, dtk::Less<int>() );
    BOOST_TEST( dtk::isHeap( a.data(), a.data() + 4, dtk::Less<int>() ) );
    BOOST_TEST( !dtk::isHeap( a.data(), a.data() + 5, dtk::Less<int>() ) );
    Kokkos::Array<int, 5> ref = {8, 5, 4, 3, 11};
    BOOST_TEST( a == ref, tt::per_element() );
}

BOOST_AUTO_TEST_CASE( max_heap )
{
    // Here attempting to reproduce examples code from cppreference.com for
    // std::push_heap() and std::pop_heap().  It turns out that calling
    // std::make_heap() on { 3, 1, 4, 1, 5, 9 } is not equivalent to inserting
    // one element at a time (I also tried to insert them in reverse order out
    // of curiousity) and yields 9 5 4 1 1 3 :/
    // cpluplus.com does make a note that the order of the elements will depend
    // on implementation.
    // Nevertheless, I thought it helps understanding how the algorithm works so
    // I decided to keep it as a unit test.
    Kokkos::Array<int, 7> a = {3, 1, 4, 1, 5, 9, 6};
    Kokkos::Array<int, 7> ref;

    dtk::pushHeap( a.data(), a.data() + 0, dtk::Less<int>() );
    ref = {3, 1, 4, 1, 5, 9, 6};
    BOOST_TEST( a == ref, tt::per_element() );

    dtk::pushHeap( a.data(), a.data() + 1, dtk::Less<int>() );
    ref = {3, 1, 4, 1, 5, 9, 6};
    BOOST_TEST( a == ref, tt::per_element() );

    dtk::pushHeap( a.data(), a.data() + 2, dtk::Less<int>() );
    ref = {3, 1, 4, 1, 5, 9, 6};
    BOOST_TEST( a == ref, tt::per_element() );

    dtk::pushHeap( a.data(), a.data() + 3, dtk::Less<int>() );
    ref = {4, 1, 3, 1, 5, 9, 6};
    BOOST_TEST( a == ref, tt::per_element() );

    dtk::pushHeap( a.data(), a.data() + 4, dtk::Less<int>() );
    ref = {4, 1, 3, 1, 5, 9, 6};
    BOOST_TEST( a == ref, tt::per_element() );

    dtk::pushHeap( a.data(), a.data() + 5, dtk::Less<int>() );
    ref = {5, 4, 3, 1, 1, 9, 6};
    BOOST_TEST( a == ref, tt::per_element() );

    dtk::pushHeap( a.data(), a.data() + 6, dtk::Less<int>() );
    ref = {9, 4, 5, 1, 1, 3, 6};
    BOOST_TEST( a == ref, tt::per_element() );

    dtk::popHeap( a.data(), a.data() + 6, dtk::Less<int>() );
    ref = {5, 4, 3, 1, 1, 9, 6};
    BOOST_TEST( a == ref, tt::per_element() );

    dtk::pushHeap( a.data(), a.data() + 6, dtk::Less<int>() );
    ref = {9, 4, 5, 1, 1, 3, 6};
    BOOST_TEST( a == ref, tt::per_element() );

    dtk::pushHeap( a.data(), a.data() + 7, dtk::Less<int>() );
    ref = {9, 4, 6, 1, 1, 3, 5};
    BOOST_TEST( a == ref, tt::per_element() );
}

BOOST_AUTO_TEST_CASE( min_heap )
{
    // Here reproducing the example of a binary min heap from Wikipedia and then
    // popping all elements one by one.
    // This helped resolving an issue with ProprityQueue::pop() that was not
    // exposed by other unit tests in this file, partly because they were too
    // trivial.  The bug was only showing with real kNN search problems when
    // comparing results from BoundaryVolumeHierarchy with boost::rtree.
    Kokkos::Array<int, 9> a = {1, 2, 3, 17, 19, 36, 7, 25, 100};
    Kokkos::Array<int, 9> ref;
    for ( int i = 0; i < 9; ++i )
    {
        dtk::pushHeap( a.data(), a.data() + i, dtk::Greater<int>() );
        ref = {1, 2, 3, 17, 19, 36, 7, 25, 100};
        BOOST_TEST( a == ref, tt::per_element() );
    }

    dtk::popHeap( a.data(), a.data() + 9, dtk::Greater<int>() );
    ref = {2, 17, 3, 25, 19, 36, 7, 100, 1};
    BOOST_TEST( a == ref, tt::per_element() );

    dtk::popHeap( a.data(), a.data() + 8, dtk::Greater<int>() );
    ref = {3, 17, 7, 25, 19, 36, 100, 2, 1};
    BOOST_TEST( a == ref, tt::per_element() );

    dtk::popHeap( a.data(), a.data() + 7, dtk::Greater<int>() );
    ref = {7, 17, 36, 25, 19, 100, 3, 2, 1};
    BOOST_TEST( a == ref, tt::per_element() );

    dtk::popHeap( a.data(), a.data() + 6, dtk::Greater<int>() );
    ref = {17, 19, 36, 25, 100, 7, 3, 2, 1};
    BOOST_TEST( a == ref, tt::per_element() );

    dtk::popHeap( a.data(), a.data() + 5, dtk::Greater<int>() );
    ref = {19, 25, 36, 100, 17, 7, 3, 2, 1};
    BOOST_TEST( a == ref, tt::per_element() );

    dtk::popHeap( a.data(), a.data() + 4, dtk::Greater<int>() );
    ref = {25, 100, 36, 19, 17, 7, 3, 2, 1};
    BOOST_TEST( a == ref, tt::per_element() );

    dtk::popHeap( a.data(), a.data() + 3, dtk::Greater<int>() );
    ref = {36, 100, 25, 19, 17, 7, 3, 2, 1};
    BOOST_TEST( a == ref, tt::per_element() );

    dtk::popHeap( a.data(), a.data() + 2, dtk::Greater<int>() );
    ref = {100, 36, 25, 19, 17, 7, 3, 2, 1};
    BOOST_TEST( a == ref, tt::per_element() );

    dtk::popHeap( a.data(), a.data() + 1, dtk::Greater<int>() );
    ref = {100, 36, 25, 19, 17, 7, 3, 2, 1};
    BOOST_TEST( a == ref, tt::per_element() );

    dtk::popHeap( a.data(), a.data() + 0, dtk::Greater<int>() );
    ref = {100, 36, 25, 19, 17, 7, 3, 2, 1};
    BOOST_TEST( a == ref, tt::per_element() );
}

BOOST_AUTO_TEST_CASE( sort_heap )
{
    for ( auto heap : {std::vector<int>{36, 19, 25, 17, 3, 7, 1, 2, 9},
                       std::vector<int>{36, 19, 25, 17, 3, 9, 1, 2, 7},
                       std::vector<int>{100, 19, 36, 17, 3, 25, 1, 2, 7},
                       std::vector<int>{15, 5, 11, 3, 4, 8}} )
    {
        dtk::sortHeap( heap.data(), heap.data() + heap.size(),
                       dtk::Less<int>() );
        // std::sort_heap( heap.begin(), heap.end() );
        BOOST_TEST( std::is_sorted( heap.begin(), heap.end() ) );
    }
}

BOOST_AUTO_TEST_CASE( is_heap )
{
    for ( auto heap : {std::vector<int>{36, 19, 25, 17, 3, 7, 1, 2, 9},
                       std::vector<int>{36, 19, 25, 17, 3, 9, 1, 2, 7},
                       std::vector<int>{100, 19, 36, 17, 3, 25, 1, 2, 7},
                       std::vector<int>{15, 5, 11, 3, 4, 8}} )
    {
        BOOST_TEST( dtk::isHeap( heap.data(), heap.data() + heap.size(),
                                 dtk::Less<int>() ) );
    }
    for ( auto not_heap : {std::vector<int>{0, 1, 2, 3, 4, 3, 2, 1, 0},
                           std::vector<int>{2, 1, 0, 1, 2}} )
    {
        BOOST_TEST( !dtk::isHeap( not_heap.data(),
                                  not_heap.data() + not_heap.size(),
                                  dtk::Less<int>() ) );
    }
}

#define BOOST_TEST_MODULE PriorityQueue

BOOST_AUTO_TEST_CASE( pop_push )
{
    // note that calling pop_push(x) does not necessarily yield the same heap
    // than calling consecutively pop() and push(x)
    // below is a max heap example to illustate this interesting property
    dtk::PriorityQueue<int> queue;

    std::vector<int> ref = {100, 19, 36, 17, 3, 25, 1, 2, 7};
    for ( auto x : ref )
        queue.push( x );
    check_heap( queue, ref );

    queue.pop();
    check_heap( queue, {36, 19, 25, 17, 3, 7, 1, 2} );

    queue.push( 9 );
    check_heap( queue, {36, 19, 25, 17, 3, 7, 1, 2, 9} );
    //                                    ^^       ^^

    // Clear the content of the queue
    queue = dtk::PriorityQueue<int>();
    for ( auto x : ref )
        queue.push( x );
    check_heap( queue, ref );

    queue.popPush( 9 );
    check_heap( queue, {36, 19, 25, 17, 3, 9, 1, 2, 7} );
    //                                    ^^       ^^
}

template <typename PriorityQueue>
void check_heap( PriorityQueue const &queue )
{
    using ValueType = typename PriorityQueue::value_type;
    int const size = queue.size();
    auto const compare = queue.valueComp();
    auto heap = reinterpret_cast<ValueType const *>( &queue );
    for ( int i = 0; i < size; ++i )
    {
        int parent = ( i - 1 ) / 2;
        if ( i > 0 )
            BOOST_TEST( !compare( heap[parent], heap[i] ) );
        for ( int child : {2 * i + 1, 2 * i + 2} )
            if ( child < size )
                BOOST_TEST( !compare( heap[i], heap[child] ) );
    }
}

BOOST_AUTO_TEST_CASE( maintain_heap_properties )
{
    DataTransferKit::Details::PriorityQueue<int> queue;

    std::default_random_engine generator;
    std::uniform_int_distribution<int> uniform_distribution( 0, 100 );

    enum OperationType : int { POP, PUSH, POP_PUSH };
    std::discrete_distribution<int> discrete_distribution(
        {POP, PUSH, POP_PUSH} );

    // initially insert a number of elements in the queue so that it is
    // unlikely that an error will be raised (this would happen if pop() is
    // called on an empty queue or push(x) on a queue that is already at maximum
    // capacity)
    for ( int i = 0; i < 64; ++i )
    {
        queue.push( uniform_distribution( generator ) );
        check_heap( queue );
    }

    // choose randomly whether to call pop(), push(x), or pop_push(x) and check
    // that the heap properties are maintained
    for ( int i = 0; i < 512; ++i )
    {
        switch ( discrete_distribution( generator ) )
        {
        case POP:
            queue.pop();
            break;
        case PUSH:
            queue.push( uniform_distribution( generator ) );
            break;
        case POP_PUSH:
            queue.popPush( uniform_distribution( generator ) );
            break;
        default:
            throw std::runtime_error( "something went wrong" );
        }
        check_heap( queue );
    }

    // remove all elements from the queue one by one
    while ( !queue.empty() )
    {
        queue.pop();
        check_heap( queue );
    }
}
