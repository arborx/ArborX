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

#include <DTK_DetailsContainers.hpp>
#include <DTK_DetailsPriorityQueue.hpp>
#include <DTK_DetailsStack.hpp>

#include <Teuchos_UnitTestHarness.hpp>

#include <Kokkos_Array.hpp>

#include <random>

namespace dtk = DataTransferKit::Details;

TEUCHOS_UNIT_TEST( Containers, dynamic_array_with_fixed_maximum_size )
{
    dtk::Vector<int, 4> a;

    TEST_ASSERT( a.empty() );
    TEST_EQUALITY( a.size(), 0 );
    TEST_EQUALITY( a.maxSize(), 4 );
    TEST_EQUALITY( a.capacity(), 4 );

    a.pushBack( 255 );
    TEST_ASSERT( !a.empty() );
    TEST_EQUALITY( a.size(), 1 );
    TEST_EQUALITY( a.maxSize(), 4 );
    TEST_EQUALITY( a.capacity(), 4 );
    TEST_EQUALITY( a.front(), 255 );
    TEST_EQUALITY( a[0], 255 );
    TEST_EQUALITY( a.back(), 255 );

    a.pushBack( -1 );
    TEST_ASSERT( !a.empty() );
    TEST_EQUALITY( a.size(), 2 );
    TEST_EQUALITY( a.maxSize(), 4 );
    TEST_EQUALITY( a.capacity(), 4 );
    TEST_EQUALITY( a.front(), 255 );
    TEST_EQUALITY( a[0], 255 );
    TEST_EQUALITY( a.back(), -1 );
    TEST_EQUALITY( a[1], -1 );

    a.pushBack( 33 );
    TEST_ASSERT( !a.empty() );
    TEST_EQUALITY( a.size(), 3 );
    TEST_EQUALITY( a.maxSize(), 4 );
    TEST_EQUALITY( a.capacity(), 4 );
    TEST_EQUALITY( a.front(), 255 );
    TEST_EQUALITY( a[0], 255 );
    TEST_EQUALITY( a[1], -1 );
    TEST_EQUALITY( a.back(), 33 );
    TEST_EQUALITY( a[2], 33 );

    a.popBack();
    TEST_ASSERT( !a.empty() );
    TEST_EQUALITY( a.size(), 2 );
    TEST_EQUALITY( a.maxSize(), 4 );
    TEST_EQUALITY( a.capacity(), 4 );
    TEST_EQUALITY( a.front(), 255 );
    TEST_EQUALITY( a.back(), -1 );

    a.popBack();
    TEST_ASSERT( !a.empty() );
    TEST_EQUALITY( a.size(), 1 );
    TEST_EQUALITY( a.maxSize(), 4 );
    TEST_EQUALITY( a.capacity(), 4 );
    TEST_EQUALITY( a.front(), 255 );
    TEST_EQUALITY( a.back(), 255 );

    a.popBack();
    TEST_ASSERT( a.empty() );
    TEST_EQUALITY( a.size(), 0 );
    TEST_EQUALITY( a.maxSize(), 4 );
    TEST_EQUALITY( a.capacity(), 4 );
}

TEUCHOS_UNIT_TEST( Containers, non_owning_view_over_dynamic_array )
{
    float data[6] = {255, 255, 255, 255, 255, 255};
    //                        ^^^^ ^^^^ ^^^^
    dtk::UnmanagedVector<float> a( data + 2, 3 );

    TEST_ASSERT( !std::is_default_constructible<decltype( a )>::value );
    TEST_EQUALITY( a.data(), data + 2 );

    TEST_ASSERT( a.empty() );
    TEST_EQUALITY( a.size(), 0 );
    TEST_EQUALITY( a.maxSize(), 3 );
    TEST_EQUALITY( a.capacity(), 3 );

    a.pushBack( 0 );
    TEST_ASSERT( !a.empty() );
    TEST_EQUALITY( a.size(), 1 );
    TEST_EQUALITY( a.maxSize(), 3 );
    TEST_EQUALITY( a.capacity(), 3 );
    TEST_EQUALITY( a.front(), 0 );
    TEST_EQUALITY( a.back(), 0 );
    TEST_EQUALITY( a[0], 0 );

    a.pushBack( 1 );
    TEST_ASSERT( !a.empty() );
    TEST_EQUALITY( a.size(), 2 );
    TEST_EQUALITY( a.maxSize(), 3 );
    TEST_EQUALITY( a.capacity(), 3 );
    TEST_EQUALITY( a.front(), 0 );
    TEST_EQUALITY( a[0], 0 );
    TEST_EQUALITY( a.back(), 1 );
    TEST_EQUALITY( a[1], 1 );

    a.pushBack( 2 );
    TEST_ASSERT( !a.empty() );
    TEST_EQUALITY( a.size(), 3 );
    TEST_EQUALITY( a.maxSize(), 3 );
    TEST_EQUALITY( a.capacity(), 3 );
    TEST_EQUALITY( a.front(), 0 );
    TEST_EQUALITY( a[0], 0 );
    TEST_EQUALITY( a[1], 1 );
    TEST_EQUALITY( a.back(), 2 );
    TEST_EQUALITY( a[2], 2 );

    a.popBack();
    TEST_ASSERT( !a.empty() );
    TEST_EQUALITY( a.size(), 2 );
    TEST_EQUALITY( a.maxSize(), 3 );
    TEST_EQUALITY( a.capacity(), 3 );
    TEST_EQUALITY( a.front(), 0 );
    TEST_EQUALITY( a[0], 0 );
    TEST_EQUALITY( a.back(), 1 );
    TEST_EQUALITY( a[1], 1 );

    TEST_EQUALITY( data[0], 255 );
    TEST_EQUALITY( data[1], 255 );
    TEST_EQUALITY( data[2], 0 );
    TEST_EQUALITY( data[3], 1 );
    TEST_EQUALITY( data[4], 2 );
    TEST_EQUALITY( data[5], 255 );
}

TEUCHOS_UNIT_TEST( LinearBVH, stack )
{
    // stack is empty at construction
    DataTransferKit::Details::Stack<int> stack;
    TEST_ASSERT( stack.empty() );
    TEST_ASSERT( stack.size() == 0 );
    // insert element
    stack.push( 2 );
    TEST_ASSERT( !stack.empty() );
    TEST_ASSERT( stack.size() == 1 );
    TEST_ASSERT( stack.top() == 2 );
    // insert another element
    stack.push( 5 );
    TEST_ASSERT( !stack.empty() );
    TEST_ASSERT( stack.size() == 2 );
    TEST_ASSERT( stack.top() == 5 );
    // remove it
    stack.pop();
    TEST_ASSERT( !stack.empty() );
    TEST_ASSERT( stack.size() == 1 );
    TEST_ASSERT( stack.top() == 2 );
    // empty the stack
    stack.pop();
    TEST_ASSERT( stack.empty() );
    TEST_ASSERT( stack.size() == 0 );
    // add a few elements again and clear the stack
    for ( int x : {0, 1, 1, 2, 3, 5, 8, 13, 21, 34} )
        stack.push( x );
    stack.clear();
    TEST_ASSERT( stack.empty() );
}

TEUCHOS_UNIT_TEST( LinearBVH, priority_queue )
{
    DataTransferKit::Details::PriorityQueue<int> queue;
    // queue is empty at construction
    TEST_ASSERT( queue.empty() );
    // insert element
    queue.push( 33 );
    TEST_ASSERT( !queue.empty() );
    TEST_EQUALITY( queue.top(), 33 );
    // smaller distance stays on top of the priority queue
    queue.push( 24 );
    TEST_EQUALITY( queue.top(), 33 );
    // remove highest priority element
    queue.pop();
    TEST_EQUALITY( queue.top(), 24 );
    // insert element with higher priority and check it shows up on top
    queue.push( 33 );
    TEST_EQUALITY( queue.top(), 33 );
    // empty the queue
    queue.pop();
    queue.pop();
    TEST_ASSERT( queue.empty() );
}

template <typename PriorityQueue>
void check_heap( PriorityQueue const &queue,
                 std::vector<typename PriorityQueue::ValueType> const &heap_ref,
                 bool &success, Teuchos::FancyOStream &out )
{
    auto const size = queue.size();
    TEST_EQUALITY( size, static_cast<decltype( size )>( heap_ref.size() ) );

    // NOTE Shameless hack to inspect the private data of the priority queue.
    // Will break if data is reordered in the PriorityQueue class declaration.
    auto heap =
        reinterpret_cast<typename PriorityQueue::ValueType const *>( &queue );
    for ( typename PriorityQueue::IndexType i = 0; i < size; ++i )
        TEST_EQUALITY( heap[i], heap_ref[i] );
}

TEUCHOS_UNIT_TEST( HeapOperations, push_heap )
{
    // Here checking against the example of binary heap insertion from
    // https://en.wikipedia.org/wiki/Binary_heap#Insert
    Kokkos::Array<int, 6> a = {11, 5, 8, 3, 4, 15};
    TEST_ASSERT( dtk::isHeap( a.data(), a.data() + 5, dtk::Less<int>() ) );
    TEST_ASSERT( !dtk::isHeap( a.data(), a.data() + 6, dtk::Less<int>() ) );
    dtk::pushHeap( a.data(), a.data() + 6, dtk::Less<int>() );
    TEST_ASSERT( dtk::isHeap( a.data(), a.data() + 6, dtk::Less<int>() ) );
    Kokkos::Array<int, 6> ref = {15, 5, 11, 3, 4, 8};
    TEST_COMPARE_ARRAYS( a, ref );
}

TEUCHOS_UNIT_TEST( HeapOperations, pop_heap )
{
    // See https://en.wikipedia.org/wiki/Binary_heap#Extract
    Kokkos::Array<int, 5> a = {11, 5, 8, 3, 4};
    TEST_ASSERT( dtk::isHeap( a.data(), a.data() + 5, dtk::Less<int>() ) );
    dtk::popHeap( a.data(), a.data() + 5, dtk::Less<int>() );
    TEST_ASSERT( dtk::isHeap( a.data(), a.data() + 4, dtk::Less<int>() ) );
    TEST_ASSERT( !dtk::isHeap( a.data(), a.data() + 5, dtk::Less<int>() ) );
    Kokkos::Array<int, 5> ref = {8, 5, 4, 3, 11};
    TEST_COMPARE_ARRAYS( a, ref );
}

TEUCHOS_UNIT_TEST( HeapOperations, max_heap )
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
    TEST_COMPARE_ARRAYS( a, ref );

    dtk::pushHeap( a.data(), a.data() + 1, dtk::Less<int>() );
    ref = {3, 1, 4, 1, 5, 9, 6};
    TEST_COMPARE_ARRAYS( a, ref );

    dtk::pushHeap( a.data(), a.data() + 2, dtk::Less<int>() );
    ref = {3, 1, 4, 1, 5, 9, 6};
    TEST_COMPARE_ARRAYS( a, ref );

    dtk::pushHeap( a.data(), a.data() + 3, dtk::Less<int>() );
    ref = {4, 1, 3, 1, 5, 9, 6};
    TEST_COMPARE_ARRAYS( a, ref );

    dtk::pushHeap( a.data(), a.data() + 4, dtk::Less<int>() );
    ref = {4, 1, 3, 1, 5, 9, 6};
    TEST_COMPARE_ARRAYS( a, ref );

    dtk::pushHeap( a.data(), a.data() + 5, dtk::Less<int>() );
    ref = {5, 4, 3, 1, 1, 9, 6};
    TEST_COMPARE_ARRAYS( a, ref );

    dtk::pushHeap( a.data(), a.data() + 6, dtk::Less<int>() );
    ref = {9, 4, 5, 1, 1, 3, 6};
    TEST_COMPARE_ARRAYS( a, ref );

    dtk::popHeap( a.data(), a.data() + 6, dtk::Less<int>() );
    ref = {5, 4, 3, 1, 1, 9, 6};
    TEST_COMPARE_ARRAYS( a, ref );

    dtk::pushHeap( a.data(), a.data() + 6, dtk::Less<int>() );
    ref = {9, 4, 5, 1, 1, 3, 6};
    TEST_COMPARE_ARRAYS( a, ref );

    dtk::pushHeap( a.data(), a.data() + 7, dtk::Less<int>() );
    ref = {9, 4, 6, 1, 1, 3, 5};
    TEST_COMPARE_ARRAYS( a, ref );
}

TEUCHOS_UNIT_TEST( HeapOperations, min_heap )
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
        TEST_COMPARE_ARRAYS( a, ref );
    }

    dtk::popHeap( a.data(), a.data() + 9, dtk::Greater<int>() );
    ref = {2, 17, 3, 25, 19, 36, 7, 100, 1};
    TEST_COMPARE_ARRAYS( a, ref );

    dtk::popHeap( a.data(), a.data() + 8, dtk::Greater<int>() );
    ref = {3, 17, 7, 25, 19, 36, 100, 2, 1};
    TEST_COMPARE_ARRAYS( a, ref );

    dtk::popHeap( a.data(), a.data() + 7, dtk::Greater<int>() );
    ref = {7, 17, 36, 25, 19, 100, 3, 2, 1};
    TEST_COMPARE_ARRAYS( a, ref );

    dtk::popHeap( a.data(), a.data() + 6, dtk::Greater<int>() );
    ref = {17, 19, 36, 25, 100, 7, 3, 2, 1};
    TEST_COMPARE_ARRAYS( a, ref );

    dtk::popHeap( a.data(), a.data() + 5, dtk::Greater<int>() );
    ref = {19, 25, 36, 100, 17, 7, 3, 2, 1};
    TEST_COMPARE_ARRAYS( a, ref );

    dtk::popHeap( a.data(), a.data() + 4, dtk::Greater<int>() );
    ref = {25, 100, 36, 19, 17, 7, 3, 2, 1};
    TEST_COMPARE_ARRAYS( a, ref );

    dtk::popHeap( a.data(), a.data() + 3, dtk::Greater<int>() );
    ref = {36, 100, 25, 19, 17, 7, 3, 2, 1};
    TEST_COMPARE_ARRAYS( a, ref );

    dtk::popHeap( a.data(), a.data() + 2, dtk::Greater<int>() );
    ref = {100, 36, 25, 19, 17, 7, 3, 2, 1};
    TEST_COMPARE_ARRAYS( a, ref );

    dtk::popHeap( a.data(), a.data() + 1, dtk::Greater<int>() );
    ref = {100, 36, 25, 19, 17, 7, 3, 2, 1};
    TEST_COMPARE_ARRAYS( a, ref );

    dtk::popHeap( a.data(), a.data() + 0, dtk::Greater<int>() );
    ref = {100, 36, 25, 19, 17, 7, 3, 2, 1};
    TEST_COMPARE_ARRAYS( a, ref );
}

TEUCHOS_UNIT_TEST( LinearBVH, pop_push )
{
    // note that calling pop_push(x) does not necessarily yield the same heap
    // than calling consecutively pop() and push(x)
    // below is a max heap example to illustate this interesting property
    DataTransferKit::Details::PriorityQueue<int> queue;

    std::vector<int> ref = {100, 19, 36, 17, 3, 25, 1, 2, 7};
    for ( auto x : ref )
        queue.push( x );
    check_heap( queue, ref, success, out );

    queue.pop();
    check_heap( queue, {36, 19, 25, 17, 3, 7, 1, 2}, success, out );

    queue.push( 9 );
    check_heap( queue, {36, 19, 25, 17, 3, 7, 1, 2, 9}, success, out );
    //                                    ^^       ^^

    queue.clear();
    for ( auto x : ref )
        queue.push( x );
    check_heap( queue, ref, success, out );

    queue.pop_push( 9 );
    check_heap( queue, {36, 19, 25, 17, 3, 9, 1, 2, 7}, success, out );
    //                                    ^^       ^^
}

template <typename PriorityQueue>
void check_heap( PriorityQueue const &queue, bool &success,
                 Teuchos::FancyOStream &out )
{
    using ValueType = typename PriorityQueue::ValueType;
    int const size = queue.size();
    auto const compare = queue.value_comp();
    auto heap = reinterpret_cast<ValueType const *>( &queue );
    for ( int i = 0; i < size; ++i )
    {
        int parent = ( i - 1 ) / 2;
        if ( i > 0 )
            TEST_ASSERT( !compare( heap[parent], heap[i] ) )
        for ( int child : {2 * i + 1, 2 * i + 2} )
            if ( child < size )
                TEST_ASSERT( !compare( heap[i], heap[child] ) )
    }
}

TEUCHOS_UNIT_TEST( PriorityQueue, maintain_heap_properties )
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
        check_heap( queue, success, out );
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
            queue.pop_push( uniform_distribution( generator ) );
            break;
        default:
            throw std::runtime_error( "something went wrong" );
        }
        check_heap( queue, success, out );
    }

    // remove all elements from the queue one by one
    while ( !queue.empty() )
    {
        queue.pop();
        check_heap( queue, success, out );
    }
}

TEUCHOS_UNIT_TEST( HeapOperations, is_heap )
{
    for ( auto heap : {std::vector<int>{36, 19, 25, 17, 3, 7, 1, 2, 9},
                       std::vector<int>{36, 19, 25, 17, 3, 9, 1, 2, 7},
                       std::vector<int>{100, 19, 36, 17, 3, 25, 1, 2, 7},
                       std::vector<int>{15, 5, 11, 3, 4, 8}} )
    {
        TEST_ASSERT( dtk::isHeap( heap.data(), heap.data() + heap.size(),
                                  dtk::Less<int>() ) );
    }
    for ( auto not_heap : {std::vector<int>{0, 1, 2, 3, 4, 3, 2, 1, 0},
                           std::vector<int>{2, 1, 0, 1, 2}} )
    {
        TEST_ASSERT( !dtk::isHeap( not_heap.data(),
                                   not_heap.data() + not_heap.size(),
                                   dtk::Less<int>() ) );
    }
}
