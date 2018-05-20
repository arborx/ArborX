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
#include <DTK_DetailsPriorityQueue.hpp>
#include <DTK_DetailsStack.hpp>

#include <Teuchos_UnitTestHarness.hpp>

TEUCHOS_UNIT_TEST( LinearBVH, stack )
{
    // stack is empty at construction
    DataTransferKit::Details::Stack<int> stack;
    TEST_ASSERT( stack.empty() );
    // insert element
    stack.push( 2 );
    TEST_ASSERT( !stack.empty() );
    TEST_ASSERT( stack.top() == 2 );
    // insert another element
    stack.push( 5 );
    TEST_ASSERT( !stack.empty() );
    TEST_ASSERT( stack.top() == 5 );
    // remove it
    stack.pop();
    TEST_ASSERT( !stack.empty() );
    TEST_ASSERT( stack.top() == 2 );
    // empty the stack
    stack.pop();
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
    TEST_EQUALITY( size, heap_ref.size() );

    // NOTE Shameless hack to inspect the private data of the priority queue.
    // Will break if data is reordered in the PriorityQueue class declaration.
    auto heap =
        reinterpret_cast<typename PriorityQueue::ValueType const *>( &queue );
    for ( typename PriorityQueue::SizeType i = 0; i < size; ++i )
        TEST_EQUALITY( heap[i], heap_ref[i] );
}

TEUCHOS_UNIT_TEST( LinearBVH, push_heap )
{
    // Here checking against the example of binary heap insertion from
    // https://en.wikipedia.org/wiki/Binary_heap#Insert
    DataTransferKit::Details::PriorityQueue<int> queue;

    queue.push( 11 );
    queue.push( 5 );
    queue.push( 8 );
    queue.push( 3 );
    queue.push( 4 );

    TEST_EQUALITY( queue.top(), 11 );
    check_heap( queue, {11, 5, 8, 3, 4}, success, out );

    queue.push( 15 );

    TEST_EQUALITY( queue.top(), 15 );
    check_heap( queue, {15, 5, 11, 3, 4, 8}, success, out );
}

TEUCHOS_UNIT_TEST( LinearBVH, pop_heap )
{
    // See https://en.wikipedia.org/wiki/Binary_heap#Extract
    DataTransferKit::Details::PriorityQueue<int> queue;

    queue.push( 11 );
    queue.push( 5 );
    queue.push( 8 );
    queue.push( 3 );
    queue.push( 4 );

    TEST_EQUALITY( queue.top(), 11 );
    check_heap( queue, {11, 5, 8, 3, 4}, success, out );

    queue.pop();

    TEST_EQUALITY( queue.top(), 8 );
    check_heap( queue, {8, 5, 4, 3}, success, out );
}

TEUCHOS_UNIT_TEST( LinearBVH, heap )
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
    DataTransferKit::Details::PriorityQueue<int> queue;

    queue.push( 3 );
    check_heap( queue, {3}, success, out );

    queue.push( 1 );
    check_heap( queue, {3, 1}, success, out );

    queue.push( 4 );
    check_heap( queue, {4, 1, 3}, success, out );

    queue.push( 1 );
    check_heap( queue, {4, 1, 3, 1}, success, out );

    queue.push( 5 );
    check_heap( queue, {5, 4, 3, 1, 1}, success, out );

    queue.push( 9 );
    check_heap( queue, {9, 4, 5, 1, 1, 3}, success, out );

    queue.pop();
    check_heap( queue, {5, 4, 3, 1, 1}, success, out );

    queue.push( 9 );
    check_heap( queue, {9, 4, 5, 1, 1, 3}, success, out );

    queue.push( 6 );
    check_heap( queue, {9, 4, 6, 1, 1, 3, 5}, success, out );
}

TEUCHOS_UNIT_TEST( LinearBVH, min_heap )
{
    // Here reproducing the example of a binary min heap from Wikipedia and then
    // popping all elements one by one.
    // This helped resolving an issue with ProprityQueue::pop() that was not
    // exposed by other unit tests in this file, partly because they were too
    // trivial.  The bug was only showing with real kNN search problems when
    // comparing results from BoundaryVolumeHierarchy with boost::rtree.
    struct Greater
    {
      public:
        KOKKOS_FUNCTION bool operator()( int x, int y ) const { return x > y; }
    };
    DataTransferKit::Details::PriorityQueue<int, Greater> queue;

    queue.push( 1 );
    check_heap( queue, {1}, success, out );

    queue.push( 2 );
    check_heap( queue, {1, 2}, success, out );

    queue.push( 3 );
    check_heap( queue, {1, 2, 3}, success, out );

    queue.push( 17 );
    check_heap( queue, {1, 2, 3, 17}, success, out );

    queue.push( 19 );
    check_heap( queue, {1, 2, 3, 17, 19}, success, out );

    queue.push( 36 );
    check_heap( queue, {1, 2, 3, 17, 19, 36}, success, out );

    queue.push( 7 );
    check_heap( queue, {1, 2, 3, 17, 19, 36, 7}, success, out );

    queue.push( 25 );
    check_heap( queue, {1, 2, 3, 17, 19, 36, 7, 25}, success, out );

    queue.push( 100 );
    check_heap( queue, {1, 2, 3, 17, 19, 36, 7, 25, 100}, success, out );

    queue.pop();
    check_heap( queue, {2, 17, 3, 25, 19, 36, 7, 100}, success, out );

    queue.pop();
    check_heap( queue, {3, 17, 7, 25, 19, 36, 100}, success, out );

    queue.pop();
    check_heap( queue, {7, 17, 36, 25, 19, 100}, success, out );

    queue.pop();
    check_heap( queue, {17, 19, 36, 25, 100}, success, out );

    queue.pop();
    check_heap( queue, {19, 25, 36, 100}, success, out );

    queue.pop();
    check_heap( queue, {25, 100, 36}, success, out );

    queue.pop();
    check_heap( queue, {36, 100}, success, out );

    queue.pop();
    check_heap( queue, {100}, success, out );
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

    queue.clear();
    for ( auto x : ref )
        queue.push( x );
    check_heap( queue, ref, success, out );

    queue.pop_push( 9 );
    check_heap( queue, {36, 19, 25, 17, 3, 9, 1, 2, 7}, success, out );
}
