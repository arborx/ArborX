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

TEUCHOS_UNIT_TEST( LinearBVH, push_heap )
{
    // Here checking against the example of binary heap insertion from
    // https://en.wikipedia.org/wiki/Binary_heap#Insert
    DataTransferKit::Details::PriorityQueue<int> queue;
    // NOTE Shameless hack to inspect the private data of the priority queue.
    // Will break if data is reordered in the class.
    auto heap = reinterpret_cast<int const *>( &queue );

    queue.push( 11 );
    queue.push( 5 );
    queue.push( 8 );
    queue.push( 3 );
    queue.push( 4 );
    TEST_EQUALITY( queue.top(), 11 );
    TEST_EQUALITY( heap[0], 11 );
    TEST_EQUALITY( heap[1], 5 );
    TEST_EQUALITY( heap[2], 8 );
    TEST_EQUALITY( heap[3], 3 );
    TEST_EQUALITY( heap[4], 4 );

    queue.push( 15 );

    TEST_EQUALITY( queue.top(), 15 );
    TEST_EQUALITY( heap[0], 15 );
    TEST_EQUALITY( heap[1], 5 );
    TEST_EQUALITY( heap[2], 11 );
    TEST_EQUALITY( heap[3], 3 );
    TEST_EQUALITY( heap[4], 4 );
    TEST_EQUALITY( heap[5], 8 );
}

TEUCHOS_UNIT_TEST( LinearBVH, pop_heap )
{
    // See https://en.wikipedia.org/wiki/Binary_heap#Extract
    DataTransferKit::Details::PriorityQueue<int> queue;
    auto heap = reinterpret_cast<int const *>( &queue );

    queue.push( 11 );
    queue.push( 5 );
    queue.push( 8 );
    queue.push( 3 );
    queue.push( 4 );
    TEST_EQUALITY( queue.top(), 11 );
    TEST_EQUALITY( heap[0], 11 );
    TEST_EQUALITY( heap[1], 5 );
    TEST_EQUALITY( heap[2], 8 );
    TEST_EQUALITY( heap[3], 3 );
    TEST_EQUALITY( heap[4], 4 );

    queue.pop();

    TEST_EQUALITY( queue.top(), 8 );
    TEST_EQUALITY( heap[0], 8 );
    TEST_EQUALITY( heap[1], 5 );
    TEST_EQUALITY( heap[2], 4 );
    TEST_EQUALITY( heap[3], 3 );
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
    auto heap = reinterpret_cast<int const *>( &queue );

    queue.push( 3 );
    TEST_EQUALITY( heap[0], 3 );

    queue.push( 1 );
    TEST_EQUALITY( heap[0], 3 );
    TEST_EQUALITY( heap[1], 1 );

    queue.push( 4 );
    TEST_EQUALITY( heap[0], 4 );
    TEST_EQUALITY( heap[1], 1 );
    TEST_EQUALITY( heap[2], 3 );

    queue.push( 1 );
    TEST_EQUALITY( heap[0], 4 );
    TEST_EQUALITY( heap[1], 1 );
    TEST_EQUALITY( heap[2], 3 );
    TEST_EQUALITY( heap[3], 1 );

    queue.push( 5 );
    TEST_EQUALITY( heap[0], 5 );
    TEST_EQUALITY( heap[1], 4 );
    TEST_EQUALITY( heap[2], 3 );
    TEST_EQUALITY( heap[3], 1 );
    TEST_EQUALITY( heap[4], 1 );

    queue.push( 9 );
    TEST_EQUALITY( heap[0], 9 );
    TEST_EQUALITY( heap[1], 4 );
    TEST_EQUALITY( heap[2], 5 );
    TEST_EQUALITY( heap[3], 1 );
    TEST_EQUALITY( heap[4], 1 );
    TEST_EQUALITY( heap[5], 3 );

    queue.pop();
    TEST_EQUALITY( heap[0], 5 );
    TEST_EQUALITY( heap[1], 4 );
    TEST_EQUALITY( heap[2], 3 );
    TEST_EQUALITY( heap[3], 1 );
    TEST_EQUALITY( heap[4], 1 );

    queue.push( 9 );
    TEST_EQUALITY( heap[0], 9 );
    TEST_EQUALITY( heap[1], 4 );
    TEST_EQUALITY( heap[2], 5 );
    TEST_EQUALITY( heap[3], 1 );
    TEST_EQUALITY( heap[4], 1 );
    TEST_EQUALITY( heap[5], 3 );

    queue.push( 6 );
    TEST_EQUALITY( heap[0], 9 );
    TEST_EQUALITY( heap[1], 4 );
    TEST_EQUALITY( heap[2], 6 );
    TEST_EQUALITY( heap[3], 1 );
    TEST_EQUALITY( heap[4], 1 );
    TEST_EQUALITY( heap[5], 3 );
    TEST_EQUALITY( heap[6], 5 );
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
    auto heap = reinterpret_cast<int const *>( &queue );

    queue.push( 1 );
    TEST_EQUALITY( heap[0], 1 );

    queue.push( 2 );
    TEST_EQUALITY( heap[0], 1 );
    TEST_EQUALITY( heap[1], 2 );

    queue.push( 3 );
    TEST_EQUALITY( heap[0], 1 );
    TEST_EQUALITY( heap[1], 2 );
    TEST_EQUALITY( heap[2], 3 );

    queue.push( 17 );
    TEST_EQUALITY( heap[0], 1 );
    TEST_EQUALITY( heap[1], 2 );
    TEST_EQUALITY( heap[2], 3 );
    TEST_EQUALITY( heap[3], 17 );

    queue.push( 19 );
    TEST_EQUALITY( heap[0], 1 );
    TEST_EQUALITY( heap[1], 2 );
    TEST_EQUALITY( heap[2], 3 );
    TEST_EQUALITY( heap[3], 17 );
    TEST_EQUALITY( heap[4], 19 );

    queue.push( 36 );
    TEST_EQUALITY( heap[0], 1 );
    TEST_EQUALITY( heap[1], 2 );
    TEST_EQUALITY( heap[2], 3 );
    TEST_EQUALITY( heap[3], 17 );
    TEST_EQUALITY( heap[4], 19 );
    TEST_EQUALITY( heap[5], 36 );

    queue.push( 7 );
    TEST_EQUALITY( heap[0], 1 );
    TEST_EQUALITY( heap[1], 2 );
    TEST_EQUALITY( heap[2], 3 );
    TEST_EQUALITY( heap[3], 17 );
    TEST_EQUALITY( heap[4], 19 );
    TEST_EQUALITY( heap[5], 36 );
    TEST_EQUALITY( heap[6], 7 );

    queue.push( 25 );
    TEST_EQUALITY( heap[0], 1 );
    TEST_EQUALITY( heap[1], 2 );
    TEST_EQUALITY( heap[2], 3 );
    TEST_EQUALITY( heap[3], 17 );
    TEST_EQUALITY( heap[4], 19 );
    TEST_EQUALITY( heap[5], 36 );
    TEST_EQUALITY( heap[6], 7 );
    TEST_EQUALITY( heap[7], 25 );

    queue.push( 100 );
    TEST_EQUALITY( heap[0], 1 );
    TEST_EQUALITY( heap[1], 2 );
    TEST_EQUALITY( heap[2], 3 );
    TEST_EQUALITY( heap[3], 17 );
    TEST_EQUALITY( heap[4], 19 );
    TEST_EQUALITY( heap[5], 36 );
    TEST_EQUALITY( heap[6], 7 );
    TEST_EQUALITY( heap[7], 25 );
    TEST_EQUALITY( heap[8], 100 );

    queue.pop();
    TEST_EQUALITY( heap[0], 2 );
    TEST_EQUALITY( heap[1], 17 );
    TEST_EQUALITY( heap[2], 3 );
    TEST_EQUALITY( heap[3], 25 );
    TEST_EQUALITY( heap[4], 19 );
    TEST_EQUALITY( heap[5], 36 );
    TEST_EQUALITY( heap[6], 7 );
    TEST_EQUALITY( heap[7], 100 );

    queue.pop();
    TEST_EQUALITY( heap[0], 3 );
    TEST_EQUALITY( heap[1], 17 );
    TEST_EQUALITY( heap[2], 7 );
    TEST_EQUALITY( heap[3], 25 );
    TEST_EQUALITY( heap[4], 19 );
    TEST_EQUALITY( heap[5], 36 );
    TEST_EQUALITY( heap[6], 100 );

    queue.pop();
    TEST_EQUALITY( heap[0], 7 );
    TEST_EQUALITY( heap[1], 17 );
    TEST_EQUALITY( heap[2], 36 );
    TEST_EQUALITY( heap[3], 25 );
    TEST_EQUALITY( heap[4], 19 );
    TEST_EQUALITY( heap[5], 100 );

    queue.pop();
    TEST_EQUALITY( heap[0], 17 );
    TEST_EQUALITY( heap[1], 19 );
    TEST_EQUALITY( heap[2], 36 );
    TEST_EQUALITY( heap[3], 25 );
    TEST_EQUALITY( heap[4], 100 );

    queue.pop();
    TEST_EQUALITY( heap[0], 19 );
    TEST_EQUALITY( heap[1], 25 );
    TEST_EQUALITY( heap[2], 36 );
    TEST_EQUALITY( heap[3], 100 );

    queue.pop();
    TEST_EQUALITY( heap[0], 25 );
    TEST_EQUALITY( heap[1], 100 );
    TEST_EQUALITY( heap[2], 36 );

    queue.pop();
    TEST_EQUALITY( heap[0], 36 );
    TEST_EQUALITY( heap[1], 100 );

    queue.pop();
    TEST_EQUALITY( heap[0], 100 );
}
