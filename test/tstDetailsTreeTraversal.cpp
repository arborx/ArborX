/****************************************************************************
 * Copyright (c) 2012-2018 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
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
