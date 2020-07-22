/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_DetailsPriorityQueue.hpp>
#include <ArborX_DetailsStack.hpp>

#include <boost/test/unit_test.hpp>

using namespace ArborX::Details;

BOOST_AUTO_TEST_SUITE(ContainerAdaptors)

BOOST_AUTO_TEST_CASE(test_stack)
{
  // stack is empty at construction
  Stack<int> stack;
  BOOST_TEST(stack.empty());
  BOOST_TEST(stack.size() == 0);
  // insert element
  stack.push(2);
  BOOST_TEST(!stack.empty());
  BOOST_TEST(stack.size() == 1);
  BOOST_TEST(stack.top() == 2);
  // insert another element
  stack.push(5);
  BOOST_TEST(!stack.empty());
  BOOST_TEST(stack.size() == 2);
  BOOST_TEST(stack.top() == 5);
  // remove it
  stack.pop();
  BOOST_TEST(!stack.empty());
  BOOST_TEST(stack.size() == 1);
  BOOST_TEST(stack.top() == 2);
  // empty the stack
  stack.pop();
  BOOST_TEST(stack.empty());
  BOOST_TEST(stack.size() == 0);
}

BOOST_AUTO_TEST_CASE(priority_queue)
{
  PriorityQueue<int> queue;
  // queue is empty at construction
  BOOST_TEST(queue.empty());
  // insert element
  queue.push(33);
  BOOST_TEST(!queue.empty());
  BOOST_TEST(queue.top() == 33);
  // smaller distance stays on top of the priority queue
  queue.push(24);
  BOOST_TEST(queue.top() == 33);
  // remove highest priority element
  queue.pop();
  BOOST_TEST(queue.top() == 24);
  // insert element with higher priority and check it shows up on top
  queue.push(33);
  BOOST_TEST(queue.top() == 33);
  // empty the queue
  queue.pop();
  queue.pop();
  BOOST_TEST(queue.empty());
}

BOOST_AUTO_TEST_SUITE_END()
