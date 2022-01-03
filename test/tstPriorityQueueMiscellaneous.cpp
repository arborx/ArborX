/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_DetailsPriorityQueue.hpp>

#include <boost/test/unit_test.hpp>

#include <random>

using ArborX::Details::PriorityQueue;

namespace tt = boost::test_tools;

// NOTE The tests below check that the priority queue invariant is maintained
// while inserting and removing elements into the queue.  They rely on a hack
// (reinterpret_cast) to access the underlying container.
BOOST_AUTO_TEST_SUITE(PriorityQueueMiscellaneous)

template <typename PriorityQueue>
void check_heap(PriorityQueue const &queue,
                std::vector<typename PriorityQueue::value_type> const &heap_ref)
{
  auto const size = queue.size();
  BOOST_TEST(size == static_cast<decltype(size)>(heap_ref.size()));

  // NOTE Shameless hack to inspect the private data of the priority queue.
  // Will break if data is reordered in the PriorityQueue class declaration.
  auto heap =
      reinterpret_cast<typename PriorityQueue::value_type const *>(&queue);
  for (typename PriorityQueue::size_type i = 0; i < size; ++i)
    BOOST_TEST(heap[i] == heap_ref[i]);
}

BOOST_AUTO_TEST_CASE(pop_push)
{
  // note that calling pop_push(x) does not necessarily yield the same heap
  // than calling consecutively pop() and push(x)
  // below is a max heap example to illustrate this interesting property
  PriorityQueue<int> queue;

  std::vector<int> ref = {100, 19, 36, 17, 3, 25, 1, 2, 7};
  for (auto x : ref)
    queue.push(x);
  check_heap(queue, ref);

  queue.pop();
  check_heap(queue, {36, 19, 25, 17, 3, 7, 1, 2});

  queue.push(9);
  check_heap(queue, {36, 19, 25, 17, 3, 7, 1, 2, 9});
  //                                    ^^       ^^

  // Clear the content of the queue
  queue = PriorityQueue<int>();
  for (auto x : ref)
    queue.push(x);
  check_heap(queue, ref);

  queue.popPush(9);
  check_heap(queue, {36, 19, 25, 17, 3, 9, 1, 2, 7});
  //                                    ^^       ^^
}

template <typename PriorityQueue>
void check_heap(PriorityQueue const &queue)
{
  using ValueType = typename PriorityQueue::value_type;
  int const size = queue.size();
  auto const compare = queue.valueComp();
  auto heap = reinterpret_cast<ValueType const *>(&queue);
  for (int i = 0; i < size; ++i)
  {
    int parent = (i - 1) / 2;
    if (i > 0)
      BOOST_TEST(!compare(heap[parent], heap[i]));
    for (int child : {2 * i + 1, 2 * i + 2})
      if (child < size)
        BOOST_TEST(!compare(heap[i], heap[child]));
  }
}

BOOST_AUTO_TEST_CASE(maintain_heap_properties)
{
  PriorityQueue<int> queue;

  std::default_random_engine generator;
  std::uniform_int_distribution<int> uniform_distribution(0, 100);

  enum OperationType : int
  {
    POP,
    PUSH,
    POP_PUSH
  };
  std::discrete_distribution<int> discrete_distribution({POP, PUSH, POP_PUSH});

  // initially insert a number of elements in the queue so that it is
  // unlikely that an error will be raised (this would happen if pop() is
  // called on an empty queue or push(x) on a queue that is already at maximum
  // capacity)
  for (int i = 0; i < 64; ++i)
  {
    queue.push(uniform_distribution(generator));
    check_heap(queue);
  }

  // choose randomly whether to call pop(), push(x), or pop_push(x) and check
  // that the heap properties are maintained
  for (int i = 0; i < 512; ++i)
  {
    switch (discrete_distribution(generator))
    {
    case POP:
      queue.pop();
      break;
    case PUSH:
      queue.push(uniform_distribution(generator));
      break;
    case POP_PUSH:
      queue.popPush(uniform_distribution(generator));
      break;
    default:
      throw std::runtime_error("something went wrong");
    }
    check_heap(queue);
  }

  // remove all elements from the queue one by one
  while (!queue.empty())
  {
    queue.pop();
    check_heap(queue);
  }
}

BOOST_AUTO_TEST_SUITE_END()
