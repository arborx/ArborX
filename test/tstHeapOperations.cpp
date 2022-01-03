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

#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_DetailsHeap.hpp>
#include <ArborX_DetailsPriorityQueue.hpp> // Less, Greater

#include <Kokkos_Array.hpp>

#include <boost/test/unit_test.hpp>

using namespace ArborX::Details;
namespace tt = boost::test_tools;

BOOST_AUTO_TEST_SUITE(HeapOperations)

BOOST_AUTO_TEST_CASE(push_heap)
{
  // Here checking against the example of binary heap insertion from
  // https://en.wikipedia.org/wiki/Binary_heap#Insert
  Kokkos::Array<int, 6> a = {11, 5, 8, 3, 4, 15};
  BOOST_TEST(isHeap(a.data(), a.data() + 5, Less<int>()));
  BOOST_TEST(!isHeap(a.data(), a.data() + 6, Less<int>()));
  pushHeap(a.data(), a.data() + 6, Less<int>());
  BOOST_TEST(isHeap(a.data(), a.data() + 6, Less<int>()));
  Kokkos::Array<int, 6> ref = {15, 5, 11, 3, 4, 8};
  BOOST_TEST(a == ref, tt::per_element());
}

BOOST_AUTO_TEST_CASE(pop_heap)
{
  // See https://en.wikipedia.org/wiki/Binary_heap#Extract
  Kokkos::Array<int, 5> a = {11, 5, 8, 3, 4};
  BOOST_TEST(isHeap(a.data(), a.data() + 5, Less<int>()));
  popHeap(a.data(), a.data() + 5, Less<int>());
  BOOST_TEST(isHeap(a.data(), a.data() + 4, Less<int>()));
  BOOST_TEST(!isHeap(a.data(), a.data() + 5, Less<int>()));
  Kokkos::Array<int, 5> ref = {8, 5, 4, 3, 11};
  BOOST_TEST(a == ref, tt::per_element());
}

BOOST_AUTO_TEST_CASE(max_heap)
{
  // Here attempting to reproduce examples code from cppreference.com for
  // std::push_heap() and std::pop_heap().  It turns out that calling
  // std::make_heap() on { 3, 1, 4, 1, 5, 9 } is not equivalent to inserting
  // one element at a time (I also tried to insert them in reverse order out
  // of curiosity) and yields 9 5 4 1 1 3 :/
  // cpluplus.com does make a note that the order of the elements will depend
  // on implementation.
  // Nevertheless, I thought it helps understanding how the algorithm works so
  // I decided to keep it as a unit test.
  Kokkos::Array<int, 7> a = {3, 1, 4, 1, 5, 9, 6};
  Kokkos::Array<int, 7> ref;

  pushHeap(a.data(), a.data() + 0, Less<int>());
  ref = {3, 1, 4, 1, 5, 9, 6};
  BOOST_TEST(a == ref, tt::per_element());

  pushHeap(a.data(), a.data() + 1, Less<int>());
  ref = {3, 1, 4, 1, 5, 9, 6};
  BOOST_TEST(a == ref, tt::per_element());

  pushHeap(a.data(), a.data() + 2, Less<int>());
  ref = {3, 1, 4, 1, 5, 9, 6};
  BOOST_TEST(a == ref, tt::per_element());

  pushHeap(a.data(), a.data() + 3, Less<int>());
  ref = {4, 1, 3, 1, 5, 9, 6};
  BOOST_TEST(a == ref, tt::per_element());

  pushHeap(a.data(), a.data() + 4, Less<int>());
  ref = {4, 1, 3, 1, 5, 9, 6};
  BOOST_TEST(a == ref, tt::per_element());

  pushHeap(a.data(), a.data() + 5, Less<int>());
  ref = {5, 4, 3, 1, 1, 9, 6};
  BOOST_TEST(a == ref, tt::per_element());

  pushHeap(a.data(), a.data() + 6, Less<int>());
  ref = {9, 4, 5, 1, 1, 3, 6};
  BOOST_TEST(a == ref, tt::per_element());

  popHeap(a.data(), a.data() + 6, Less<int>());
  ref = {5, 4, 3, 1, 1, 9, 6};
  BOOST_TEST(a == ref, tt::per_element());

  pushHeap(a.data(), a.data() + 6, Less<int>());
  ref = {9, 4, 5, 1, 1, 3, 6};
  BOOST_TEST(a == ref, tt::per_element());

  pushHeap(a.data(), a.data() + 7, Less<int>());
  ref = {9, 4, 6, 1, 1, 3, 5};
  BOOST_TEST(a == ref, tt::per_element());
}

BOOST_AUTO_TEST_CASE(min_heap)
{
  // Here reproducing the example of a binary min heap from Wikipedia and then
  // popping all elements one by one.
  // This helped resolving an issue with ProprityQueue::pop() that was not
  // exposed by other unit tests in this file, partly because they were too
  // trivial.  The bug was only showing with real kNN search problems when
  // comparing results from BoundingVolumeHierarchy with boost::rtree.
  Kokkos::Array<int, 9> a = {1, 2, 3, 17, 19, 36, 7, 25, 100};
  Kokkos::Array<int, 9> ref;
  for (int i = 0; i < 9; ++i)
  {
    pushHeap(a.data(), a.data() + i, Greater<int>());
    ref = {1, 2, 3, 17, 19, 36, 7, 25, 100};
    BOOST_TEST(a == ref, tt::per_element());
  }

  popHeap(a.data(), a.data() + 9, Greater<int>());
  ref = {2, 17, 3, 25, 19, 36, 7, 100, 1};
  BOOST_TEST(a == ref, tt::per_element());

  popHeap(a.data(), a.data() + 8, Greater<int>());
  ref = {3, 17, 7, 25, 19, 36, 100, 2, 1};
  BOOST_TEST(a == ref, tt::per_element());

  popHeap(a.data(), a.data() + 7, Greater<int>());
  ref = {7, 17, 36, 25, 19, 100, 3, 2, 1};
  BOOST_TEST(a == ref, tt::per_element());

  popHeap(a.data(), a.data() + 6, Greater<int>());
  ref = {17, 19, 36, 25, 100, 7, 3, 2, 1};
  BOOST_TEST(a == ref, tt::per_element());

  popHeap(a.data(), a.data() + 5, Greater<int>());
  ref = {19, 25, 36, 100, 17, 7, 3, 2, 1};
  BOOST_TEST(a == ref, tt::per_element());

  popHeap(a.data(), a.data() + 4, Greater<int>());
  ref = {25, 100, 36, 19, 17, 7, 3, 2, 1};
  BOOST_TEST(a == ref, tt::per_element());

  popHeap(a.data(), a.data() + 3, Greater<int>());
  ref = {36, 100, 25, 19, 17, 7, 3, 2, 1};
  BOOST_TEST(a == ref, tt::per_element());

  popHeap(a.data(), a.data() + 2, Greater<int>());
  ref = {100, 36, 25, 19, 17, 7, 3, 2, 1};
  BOOST_TEST(a == ref, tt::per_element());

  popHeap(a.data(), a.data() + 1, Greater<int>());
  ref = {100, 36, 25, 19, 17, 7, 3, 2, 1};
  BOOST_TEST(a == ref, tt::per_element());

  popHeap(a.data(), a.data() + 0, Greater<int>());
  ref = {100, 36, 25, 19, 17, 7, 3, 2, 1};
  BOOST_TEST(a == ref, tt::per_element());
}

BOOST_AUTO_TEST_CASE(make_heap)
{
  for (auto v : {
           std::vector<int>{}, std::vector<int>{6}, std::vector<int>{2, 1},
           std::vector<int>{1, 6, 2, 2, 9, 4, 16},
           std::vector<int>{8, 6, 7, 2, 0}, std::vector<int>{3, 3, 3, 3, 3, 1},
           std::vector<int>{36, 19, 25, 17, 3, 7, 1, 2, 9}, // <- already a heap
       })
  {
    makeHeap(v.data(), v.data() + v.size(), Less<int>());
    BOOST_TEST(std::is_heap(v.begin(), v.end()));
  }
}

BOOST_AUTO_TEST_CASE(sort_heap)
{
  for (auto heap : {std::vector<int>{}, std::vector<int>{3},
                    std::vector<int>{36, 19, 25, 17, 3, 7, 1, 2, 9},
                    std::vector<int>{36, 19, 25, 17, 3, 9, 1, 2, 7},
                    std::vector<int>{100, 19, 36, 17, 3, 25, 1, 2, 7},
                    std::vector<int>{15, 5, 11, 3, 4, 8}})
  {
    sortHeap(heap.data(), heap.data() + heap.size(), Less<int>());
    // std::sort_heap( heap.begin(), heap.end() );
    BOOST_TEST(std::is_sorted(heap.begin(), heap.end()));
  }
}

BOOST_AUTO_TEST_CASE(is_heap)
{
  for (auto heap : {std::vector<int>{36, 19, 25, 17, 3, 7, 1, 2, 9},
                    std::vector<int>{36, 19, 25, 17, 3, 9, 1, 2, 7},
                    std::vector<int>{100, 19, 36, 17, 3, 25, 1, 2, 7},
                    std::vector<int>{15, 5, 11, 3, 4, 8}})
  {
    BOOST_TEST(isHeap(heap.data(), heap.data() + heap.size(), Less<int>()));
  }
  for (auto not_heap : {std::vector<int>{0, 1, 2, 3, 4, 3, 2, 1, 0},
                        std::vector<int>{2, 1, 0, 1, 2}})
  {
    BOOST_TEST(!isHeap(not_heap.data(), not_heap.data() + not_heap.size(),
                       Less<int>()));
  }
}

BOOST_AUTO_TEST_SUITE_END()
