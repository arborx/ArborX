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

#include <ArborX_DetailsContainers.hpp>

#define BOOST_TEST_MODULE Containers
#include <boost/test/unit_test.hpp>

using ArborX::Details::StaticVector;
using ArborX::Details::UnmanagedStaticVector;

BOOST_AUTO_TEST_SUITE(SequenceContainers)

BOOST_AUTO_TEST_CASE(dynamic_array_with_fixed_maximum_size)
{
  StaticVector<int, 4> a;

  BOOST_TEST(a.empty());
  BOOST_TEST(a.size() == 0);
  BOOST_TEST(a.maxSize() == 4);
  BOOST_TEST(a.capacity() == 4);

  a.pushBack(255);
  BOOST_TEST(!a.empty());
  BOOST_TEST(a.size() == 1);
  BOOST_TEST(a.maxSize() == 4);
  BOOST_TEST(a.capacity() == 4);
  BOOST_TEST(a.front() == 255);
  BOOST_TEST(a[0] == 255);
  BOOST_TEST(a.back() == 255);

  a.pushBack(-1);
  BOOST_TEST(!a.empty());
  BOOST_TEST(a.size() == 2);
  BOOST_TEST(a.maxSize() == 4);
  BOOST_TEST(a.capacity() == 4);
  BOOST_TEST(a.front() == 255);
  BOOST_TEST(a[0] == 255);
  BOOST_TEST(a.back() == -1);
  BOOST_TEST(a[1] == -1);

  a.pushBack(33);
  BOOST_TEST(!a.empty());
  BOOST_TEST(a.size() == 3);
  BOOST_TEST(a.maxSize() == 4);
  BOOST_TEST(a.capacity() == 4);
  BOOST_TEST(a.front() == 255);
  BOOST_TEST(a[0] == 255);
  BOOST_TEST(a[1] == -1);
  BOOST_TEST(a.back() == 33);
  BOOST_TEST(a[2] == 33);

  a.popBack();
  BOOST_TEST(!a.empty());
  BOOST_TEST(a.size() == 2);
  BOOST_TEST(a.maxSize() == 4);
  BOOST_TEST(a.capacity() == 4);
  BOOST_TEST(a.front() == 255);
  BOOST_TEST(a.back() == -1);

  a.popBack();
  BOOST_TEST(!a.empty());
  BOOST_TEST(a.size() == 1);
  BOOST_TEST(a.maxSize() == 4);
  BOOST_TEST(a.capacity() == 4);
  BOOST_TEST(a.front() == 255);
  BOOST_TEST(a.back() == 255);

  a.popBack();
  BOOST_TEST(a.empty());
  BOOST_TEST(a.size() == 0);
  BOOST_TEST(a.maxSize() == 4);
  BOOST_TEST(a.capacity() == 4);
}

BOOST_AUTO_TEST_CASE(non_owning_view_over_dynamic_array)
{
  float data[6] = {255, 255, 255, 255, 255, 255};
  //                        ^^^^ ^^^^ ^^^^
  UnmanagedStaticVector<float> a(data + 2, 3);

  BOOST_TEST(!std::is_default_constructible<decltype(a)>::value);
  BOOST_TEST(a.data() == data + 2);

  BOOST_TEST(a.empty());
  BOOST_TEST(a.size() == 0);
  BOOST_TEST(a.maxSize() == 3);
  BOOST_TEST(a.capacity() == 3);

  a.pushBack(0);
  BOOST_TEST(!a.empty());
  BOOST_TEST(a.size() == 1);
  BOOST_TEST(a.maxSize() == 3);
  BOOST_TEST(a.capacity() == 3);
  BOOST_TEST(a.front() == 0);
  BOOST_TEST(a.back() == 0);
  BOOST_TEST(a[0] == 0);

  a.pushBack(1);
  BOOST_TEST(!a.empty());
  BOOST_TEST(a.size() == 2);
  BOOST_TEST(a.maxSize() == 3);
  BOOST_TEST(a.capacity() == 3);
  BOOST_TEST(a.front() == 0);
  BOOST_TEST(a[0] == 0);
  BOOST_TEST(a.back() == 1);
  BOOST_TEST(a[1] == 1);

  a.pushBack(2);
  BOOST_TEST(!a.empty());
  BOOST_TEST(a.size() == 3);
  BOOST_TEST(a.maxSize() == 3);
  BOOST_TEST(a.capacity() == 3);
  BOOST_TEST(a.front() == 0);
  BOOST_TEST(a[0] == 0);
  BOOST_TEST(a[1] == 1);
  BOOST_TEST(a.back() == 2);
  BOOST_TEST(a[2] == 2);

  a.popBack();
  BOOST_TEST(!a.empty());
  BOOST_TEST(a.size() == 2);
  BOOST_TEST(a.maxSize() == 3);
  BOOST_TEST(a.capacity() == 3);
  BOOST_TEST(a.front() == 0);
  BOOST_TEST(a[0] == 0);
  BOOST_TEST(a.back() == 1);
  BOOST_TEST(a[1] == 1);

  BOOST_TEST(data[0] == 255);
  BOOST_TEST(data[1] == 255);
  BOOST_TEST(data[2] == 0);
  BOOST_TEST(data[3] == 1);
  BOOST_TEST(data[4] == 2);
  BOOST_TEST(data[5] == 255);
}

BOOST_AUTO_TEST_SUITE_END()
