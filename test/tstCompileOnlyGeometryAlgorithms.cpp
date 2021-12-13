/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_DetailsAlgorithms.hpp>

using ArborX::Details::intersects;

namespace Test
{
struct Foo
{
};
struct Bar
{
};
struct Baz
{
};
constexpr bool intersects(Foo, Foo) { return true; }
constexpr bool intersects(Foo, Bar) { return true; }
constexpr bool intersects(Foo, Baz) { return true; }
constexpr bool intersects(Baz, Foo) { return false; }
} // namespace Test

using Test::Bar;
using Test::Baz;
using Test::Foo;

#define STATIC_ASSERT(cond) static_assert(cond, "");

STATIC_ASSERT(intersects(Foo{}, Foo{}));
STATIC_ASSERT(intersects(Foo{}, Bar{}));
STATIC_ASSERT(intersects(Foo{}, Baz{}));
STATIC_ASSERT(intersects(Bar{}, Foo{})); // fallback
// STATIC_ASSERT(intersects(Bar{}, Bar{})); // not defined
// STATIC_ASSERT(intersects(Bar{}, Baz{})); // not defined
STATIC_ASSERT(!intersects(Baz{}, Foo{}));
// STATIC_ASSERT(intersects(Baz{}, Bar{})); // not defined
// STATIC_ASSERT(intersects(Baz{}, Baz{})); // not defined
