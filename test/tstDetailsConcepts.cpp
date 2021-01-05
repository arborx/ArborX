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

#include <ArborX_DetailsConcepts.hpp>

#include <Kokkos_Core.hpp>

#include <tuple>

using ArborX::Box;
using ArborX::Point;
using ArborX::Sphere;
using ArborX::Details::has_centroid;
using ArborX::Details::is_expandable;

static_assert(is_expandable<Box, Box>::value, "");
static_assert(is_expandable<Box, Box const>::value, "");
static_assert(!is_expandable<Box const, Box>::value, "");

static_assert(is_expandable<Box, Point>::value, "");
static_assert(is_expandable<Box, Sphere>::value, "");

static_assert(!is_expandable<Point, Point>::value, "");
static_assert(!is_expandable<Point, Box>::value, "");
static_assert(!is_expandable<Point, Sphere>::value, "");

// NOTE Possible but not implemented
static_assert(!is_expandable<Sphere, Point>::value, "");
static_assert(!is_expandable<Sphere, Box>::value, "");
static_assert(!is_expandable<Sphere, Sphere>::value, "");

static_assert(has_centroid<Box, Point>::value, "");
static_assert(has_centroid<Point, Point>::value, "");
static_assert(has_centroid<Sphere, Point>::value, "");

using ArborX::Details::is_complete;

template <typename T, typename Enable = void>
struct NotSpecializedForIntegralTypes;
template <typename T>
struct NotSpecializedForIntegralTypes<
    T, std::enable_if_t<!std::is_integral<T>::value>>
{
};

static_assert(is_complete<NotSpecializedForIntegralTypes<float>>::value, "");
static_assert(!is_complete<NotSpecializedForIntegralTypes<int>>::value, "");

using ArborX::Details::first_template_parameter_t;
static_assert(std::is_same<first_template_parameter_t<std::tuple<int, float>>,
                           int>::value,
              "");

int main() {}
