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

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <iostream>
#include <random>
#include <tuple>

#include "Search_UnitTestHelpers.hpp"
// clang-format off
#include "ArborXTest_TreeTypeTraits.hpp"
// clang-format on
#include <ArborX_Ray.hpp> // Vector

namespace tt = boost::test_tools;

struct Segment
{
  ArborX::Point start;
  ArborX::Point end;
};

template <typename MemorySpace>
struct Segments
{
  Kokkos::View<Segment *, MemorySpace> segments;
};

template <typename MemorySpace>
struct ArborX::AccessTraits<Segments<MemorySpace>, ArborX::PrimitivesTag>
{
  using memory_space = MemorySpace;
  using Primitives = Segments<MemorySpace>;

  static KOKKOS_FUNCTION std::size_t size(Primitives const &primitives)
  {
    return primitives.segments.extent(0);
  }
  static KOKKOS_FUNCTION ArborX::Box get(Primitives const &primitives,
                                         std::size_t i)
  {
    using namespace ArborX::Details;

    ArborX::Box box;
    expand(box, primitives.segments(i).start);
    expand(box, primitives.segments(i).end);
    return box;
  }
};

// distance point-segment
KOKKOS_INLINE_FUNCTION
float distance(ArborX::Point const &point, Segment const &segment)
{
  using ArborX::Details::distance;
  using ArborX::Experimental::dotProduct;
  using ArborX::Experimental::makeVector;
  using KokkosExt::max;
  using KokkosExt::min;

  auto const dir = makeVector(segment.start, segment.end);

  // The line of the segment [a,b] is parametrized as a + t * (b - a).
  // Find the projection of the point to that line, and clamp it.
  float t =
      max(0.f, min(1.f, dotProduct(dir, makeVector(segment.start, point)) /
                            dotProduct(dir, dir)));

  ArborX::Point projection;
  for (int d = 0; d < 3; ++d)
    projection[d] = segment.start[d] + t * dir[d];

  return distance(point, projection);
}

template <typename MemorySpace>
struct DistanceCallback
{
  Segments<MemorySpace> _segments;

  template <typename Query, typename Insert>
  KOKKOS_FUNCTION void operator()(Query const &, int index,
                                  Insert const &insert) const
  {
    insert(index);
  }

  template <typename Query>
  KOKKOS_FUNCTION float distance(Query const &query, int index) const
  {
    auto const &geometry = ArborX::getGeometry(query);
    auto const &segment = _segments.segments(index);
    return ::distance(geometry, segment);
  }
};

BOOST_AUTO_TEST_SUITE(ExactNearestDistance)

#ifndef ARBORX_TEST_DISABLE_NEAREST_QUERY
BOOST_AUTO_TEST_CASE_TEMPLATE(two_segments, TreeTypeTraits, TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;
  using MemorySpace = typename DeviceType::memory_space;

  Kokkos::View<Segment *, DeviceType> segments("segments", 2);
  auto segments_host = Kokkos::create_mirror_view(segments);

  // Two crossed segments. The bounding box of the larger one fully encompases
  // the bounding box of the smaller one. The point is in the left top corner.
  // x    /
  //    -/
  //    /-
  //   /
  ArborX::Point point{0.f, 1.f, 0.f};
  segments_host[0] = {{0.f, 0.f, 0.f}, {1.f, 1.f, 0.f}};
  segments_host[1] = {{0.4f, 0.6f, 0.f}, {0.6f, 0.4f, 0.f}};

  Kokkos::deep_copy(segments, segments_host);

  Tree const tree(ExecutionSpace{}, Segments<MemorySpace>{segments});

  std::vector<int> offset_ref({0, 1});
  std::vector<int> correct_indices_ref({1});
  std::vector<int> incorrect_indices_ref({0});

  // Regular kNN will produce wrong result
  ARBORX_TEST_QUERY_TREE(
      ExecutionSpace{}, tree, (makeNearestQueries<DeviceType>({{point, 1}})),
      make_compressed_storage(offset_ref, incorrect_indices_ref));

  ARBORX_TEST_QUERY_TREE_CALLBACK(
      ExecutionSpace{}, tree, (makeNearestQueries<DeviceType>({{point, 1}})),
      (DistanceCallback<MemorySpace>{Segments<MemorySpace>{segments}}),
      make_compressed_storage(offset_ref, correct_indices_ref));
}
#endif

BOOST_AUTO_TEST_SUITE_END()
