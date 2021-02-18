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

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <iostream>
#include <random>
#include <tuple>

#include "Search_UnitTestHelpers.hpp"
// clang-format off
#include "ArborXTest_TreeTypeTraits.hpp"
// clang-format on

BOOST_AUTO_TEST_SUITE(ManufacturedSolution)

namespace tt = boost::test_tools;

// FIXME temporary workaround bug in HIP-Clang (register spill)
#if defined(KOKKOS_ENABLE_HIP)
BOOST_TEST_DECORATOR(*boost::unit_test::expected_failures(3))
#endif
BOOST_AUTO_TEST_CASE_TEMPLATE(structured_grid, TreeTypeTraits,
                              TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;

  float Lx = 100.0;
  float Ly = 100.0;
  float Lz = 100.0;
  int nx = 11;
  int ny = 11;
  int nz = 11;
  int n = nx * ny * nz;
  float hx = Lx / (nx - 1);
  float hy = Ly / (ny - 1);
  float hz = Lz / (nz - 1);

  std::function<int(int, int, int)> ind = [nx, ny](int i, int j, int k) {
    return i + j * nx + k * (nx * ny);
  };

  Kokkos::View<ArborX::Box *, DeviceType> bounding_boxes("bounding_boxes", n);
  auto bounding_boxes_host = Kokkos::create_mirror_view(bounding_boxes);

  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k)
      {
        ArborX::Point p{{i * hx, j * hy, k * hz}};
        bounding_boxes_host[ind(i, j, k)] = {p, p};
      }
  Kokkos::deep_copy(bounding_boxes, bounding_boxes_host);

  Tree const tree(ExecutionSpace{}, bounding_boxes);

  std::vector<int> offset_ref(n + 1);
  std::vector<int> indices_ref;

  // (i) use the same objects for the queries as the objects used in BVH
  // construction
  // i-2  i-1  i  i+1
  //
  //  o    o   o   o   j+1
  //          ---
  //  o    o | x | o   j
  //          ---
  //  o    o   o   o   j-1
  //
  //  o    o   o   o   j-2
  //
  indices_ref.resize(n);
  std::iota(offset_ref.begin(), offset_ref.end(), 0);
  std::iota(indices_ref.begin(), indices_ref.end(), 0);

  Kokkos::View<decltype(ArborX::intersects(ArborX::Box{})) *, DeviceType>
      queries("queries", n);
  Kokkos::parallel_for("fill_queries",
                       Kokkos::RangePolicy<ExecutionSpace>(0, n),
                       KOKKOS_LAMBDA(int i) {
                         queries(i) = ArborX::intersects(bounding_boxes(i));
                       });
  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree, queries,
                         make_reference_solution(indices_ref, offset_ref));

  // (ii) use bounding boxes that intersects with first neighbors
  //
  // i-2  i-1  i  i+1
  //
  //  o    x---x---x   j+1
  //       |       |
  //  o    x   x   x   j
  //       |       |
  //  o    x---x---x   j-1
  //
  //  o    o   o   o   j-2
  //
  std::vector<std::set<int>> ref(n);
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k)
      {
        int const index = ind(i, j, k);
        // bounding box around nodes of the structured grid will
        // intersect with neighboring nodes
        bounding_boxes_host[index] = {
            {{(i - 1) * hx, (j - 1) * hy, (k - 1) * hz}},
            {{(i + 1) * hx, (j + 1) * hy, (k + 1) * hz}}};
        // fill in reference solution to check against the collision
        // list computed during the tree traversal
        if ((i > 0) && (j > 0) && (k > 0))
          ref[index].emplace(ind(i - 1, j - 1, k - 1));
        if ((i > 0) && (k > 0))
          ref[index].emplace(ind(i - 1, j, k - 1));
        if ((i > 0) && (j < ny - 1) && (k > 0))
          ref[index].emplace(ind(i - 1, j + 1, k - 1));
        if ((i > 0) && (j > 0))
          ref[index].emplace(ind(i - 1, j - 1, k));
        if (i > 0)
          ref[index].emplace(ind(i - 1, j, k));
        if ((i > 0) && (j < ny - 1))
          ref[index].emplace(ind(i - 1, j + 1, k));
        if ((i > 0) && (j > 0) && (k < nz - 1))
          ref[index].emplace(ind(i - 1, j - 1, k + 1));
        if ((i > 0) && (k < nz - 1))
          ref[index].emplace(ind(i - 1, j, k + 1));
        if ((i > 0) && (j < ny - 1) && (k < nz - 1))
          ref[index].emplace(ind(i - 1, j + 1, k + 1));

        if ((j > 0) && (k > 0))
          ref[index].emplace(ind(i, j - 1, k - 1));
        if (k > 0)
          ref[index].emplace(ind(i, j, k - 1));
        if ((j < ny - 1) && (k > 0))
          ref[index].emplace(ind(i, j + 1, k - 1));
        if (j > 0)
          ref[index].emplace(ind(i, j - 1, k));
        if (true) // NOLINT
          ref[index].emplace(ind(i, j, k));
        if (j < ny - 1)
          ref[index].emplace(ind(i, j + 1, k));
        if ((j > 0) && (k < nz - 1))
          ref[index].emplace(ind(i, j - 1, k + 1));
        if (k < nz - 1)
          ref[index].emplace(ind(i, j, k + 1));
        if ((j < ny - 1) && (k < nz - 1))
          ref[index].emplace(ind(i, j + 1, k + 1));

        if ((i < nx - 1) && (j > 0) && (k > 0))
          ref[index].emplace(ind(i + 1, j - 1, k - 1));
        if ((i < nx - 1) && (k > 0))
          ref[index].emplace(ind(i + 1, j, k - 1));
        if ((i < nx - 1) && (j < ny - 1) && (k > 0))
          ref[index].emplace(ind(i + 1, j + 1, k - 1));
        if ((i < nx - 1) && (j > 0))
          ref[index].emplace(ind(i + 1, j - 1, k));
        if (i < nx - 1)
          ref[index].emplace(ind(i + 1, j, k));
        if ((i < nx - 1) && (j < ny - 1))
          ref[index].emplace(ind(i + 1, j + 1, k));
        if ((i < nx - 1) && (j > 0) && (k < nz - 1))
          ref[index].emplace(ind(i + 1, j - 1, k + 1));
        if ((i < nx - 1) && (k < nz - 1))
          ref[index].emplace(ind(i + 1, j, k + 1));
        if ((i < nx - 1) && (j < ny - 1) && (k < nz - 1))
          ref[index].emplace(ind(i + 1, j + 1, k + 1));
      }
  Kokkos::deep_copy(bounding_boxes, bounding_boxes_host);

  indices_ref.resize(0);
  for (int i = 0; i < n; ++i)
  {
    std::copy(ref[i].begin(), ref[i].end(), std::back_inserter(indices_ref));
    offset_ref[i + 1] = indices_ref.size();
  }

  Kokkos::parallel_for("fill_first_neighbors_queries",
                       Kokkos::RangePolicy<ExecutionSpace>(0, n),
                       KOKKOS_LAMBDA(int i) {
                         queries[i] = ArborX::intersects(bounding_boxes[i]);
                       });
  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree, queries,
                         make_reference_solution(indices_ref, offset_ref));

  // (iii) use random points
  //
  // i-1      i      i+1
  //
  //  o       o       o   j+1
  //         -------
  //        |       |
  //        |   +   |
  //  o     | x     | o   j
  //         -------
  //
  //  o       o       o   j-1
  //
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution_x(0.0, Lz);
  std::uniform_real_distribution<double> distribution_y(0.0, Ly);
  std::uniform_real_distribution<double> distribution_z(0.0, Lz);

  std::iota(offset_ref.begin(), offset_ref.end(), 0);
  indices_ref.resize(n);
  for (int l = 0; l < n; ++l)
  {
    double x = distribution_x(generator);
    double y = distribution_y(generator);
    double z = distribution_z(generator);
    bounding_boxes_host(l) = {{{x - 0.5 * hx, y - 0.5 * hy, z - 0.5 * hz}},
                              {{x + 0.5 * hx, y + 0.5 * hy, z + 0.5 * hz}}};

    auto const i = static_cast<int>(std::round(x / hx));
    auto const j = static_cast<int>(std::round(y / hy));
    auto const k = static_cast<int>(std::round(z / hz));
    // Save the indices for the check
    indices_ref[l] = ind(i, j, k);
  }
  Kokkos::deep_copy(bounding_boxes, bounding_boxes_host);

  Kokkos::parallel_for("fill_first_neighbors_queries",
                       Kokkos::RangePolicy<ExecutionSpace>(0, n),
                       KOKKOS_LAMBDA(int i) {
                         queries[i] = ArborX::intersects(bounding_boxes[i]);
                       });
  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree, queries,
                         make_reference_solution(indices_ref, offset_ref));
}

BOOST_AUTO_TEST_SUITE_END()
