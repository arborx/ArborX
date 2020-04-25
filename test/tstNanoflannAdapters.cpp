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

#include "ArborX_NanoflannAdapters.hpp"

#include <boost/test/unit_test.hpp>

#include <vector>

#define BOOST_TEST_MODULE NanoflannAdapters

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE(point_cloud)
{
  Kokkos::View<ArborX::Point[2], Kokkos::HostSpace> v("v");
  v(0) = {{0., 0., 0.}};
  v(1) = {{1., 1., 1.}};

  using DatasetAdapter = ArborX::NanoflannPointCloudAdapter<Kokkos::HostSpace>;
  using DistanceType = nanoflann::L2_Simple_Adaptor<double, DatasetAdapter>;
  using KDTree = nanoflann::KDTreeSingleIndexAdaptor<DistanceType,
                                                     DatasetAdapter, 3, size_t>;

  DatasetAdapter dataset_adapter(v);
  KDTree kdtree(3, dataset_adapter);
  kdtree.buildIndex();

  size_t const k = 3;
  std::vector<size_t> indices(k);
  std::vector<double> distances_sq(k);
  std::array<double, 3> query_point = {{1., 0., 0.}};
  size_t const n = kdtree.knnSearch(query_point.data(), k, indices.data(),
                                    distances_sq.data());
  BOOST_TEST(n == 2);
  BOOST_TEST(indices[0] == 0);
  BOOST_TEST(distances_sq[0] == 1., tt::tolerance(1e-14));
  BOOST_TEST(indices[1] == 1);
  BOOST_TEST(distances_sq[1] == 2., tt::tolerance(1e-14));

  std::cout << "knnSearch\n";
  for (size_t i = 0; i < n; ++i)
    std::cout << indices[i] << "  " << distances_sq[i] << "\n";

  std::vector<std::pair<size_t, double>> indices_distances;
  double const radius = 1.1;
  nanoflann::SearchParams search_params;
  size_t const m = kdtree.radiusSearch(query_point.data(), radius,
                                       indices_distances, search_params);
  BOOST_TEST(m == 1);
  BOOST_TEST(indices_distances[0].first == 0);
  BOOST_TEST(indices_distances[0].second == 1., tt::tolerance(1e-14));

  std::cout << "radiusSearch\n";
  for (size_t j = 0; j < m; ++j)
    std::cout << indices_distances[j].first << "  "
              << indices_distances[j].second << "\n";
}
