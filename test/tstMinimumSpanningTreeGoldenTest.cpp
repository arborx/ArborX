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

#include "ArborXTest_StdVectorToKokkosView.hpp"
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_MinimumSpanningTree.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>
#include <boost/tokenizer.hpp>

#include <fstream>
#include <map>
#include <numeric> // accumulate

namespace Test
{
struct UndirectedEdge : ArborX::Details::WeightedEdge
{
private:
  friend bool operator==(UndirectedEdge const &lhs, UndirectedEdge const &rhs)
  {
    return ((lhs.source == rhs.source && lhs.target == rhs.target) ||
            (lhs.source == rhs.target && lhs.target == rhs.source));
  }
  friend std::ostream &operator<<(std::ostream &os,
                                  ArborX::Details::WeightedEdge const &e)
  {
    os << e.source << " -> " << e.target << " [weight=" << e.weight << "]";
    return os;
  }
};

auto parsePointsFromCSVFile(std::string const &filename)
{
  std::fstream fin(filename, std::ios::in);
  using Tokenizer = boost::tokenizer<boost::escaped_list_separator<char>>;
  std::string line;
  std::vector<ArborX::Point> points;
  assert(fin.is_open());
  while (std::getline(fin, line))
  {
    Tokenizer tok(line);
    auto first = tok.begin();
    auto const last = tok.end();
    points.emplace_back(ArborX::Point{std::stof(*first++), std::stof(*first++),
                                      std::stof(*first++)});
    assert(first == last);
  }
  return points;
}

auto parseEdgesFromCSVFile(std::string const &filename)
{
  std::fstream fin(filename, std::ios::in);
  using Tokenizer = boost::tokenizer<boost::escaped_list_separator<char>>;
  std::string line;
  std::vector<Test::UndirectedEdge> edges;
  assert(fin.is_open());
  while (std::getline(fin, line))
  {
    Tokenizer tok(line);
    auto first = tok.begin();
    auto const last = tok.end();
    edges.emplace_back(Test::UndirectedEdge{
        std::stoi(*first++), std::stoi(*first++), std::stof(*first++)});
    assert(first == last);
  }
  return edges;
}
} // namespace Test

BOOST_AUTO_TEST_CASE_TEMPLATE(minimum_spanning_tree_golden_test, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  ExecutionSpace exec_space;

  auto points = ArborXTest::toView<ExecutionSpace>(
      Test::parsePointsFromCSVFile("mst_golden_test_points.csv"),
      "Tests::points");

  auto edges_ref = Test::parseEdgesFromCSVFile("mst_golden_test_edges.csv");
  std::sort(edges_ref.data(), edges_ref.data() + edges_ref.size());

  using ArborX::Details::MinimumSpanningTree;
  auto edges = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{},
      MinimumSpanningTree<MemorySpace>(exec_space, points).edges);
  std::sort(edges.data(), edges.data() + edges.size());

  BOOST_TEST(
      edges_ref ==
          (Kokkos::View<Test::UndirectedEdge const *, Kokkos::HostSpace>(
              reinterpret_cast<Test::UndirectedEdge const *>(edges.data()),
              edges.size())),
      boost::test_tools::per_element());

  // clang-format off
  // Computed with the following Python code with variable k:
  //   import numpy as np
  //   import hdbscan
  //   clusterer = hdbscan.HDBSCAN(min_cluster_size=k-1, gen_min_span_tree=True)
  //   filename = "mst_golden_test_points.csv"
  //   points = np.loadtxt(filename, delimiter=",", dtype="double", comments="#")
  //   clusterer.fit(points)
  //   print(np.sum(clusterer.minimum_spanning_tree_._mst[:,2]))
  // clang-format on
  std::map<int, double> ref_total_weight;
  ref_total_weight[5] = 102.68084503576422;
  ref_total_weight[10] = 138.0244333174116;
  ref_total_weight[15] = 162.51948793942978;

  std::map<int, double> total_weight;
  for (auto k : {5, 10, 15})
  {
    auto edges = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{},
        MinimumSpanningTree<MemorySpace>(exec_space, points, k).edges);
    total_weight[k] = std::accumulate(
        edges.data(), edges.data() + edges.size(), 0.,
        [](auto const &sum, auto const &b) { return sum + b.weight; });
  }

  namespace tt = boost::test_tools;
  BOOST_TEST(total_weight[5] == ref_total_weight[5], tt::tolerance(1e-8));
  BOOST_TEST(total_weight[10] == ref_total_weight[10], tt::tolerance(1e-8));
  BOOST_TEST(total_weight[15] == ref_total_weight[15], tt::tolerance(1e-8));
}
