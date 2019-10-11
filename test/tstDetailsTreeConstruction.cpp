/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_DetailsKokkosExt.hpp>  // clz
#include <ArborX_DetailsMortonCode.hpp> // expandBits, morton3D
#include <ArborX_DetailsSortUtils.hpp>  // sortObjects
#include <ArborX_DetailsTreeConstruction.hpp>
#include <ArborX_DetailsUtils.hpp> // iota

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <array>
#include <bitset>
#include <functional>
#include <limits>
#include <sstream>
#include <vector>

#define BOOST_TEST_MODULE DetailsTreeConstruction

namespace details = ArborX::Details;

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE_TEMPLATE(morton_codes, DeviceType, ARBORX_DEVICE_TYPES)
{
  std::vector<ArborX::Point> points = {
      {{0.0, 0.0, 0.0}},          {{0.25, 0.75, 0.25}}, {{0.75, 0.25, 0.25}},
      {{0.75, 0.75, 0.25}},       {{1.33, 2.33, 3.33}}, {{1.66, 2.66, 3.66}},
      {{1024.0, 1024.0, 1024.0}},
  };
  int const n = points.size();
  // lower left front corner corner of the octant the points fall in
  std::vector<std::array<unsigned int, 3>> anchors = {
      {{0, 0, 0}}, {{0, 0, 0}}, {{0, 0, 0}},         {{0, 0, 0}},
      {{1, 2, 3}}, {{1, 2, 3}}, {{1023, 1023, 1023}}};
  auto fun = [](std::array<unsigned int, 3> const &anchor) {
    unsigned int i = std::get<0>(anchor);
    unsigned int j = std::get<1>(anchor);
    unsigned int k = std::get<2>(anchor);
    return 4 * details::expandBits(i) + 2 * details::expandBits(j) +
           details::expandBits(k);
  };
  std::vector<unsigned int> ref(n, std::numeric_limits<unsigned int>::max());
  for (int i = 0; i < n; ++i)
    ref[i] = fun(anchors[i]);
  // using points rather than boxes for convenience here but still have to
  // build the axis-aligned bounding boxes around them
  Kokkos::View<ArborX::Box *, DeviceType> boxes("boxes", n);
  auto boxes_host = Kokkos::create_mirror_view(boxes);
  for (int i = 0; i < n; ++i)
    details::expand(boxes_host(i), points[i]);
  Kokkos::deep_copy(boxes, boxes_host);

  ArborX::Box scene_host;
  details::TreeConstruction<DeviceType>::calculateBoundingBoxOfTheScene(
      boxes, scene_host);

  BOOST_TEST(
      details::equals(scene_host, {{{0., 0., 0.}}, {{1024., 1024., 1024.}}}));

  Kokkos::View<unsigned int *, DeviceType> morton_codes("morton_codes", n);
  details::TreeConstruction<DeviceType>::assignMortonCodes(boxes, morton_codes,
                                                           scene_host);
  auto morton_codes_host = Kokkos::create_mirror_view(morton_codes);
  Kokkos::deep_copy(morton_codes_host, morton_codes);
  BOOST_TEST(morton_codes_host == ref, tt::per_element());
}

template <typename DeviceType>
class FillK
{
public:
  FillK(Kokkos::View<unsigned int *, DeviceType> k)
      : _k(k)
  {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(int const i) const { _k[i] = 4 - i; }

private:
  Kokkos::View<unsigned int *, DeviceType> _k;
};

BOOST_AUTO_TEST_CASE_TEMPLATE(indirect_sort, DeviceType, ARBORX_DEVICE_TYPES)
{
  // need a functionality that sort objects based on their Morton code and
  // also returns the indices in the original configuration

  // dummy unsorted Morton codes and corresponding sorted indices as reference
  // solution
  //
  using ExecutionSpace = typename DeviceType::execution_space;
  unsigned int const n = 4;
  Kokkos::View<unsigned int *, DeviceType> k("k", n);
  // Fill K with 4, 3, 2, 1
  FillK<DeviceType> fill_k_functor(k);
  Kokkos::parallel_for("fill_k", Kokkos::RangePolicy<ExecutionSpace>(0, n),
                       fill_k_functor);
  ExecutionSpace().fence();

  std::vector<size_t> ref = {3, 2, 1, 0};
  // sort morton codes and object ids
  auto ids = details::sortObjects(k);

  auto k_host = Kokkos::create_mirror_view(k);
  Kokkos::deep_copy(k_host, k);
  auto ids_host = Kokkos::create_mirror_view(ids);
  Kokkos::deep_copy(ids_host, ids);

  // check that they are sorted
  for (unsigned int i = 0; i < n; ++i)
    BOOST_TEST(k_host[i] == i + 1);
  // check that ids are properly ordered
  BOOST_TEST(ids_host == ref, tt::per_element());
}

BOOST_AUTO_TEST_CASE(number_of_leading_zero_bits)
{
  using KokkosExt::clz;
  BOOST_TEST(clz(0) == 32);
  BOOST_TEST(clz(1) == 31);
  BOOST_TEST(clz(2) == 30);
  BOOST_TEST(clz(3) == 30);
  BOOST_TEST(clz(4) == 29);
  BOOST_TEST(clz(5) == 29);
  BOOST_TEST(clz(6) == 29);
  BOOST_TEST(clz(7) == 29);
  BOOST_TEST(clz(8) == 28);
  BOOST_TEST(clz(9) == 28);
  // bitwise exclusive OR operator to compare bits
  BOOST_TEST(clz(1 ^ 0) == 31);
  BOOST_TEST(clz(2 ^ 0) == 30);
  BOOST_TEST(clz(2 ^ 1) == 30);
  BOOST_TEST(clz(3 ^ 0) == 30);
  BOOST_TEST(clz(3 ^ 1) == 30);
  BOOST_TEST(clz(3 ^ 2) == 31);
  BOOST_TEST(clz(4 ^ 0) == 29);
  BOOST_TEST(clz(4 ^ 1) == 29);
  BOOST_TEST(clz(4 ^ 2) == 29);
  BOOST_TEST(clz(4 ^ 3) == 29);
}

template <typename DeviceType>
class FillFi
{
public:
  KOKKOS_INLINE_FUNCTION
  FillFi(Kokkos::View<unsigned int *, DeviceType> fi)
      : _fi(fi)
  {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(int const i) const
  {
    // NOTE: Morton codes below are **not** unique
    unsigned int fi_array[] = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144};

    _fi[i] = fi_array[i];
  }

private:
  Kokkos::View<unsigned int *, DeviceType> _fi;
};

template <typename DeviceType>
class ComputeResults
{
public:
  KOKKOS_INLINE_FUNCTION
  ComputeResults(Kokkos::View<unsigned int *, DeviceType> fi,
                 Kokkos::View<int *, DeviceType> results)
      : _fi(fi)
      , _results(results)
  {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(int const i) const
  {
    int index_1[] = {0, 0, 1, 1, 1, 2, 2, 0, 12, 12};
    int index_2[] = {0, 1, 0, 1, 2, 1, 2, -1, 12, 13};

    _results[i] = ArborX::Details::TreeConstruction<DeviceType>::commonPrefix(
        _fi, index_1[i], index_2[i]);
  }

private:
  Kokkos::View<unsigned int *, DeviceType> _fi;
  Kokkos::View<int *, DeviceType> _results;
};

BOOST_AUTO_TEST_CASE_TEMPLATE(common_prefix, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  int const n = 13;
  Kokkos::View<unsigned int *, DeviceType> fi("fi", n);
  FillFi<DeviceType> fill_fi_functor(fi);
  Kokkos::parallel_for("fill_fi", Kokkos::RangePolicy<ExecutionSpace>(0, n),
                       fill_fi_functor);
  ExecutionSpace().fence();

  int const n_tests = 10;
  Kokkos::View<int *, DeviceType> results("results", n_tests);

  ComputeResults<DeviceType> compute_results_functor(fi, results);
  Kokkos::parallel_for("compute_results",
                       Kokkos::RangePolicy<ExecutionSpace>(0, n_tests),
                       compute_results_functor);
  ExecutionSpace().fence();

  auto results_host = Kokkos::create_mirror_view(results);
  Kokkos::deep_copy(results_host, results);

  auto fi_host = Kokkos::create_mirror_view(fi);
  Kokkos::deep_copy(fi_host, fi);

  BOOST_TEST(results_host[0] == 32 + 32);
  BOOST_TEST(results_host[1] == 31);
  BOOST_TEST(results_host[2] == 31);
  // duplicate Morton codes
  BOOST_TEST(fi_host[1] == 1);
  BOOST_TEST(fi_host[1] == fi_host[2]);
  BOOST_TEST(results_host[3] == 64);
  BOOST_TEST(results_host[4] == 32 + 30);
  BOOST_TEST(results_host[5] == 62);
  BOOST_TEST(results_host[6] == 64);
  // by definition \delta(i, j) = -1 when j \notin [0, n-1]
  BOOST_TEST(results_host[7] == -1);
  BOOST_TEST(results_host[8] == 64);
  BOOST_TEST(results_host[9] == -1);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(example_tree_construction, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  if (!Kokkos::Impl::SpaceAccessibility<
          Kokkos::HostSpace, typename DeviceType::memory_space>::accessible)
    return;

  // This is the example from the articles by Karras.
  // See
  // https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
  int const n = 8;
  Kokkos::View<unsigned int *, DeviceType> sorted_morton_codes(
      "sorted_morton_codes", n);
  std::vector<std::string> s{
      "00001", "00010", "00100", "00101", "10011", "11000", "11001", "11110",
  };
  for (int i = 0; i < n; ++i)
  {
    std::bitset<6> b(s[i]);
    std::cout << b << "  " << b.to_ulong() << "\n";
    sorted_morton_codes(i) = b.to_ulong();
  }

  // reference solution for a recursive traversal from top to bottom
  // starting from root, visiting first the left child and then the right one
  std::ostringstream ref;
  ref << "I0"
      << "I3"
      << "I1"
      << "L0"
      << "L1"
      << "I2"
      << "L2"
      << "L3"
      << "I4"
      << "L4"
      << "I5"
      << "I6"
      << "L5"
      << "L6"
      << "L7";
  std::cout << "ref=" << ref.str() << "\n";

  // hierarchy generation
  Kokkos::View<ArborX::Node *, DeviceType> leaf_nodes("leaf_nodes", n);
  Kokkos::View<ArborX::Node *, DeviceType> internal_nodes("internal_nodes",
                                                          n - 1);
  std::function<void(ArborX::Node *, std::ostream &)> traverseRecursive;
  traverseRecursive = [&leaf_nodes, &internal_nodes, &traverseRecursive](
                          ArborX::Node *node, std::ostream &os) {
    if (std::any_of(leaf_nodes.data(), leaf_nodes.data() + n,
                    [node](ArborX::Node const &leaf_node) {
                      return std::addressof(leaf_node) == node;
                    }))
    {
      os << "L" << node - leaf_nodes.data();
    }
    else
    {
      os << "I" << node - internal_nodes.data();
      for (ArborX::Node *child : {node->children.first, node->children.second})
        traverseRecursive(child, os);
    }
  };

  Kokkos::View<int *, DeviceType> parents("parents", 2 * n + 1);
  Kokkos::deep_copy(parents, -1);

  details::TreeConstruction<DeviceType>::generateHierarchy(
      sorted_morton_codes, leaf_nodes, internal_nodes, parents);

  BOOST_TEST(parents(0) == -1);

  ArborX::Node *root = internal_nodes.data();

  std::ostringstream sol;
  traverseRecursive(root, sol);
  std::cout << "sol=" << sol.str() << "\n";

  BOOST_TEST(sol.str().compare(ref.str()) == 0);
}
