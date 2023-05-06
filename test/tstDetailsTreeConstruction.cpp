/****************************************************************************
 * Copyright (c) 2017-2023 by the ArborX authors                            *
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
#include <ArborX_DetailsMortonCode.hpp> // expandBits, morton32
#include <ArborX_DetailsNode.hpp>       // ROPE SENTINEL
#include <ArborX_DetailsSortUtils.hpp>  // sortObjects
#include <ArborX_DetailsTreeConstruction.hpp>

#include <boost/test/unit_test.hpp>

#include <array>
#include <bitset>
#include <functional>
#include <limits>
#include <sstream>
#include <vector>

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE_TEMPLATE(assign_morton_codes, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  // N is the number of Morton grid cells in each dimension for 64-bit Morton
  // codes.
  constexpr unsigned long long N = 1 << 21;
  std::vector<ArborX::Point> points = {{{0.0, 0.0, 0.0}},
                                       {{0.25, 0.75, 0.25}},
                                       {{0.75, 0.25, 0.25}},
                                       {{0.75, 0.75, 0.25}},
                                       {{1.33, 2.33, 3.33}},
                                       {{1.66, 2.66, 3.66}},
                                       {{(float)N, (float)N, (float)N}}};
  int const n = points.size();
  // lower left front corner of the octant the points fall in
  std::vector<std::array<unsigned long long, 3>> anchors = {
      {{0, 0, 0}},
      {{0, 0, 0}},
      {{0, 0, 0}},
      {{0, 0, 0}},
      {{1, 2, 3}},
      {{1, 2, 3}},
      {{N - 1, N - 1, N - 1}}};
  auto fun = [](std::array<unsigned long long, 3> const &anchor) {
    using ArborX::Details::expandBitsBy;
    auto i = std::get<0>(anchor);
    auto j = std::get<1>(anchor);
    auto k = std::get<2>(anchor);
    return 4 * expandBitsBy<2>(i) + 2 * expandBitsBy<2>(j) + expandBitsBy<2>(k);
  };
  std::vector<unsigned long long> ref(
      n, std::numeric_limits<unsigned long long>::max());
  for (int i = 0; i < n; ++i)
    ref[i] = fun(anchors[i]);
  // using points rather than boxes for convenience here but still have to
  // build the axis-aligned bounding boxes around them
  Kokkos::View<ArborX::Box *, DeviceType> boxes("boxes", n);
  auto boxes_host = Kokkos::create_mirror_view(boxes);
  for (int i = 0; i < n; ++i)
    ArborX::Details::expand(boxes_host(i), points[i]);
  Kokkos::deep_copy(boxes, boxes_host);

  typename DeviceType::execution_space space{};
  ArborX::Box scene_host;
  ArborX::Details::TreeConstruction::calculateBoundingBoxOfTheScene(
      space, boxes, scene_host);

  BOOST_TEST(ArborX::Details::equals(
      scene_host, {{{0.0, 0.0, 0.0}}, {{(float)N, (float)N, (float)N}}}));

  Kokkos::View<unsigned long long *, DeviceType> morton_codes("morton_codes",
                                                              n);
  ArborX::Details::TreeConstruction::projectOntoSpaceFillingCurve(
      space, boxes, ArborX::Experimental::Morton64(), scene_host, morton_codes);
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
  {}

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

  std::vector<size_t> ref = {3, 2, 1, 0};
  // sort Morton codes and object ids
  auto ids = ArborX::Details::sortObjects(ExecutionSpace{}, k);

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

template <typename Primitives, typename MortonCodes, typename LeafNodes,
          typename InternalNodes>
void generateHierarchy(Primitives primitives, MortonCodes sorted_morton_codes,
                       LeafNodes &leaf_nodes, InternalNodes &internal_nodes)
{
  using ArborX::Details::makeLeafNode;
  using DeviceType = typename MortonCodes::device_type;

  int const n = sorted_morton_codes.extent(0);

  Kokkos::realloc(leaf_nodes, n);
  Kokkos::realloc(internal_nodes, n - 1);

  typename DeviceType::execution_space space{};

  Kokkos::View<unsigned int *, DeviceType> permutation_indices(
      "Testing::indices", n);
  ArborX::iota(space, permutation_indices);

  ArborX::Details::TreeConstruction::generateHierarchy(
      space, primitives, permutation_indices, sorted_morton_codes, leaf_nodes,
      internal_nodes);
}

template <typename LeafNodes, typename InternalNodes>
void traverse(LeafNodes leaf_nodes, InternalNodes internal_nodes, int root,
              std::ostringstream &sol)
{
  int n = leaf_nodes.extent(0);

  using ArborX::Details::ROPE_SENTINEL;
  std::function<void(int, std::ostream &)> traverseRopes;
  traverseRopes = [&leaf_nodes, &internal_nodes, n,
                   &traverseRopes](int node, std::ostream &os) {
    if (node < n)
    {
      os << "L" << node;
      int rope = leaf_nodes(node).rope;
      if (rope != ROPE_SENTINEL)
        traverseRopes(rope, os);
    }
    else
    {
      node = node - n;
      os << "I" << node;
      traverseRopes(internal_nodes(node).left_child, os);
    }
  };

  traverseRopes(root, sol);
}

namespace Test
{
struct FakePrimitive
{};
struct FakeBoundingVolume
{};
KOKKOS_FUNCTION void expand(FakeBoundingVolume, FakeBoundingVolume) {}
KOKKOS_FUNCTION void expand(FakeBoundingVolume, FakePrimitive) {}
} // namespace Test

BOOST_AUTO_TEST_CASE_TEMPLATE(example_tree_construction, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  if (!Kokkos::SpaceAccessibility<
          Kokkos::HostSpace, typename DeviceType::memory_space>::accessible)
    return;

  // This is the example from the articles by Karras.
  // See
  // https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
  int const n = 8;
  Kokkos::View<unsigned long long *, DeviceType> sorted_morton_codes(
      "sorted_morton_codes", n);
  std::vector<std::string> s{
      "00001", "00010", "00100", "00101", "10011", "11000", "11001", "11110",
  };
  for (int i = 0; i < n; ++i)
  {
    std::bitset<6> b(s[i]);
    BOOST_TEST_MESSAGE(b << "  " << b.to_ullong());
    sorted_morton_codes(i) = b.to_ullong();
  }

  Kokkos::View<Test::FakePrimitive *, DeviceType> primitives(
      "Testing::primitives", n);

  // Reference solution for the depth first search
  std::ostringstream ref;
  // clang-format off
  ref << "I0" << "I3" << "I1" << "L0" << "L1" << "I2" << "L2" << "L3"
      << "I4" << "L4" << "I5" << "I6" << "L5" << "L6" << "L7";
  // clang-format on
  BOOST_TEST_MESSAGE("ref = " << ref.str());

  using LeafNode = ArborX::Details::LeafNode<
      ArborX::Details::PairIndexVolume<Test::FakeBoundingVolume>>;
  using InternalNode = ArborX::Details::InternalNode<Test::FakeBoundingVolume>;

  Kokkos::View<LeafNode *, DeviceType> leaf_nodes("Testing::leaf_nodes", 0);
  Kokkos::View<InternalNode *, DeviceType> internal_nodes(
      "Testing::internal_nodes", 0);
  generateHierarchy(primitives, sorted_morton_codes, leaf_nodes,
                    internal_nodes);

  int const root = n;

  std::ostringstream sol;
  traverse(leaf_nodes, internal_nodes, root, sol);

  BOOST_TEST_MESSAGE("sol(node_with_left_child_and_rope) = " << sol.str());

  BOOST_TEST(sol.str() == ref.str());
}
