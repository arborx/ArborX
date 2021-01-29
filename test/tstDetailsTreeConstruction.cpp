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
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_DetailsMortonCode.hpp> // expandBits, morton3D
#include <ArborX_DetailsNode.hpp>       // ROPE SENTINEL
#include <ArborX_DetailsSortUtils.hpp>  // sortObjects
#include <ArborX_DetailsTreeConstruction.hpp>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <array>
#include <bitset>
#include <functional>
#include <limits>
#include <sstream>
#include <vector>

#define BOOST_TEST_MODULE DetailsTreeConstruction

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE_TEMPLATE(assign_morton_codes, DeviceType,
                              ARBORX_DEVICE_TYPES)
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
    using ArborX::Details::expandBits;
    unsigned int i = std::get<0>(anchor);
    unsigned int j = std::get<1>(anchor);
    unsigned int k = std::get<2>(anchor);
    return 4 * expandBits(i) + 2 * expandBits(j) + expandBits(k);
  };
  std::vector<unsigned int> ref(n, std::numeric_limits<unsigned int>::max());
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
      scene_host, {{{0., 0., 0.}}, {{1024., 1024., 1024.}}}));

  Kokkos::View<unsigned int *, DeviceType> morton_codes("morton_codes", n);
  ArborX::Details::TreeConstruction::assignMortonCodes(
      space, boxes, morton_codes, scene_host);
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

template <typename Node, typename LeafNodes, typename InternalNodes>
std::enable_if_t<
    std::is_same<typename Node::Tag, ArborX::Details::NodeWithTwoChildrenTag>{}>
traverse(LeafNodes leaf_nodes, InternalNodes internal_nodes, Node const *root,
         std::ostringstream &sol)
{
  int n = leaf_nodes.extent(0);
  auto getNodePtr = [&leaf_nodes, &internal_nodes, &n](int i) {
    return i < n - 1 ? &internal_nodes(i) : &leaf_nodes(i - n + 1);
  };

  std::function<void(Node const *, std::ostream &)> traverseRecursive;
  traverseRecursive = [&leaf_nodes, &internal_nodes, &getNodePtr,
                       &traverseRecursive](Node const *node, std::ostream &os) {
    if (node->isLeaf())
    {
      os << "L" << node - leaf_nodes.data();
    }
    else
    {
      os << "I" << node - internal_nodes.data();
      for (Node const *child :
           {getNodePtr(node->left_child), getNodePtr(node->right_child)})
        traverseRecursive(child, os);
    }
  };

  traverseRecursive(root, sol);
}

template <typename Node, typename LeafNodes, typename InternalNodes>
std::enable_if_t<std::is_same<typename Node::Tag,
                              ArborX::Details::NodeWithLeftChildAndRopeTag>{}>
traverse(LeafNodes leaf_nodes, InternalNodes internal_nodes, Node const *root,
         std::ostringstream &sol)
{
  int n = leaf_nodes.extent(0);
  auto getNodePtr = [&leaf_nodes, &internal_nodes, &n](int i) {
    return i < n - 1 ? &internal_nodes(i) : &leaf_nodes(i - n + 1);
  };

  using ArborX::Details::ROPE_SENTINEL;

  std::function<void(Node const *, std::ostream &)> traverseRopes;
  traverseRopes = [&leaf_nodes, &internal_nodes, &getNodePtr,
                   &traverseRopes](Node const *node, std::ostream &os) {
    if (node->isLeaf())
    {
      os << "L" << node - leaf_nodes.data();
      if (node->rope != ROPE_SENTINEL)
        traverseRopes(getNodePtr(node->rope), os);
    }
    else
    {
      os << "I" << node - internal_nodes.data();
      traverseRopes(getNodePtr(node->left_child), os);
    }
  };

  traverseRopes(root, sol);
}

namespace Test
{
struct FakePrimitive
{
};
struct FakeBoundingVolume
{
};
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

  Kokkos::View<Test::FakePrimitive *, DeviceType> primitives(
      "Testing::primitives", n);

  // Reference solution for the depth first search
  std::ostringstream ref;
  // clang-format off
  ref << "I0" << "I3" << "I1" << "L0" << "L1" << "I2" << "L2" << "L3"
      << "I4" << "L4" << "I5" << "I6" << "L5" << "L6" << "L7";
  // clang-format on
  std::cout << "ref = " << ref.str() << "\n";

  {
    using Node = ArborX::Details::NodeWithTwoChildren<Test::FakeBoundingVolume>;

    Kokkos::View<Node *, DeviceType> leaf_nodes("Testing::leaf_nodes", 0);
    Kokkos::View<Node *, DeviceType> internal_nodes("Testing::internal_nodes",
                                                    0);
    generateHierarchy(primitives, sorted_morton_codes, leaf_nodes,
                      internal_nodes);

    auto const *root = internal_nodes.data();

    std::ostringstream sol;
    traverse(leaf_nodes, internal_nodes, root, sol);
    std::cout << "sol(node_with_two_children) = " << sol.str() << "\n";

    BOOST_TEST(sol.str() == ref.str());
  }
  {
    using Node =
        ArborX::Details::NodeWithLeftChildAndRope<Test::FakeBoundingVolume>;

    Kokkos::View<Node *, DeviceType> leaf_nodes("Testing::leaf_nodes", 0);
    Kokkos::View<Node *, DeviceType> internal_nodes("Testing::internal_nodes",
                                                    0);
    generateHierarchy(primitives, sorted_morton_codes, leaf_nodes,
                      internal_nodes);

    auto const *root = internal_nodes.data();

    std::ostringstream sol;
    traverse(leaf_nodes, internal_nodes, root, sol);
    std::cout << "sol(node_with_left_child_and_rope) = " << sol.str() << "\n";

    BOOST_TEST(sol.str() == ref.str());
  }
}
