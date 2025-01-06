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
#ifndef ARBORX_TREE_TRAVERSAL_HPP
#define ARBORX_TREE_TRAVERSAL_HPP

#include <detail/ArborX_HappyTreeFriends.hpp>
#include <detail/ArborX_NearestBufferProvider.hpp>
#include <detail/ArborX_Node.hpp> // ROPE_SENTINEL
#include <detail/ArborX_Predicates.hpp>
#include <kokkos_ext/ArborX_KokkosExtArithmeticTraits.hpp>
#include <kokkos_ext/ArborX_KokkosExtStdAlgorithms.hpp>
#include <kokkos_ext/ArborX_KokkosExtViewHelpers.hpp>
#include <misc/ArborX_Exception.hpp>
#include <misc/ArborX_PriorityQueue.hpp>
#include <misc/ArborX_Stack.hpp>

namespace ArborX
{
namespace Details
{

template <typename BVH, typename Predicates, typename Callback, typename Tag>
struct TreeTraversal
{};

template <typename BVH, typename Predicates, typename Callback>
struct TreeTraversal<BVH, Predicates, Callback, SpatialPredicateTag>
{
  BVH _bvh;
  Predicates _predicates;
  Callback _callback;

  template <typename ExecutionSpace>
  TreeTraversal(ExecutionSpace const &space, BVH const &bvh,
                Predicates const &predicates, Callback const &callback)
      : _bvh{bvh}
      , _predicates{predicates}
      , _callback{callback}
  {
    if (_bvh.empty())
    {
      // do nothing
    }
    else if (_bvh.size() == 1)
    {
      Kokkos::parallel_for(
          "ArborX::TreeTraversal::spatial::degenerated_one_leaf_tree",
          Kokkos::RangePolicy<ExecutionSpace, OneLeafTree>(space, 0,
                                                           predicates.size()),
          *this);
    }
    else
    {
      Kokkos::parallel_for("ArborX::TreeTraversal::spatial",
                           Kokkos::RangePolicy<ExecutionSpace, FullTree>(
                               space, 0, predicates.size()),
                           *this);
    }
  }

  KOKKOS_FUNCTION TreeTraversal(BVH const &bvh, Callback const &callback)
      : _bvh{bvh}
      , _callback{callback}
  {}

  struct OneLeafTree
  {};

  struct FullTree
  {};

  KOKKOS_FUNCTION void operator()(OneLeafTree, int queryIndex) const
  {
    auto const &predicate = _predicates(queryIndex);
    auto const root = 0;
    auto const &root_bounding_volume =
        HappyTreeFriends::getIndexable(_bvh, root);
    if (predicate(root_bounding_volume))
    {
      _callback(predicate, HappyTreeFriends::getValue(_bvh, 0));
    }
  }

  KOKKOS_FUNCTION void operator()(FullTree, int queryIndex) const
  {
    operator()(_predicates(queryIndex));
  }

  template <typename Predicate>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate) const
  {
    int node = HappyTreeFriends::getRoot(_bvh); // start with root
    do
    {
      if (HappyTreeFriends::isLeaf(_bvh, node))
      {
        if (predicate(HappyTreeFriends::getIndexable(_bvh, node)) &&
            invoke_callback_and_check_early_exit(
                _callback, predicate, HappyTreeFriends::getValue(_bvh, node)))
          return;
        node = HappyTreeFriends::getRope(_bvh, node);
      }
      else
      {
        node =
            (predicate(HappyTreeFriends::getInternalBoundingVolume(_bvh, node))
                 ? HappyTreeFriends::getLeftChild(_bvh, node)
                 : HappyTreeFriends::getRope(_bvh, node));
      }
    } while (node != ROPE_SENTINEL);
  }
};

template <typename BVH, typename Predicates, typename Callback>
struct TreeTraversal<BVH, Predicates, Callback, NearestPredicateTag>
{
  using MemorySpace = typename BVH::memory_space;

  BVH _bvh;
  Predicates _predicates;
  Callback _callback;

  using Coordinate = decltype(std::declval<Predicates>()(0).distance(
      HappyTreeFriends::getIndexable(_bvh, 0)));

  NearestBufferProvider<MemorySpace, Coordinate> _buffer;

  template <typename ExecutionSpace>
  TreeTraversal(ExecutionSpace const &space, BVH const &bvh,
                Predicates const &predicates, Callback const &callback)
      : _bvh{bvh}
      , _predicates{predicates}
      , _callback{callback}
  {
    if (_bvh.empty())
    {
      // do nothing
    }
    else if (_bvh.size() == 1)
    {
      Kokkos::parallel_for(
          "ArborX::TreeTraversal::nearest::degenerated_one_leaf_tree",
          Kokkos::RangePolicy<ExecutionSpace, OneLeafTree>(space, 0,
                                                           predicates.size()),
          *this);
    }
    else
    {
      _buffer.allocateBuffer(space, predicates);

      Kokkos::parallel_for("ArborX::TreeTraversal::nearest",
                           Kokkos::RangePolicy(space, 0, predicates.size()),
                           *this);
    }
  }

  struct OneLeafTree
  {};

  KOKKOS_FUNCTION void operator()(OneLeafTree, int queryIndex) const
  {
    auto const &predicate = _predicates(queryIndex);
    auto const k = getK(predicate);

    // NOTE thinking about making this a precondition
    if (k < 1)
      return;

    _callback(predicate, HappyTreeFriends::getValue(_bvh, 0));
  }

  KOKKOS_FUNCTION void operator()(int queryIndex) const
  {
    auto const &predicate = _predicates(queryIndex);
    auto const k = getK(predicate);
    auto const buffer = _buffer(queryIndex);

    // NOTE thinking about making this a precondition
    if (k < 1)
      return;

    using PairIndexDistance = typename decltype(_buffer)::PairIndexDistance;
    struct CompareDistance
    {
      KOKKOS_INLINE_FUNCTION bool operator()(PairIndexDistance const &lhs,
                                             PairIndexDistance const &rhs) const
      {
        return lhs.second < rhs.second;
      }
    };
    // Use a priority queue for convenience to store the results and
    // preserve the heap structure internally at all time.  There is no
    // memory allocation, elements are stored in the buffer passed as an
    // argument. The farthest leaf node is on top.
    KOKKOS_ASSERT(k == (int)buffer.size());
    PriorityQueue<PairIndexDistance, CompareDistance,
                  UnmanagedStaticVector<PairIndexDistance>>
        heap(UnmanagedStaticVector<PairIndexDistance>(buffer.data(),
                                                      buffer.size()));

    auto &bvh = _bvh;
    auto const distance = [&predicate, &bvh](int j) {
      return HappyTreeFriends::isLeaf(bvh, j)
                 ? predicate.distance(HappyTreeFriends::getIndexable(bvh, j))
                 : predicate.distance(
                       HappyTreeFriends::getInternalBoundingVolume(bvh, j));
    };

    constexpr int SENTINEL = -1;
    int stack[64];
    auto *stack_ptr = stack;
    *stack_ptr++ = SENTINEL;
#if !defined(__CUDA_ARCH__)
    Coordinate stack_distance[64];
    auto *stack_distance_ptr = stack_distance;
    *stack_distance_ptr++ = 0.f;
#endif

    int node = HappyTreeFriends::getRoot(_bvh);
    int left_child;
    int right_child;

    Coordinate distance_left = 0;
    Coordinate distance_right = 0;
    Coordinate distance_node = 0;

    // Nodes with a distance that exceed that radius can safely be
    // discarded. Initialize the radius to infinity and tighten it once k
    // neighbors have been found.
    auto radius = KokkosExt::ArithmeticTraits::infinity<Coordinate>::value;

    do
    {
      bool traverse_left = false;
      bool traverse_right = false;

      if (distance_node < radius)
      {
        // Insert children into the stack and make sure that the
        // closest one ends on top.
        left_child = HappyTreeFriends::getLeftChild(_bvh, node);
        right_child = HappyTreeFriends::getRightChild(_bvh, node);

        distance_left = distance(left_child);
        distance_right = distance(right_child);

        if (distance_left < radius)
        {
          if (HappyTreeFriends::isLeaf(_bvh, left_child))
          {
            auto leaf_pair = Kokkos::make_pair(left_child, distance_left);
            if ((int)heap.size() < k)
              heap.push(leaf_pair);
            else
              heap.popPush(leaf_pair);
            if ((int)heap.size() == k)
              radius = heap.top().second;
          }
          else
          {
            traverse_left = true;
          }
        }

        // Note: radius may have been already updated here from the left child
        if (distance_right < radius)
        {
          if (HappyTreeFriends::isLeaf(_bvh, right_child))
          {
            auto leaf_pair = Kokkos::make_pair(right_child, distance_right);
            if ((int)heap.size() < k)
              heap.push(leaf_pair);
            else
              heap.popPush(leaf_pair);
            if ((int)heap.size() == k)
              radius = heap.top().second;
          }
          else
          {
            traverse_right = true;
          }
        }
      }

      if (!traverse_left && !traverse_right)
      {
        node = *--stack_ptr;
#if defined(__CUDA_ARCH__)
        if (node != SENTINEL)
        {
          // This is a theoretically unnecessary duplication of distance
          // calculation for stack nodes. However, for Cuda it's better than
          // putting the distances in stack.
          distance_node = distance(node);
        }
#else
        distance_node = *--stack_distance_ptr;
#endif
      }
      else
      {
        node = (traverse_left &&
                (distance_left <= distance_right || !traverse_right))
                   ? left_child
                   : right_child;
        distance_node = (node == left_child ? distance_left : distance_right);
        if (traverse_left && traverse_right)
        {
          *stack_ptr++ = (node == left_child ? right_child : left_child);
#if !defined(__CUDA_ARCH__)
          *stack_distance_ptr++ =
              (node == left_child ? distance_right : distance_left);
#endif
        }
      }
    } while (node != SENTINEL);

    // Sort the leaf nodes and output the results.
    // NOTE: Do not try this at home.  Messing with the underlying container
    // invalidates the state of the PriorityQueue.
    sortHeap(heap.data(), heap.data() + heap.size(), heap.valueComp());
    for (decltype(heap.size()) i = 0; i < heap.size(); ++i)
    {
      _callback(predicate,
                HappyTreeFriends::getValue(_bvh, (heap.data() + i)->first));
    }
  }
};

template <class BVH, class Predicates, class Callback>
struct TreeTraversal<BVH, Predicates, Callback, OrderedSpatialPredicateTag>
{
  BVH _bvh;
  Predicates _predicates;
  Callback _callback;

  template <class ExecutionSpace>
  TreeTraversal(ExecutionSpace const &space, BVH const &bvh,
                Predicates const &predicates, Callback const &callback)
      : _bvh{bvh}
      , _predicates{predicates}
      , _callback{callback}
  {
    if (_bvh.empty())
    {
      // do nothing
    }
    else if (_bvh.size() == 1)
    {
      Kokkos::parallel_for(
          "ArborX::Experimental::TreeTraversal::OrderedSpatialPredicate"
          "degenerated_one_leaf_tree",
          Kokkos::RangePolicy<ExecutionSpace, OneLeafTree>(space, 0,
                                                           predicates.size()),
          *this);
    }
    else
    {
      Kokkos::parallel_for(
          "ArborX::Experimental::TreeTraversal::OrderedSpatialPredicate",
          Kokkos::RangePolicy<ExecutionSpace, FullTree>(space, 0,
                                                        predicates.size()),
          *this);
    }
  }

  KOKKOS_FUNCTION TreeTraversal(BVH const &bvh, Callback const &callback)
      : _bvh{bvh}
      , _callback{callback}
  {}

  struct OneLeafTree
  {};

  struct FullTree
  {};

  KOKKOS_FUNCTION void operator()(OneLeafTree, int queryIndex) const
  {
    auto const &predicate = _predicates(queryIndex);
    auto const root = 0;
    auto const &root_bounding_volume =
        HappyTreeFriends::getIndexable(_bvh, root);
    using distance_type =
        decltype(distance(getGeometry(predicate), root_bounding_volume));
    constexpr auto inf =
        KokkosExt::ArithmeticTraits::infinity<distance_type>::value;
    if (distance(getGeometry(predicate), root_bounding_volume) != inf)
    {
      _callback(predicate, HappyTreeFriends::getValue(_bvh, 0));
    }
  }

  KOKKOS_FUNCTION void operator()(FullTree, int queryIndex) const
  {
    operator()(_predicates(queryIndex));
  }

  template <typename Predicate>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate) const
  {
    using ArborX::Details::HappyTreeFriends;

    using distance_type = decltype(predicate.distance(
        HappyTreeFriends::getInternalBoundingVolume(_bvh, 0)));
    using PairIndexDistance = Kokkos::pair<int, distance_type>;
    struct CompareDistance
    {
      KOKKOS_FUNCTION bool operator()(PairIndexDistance const &lhs,
                                      PairIndexDistance const &rhs) const
      {
        return lhs.second > rhs.second;
      }
    };

    constexpr int buffer_size = 64;
    PairIndexDistance buffer[buffer_size];
    PriorityQueue<PairIndexDistance, CompareDistance,
                  UnmanagedStaticVector<PairIndexDistance>>
        heap(UnmanagedStaticVector<PairIndexDistance>(buffer, buffer_size));

    constexpr auto inf =
        KokkosExt::ArithmeticTraits::infinity<distance_type>::value;

    auto &bvh = _bvh;
    auto const distance = [&predicate, &bvh](int j) {
      return HappyTreeFriends::isLeaf(bvh, j)
                 ? predicate.distance(HappyTreeFriends::getIndexable(bvh, j))
                 : predicate.distance(
                       HappyTreeFriends::getInternalBoundingVolume(bvh, j));
    };

    int node = HappyTreeFriends::getRoot(_bvh);
    int left_child;
    int right_child;

    while (true)
    {
      if (HappyTreeFriends::isLeaf(_bvh, node))
      {
        if (invoke_callback_and_check_early_exit(
                _callback, predicate, HappyTreeFriends::getValue(_bvh, node)))
          return;

        if (heap.empty())
          return;

        node = heap.top().first;
        heap.pop();
      }
      else
      {
        left_child = HappyTreeFriends::getLeftChild(_bvh, node);
        right_child = HappyTreeFriends::getRightChild(_bvh, node);

        auto const distance_left = distance(left_child);
        auto const left_pair = Kokkos::make_pair(left_child, distance_left);

        auto const distance_right = distance(right_child);
        auto const right_pair = Kokkos::make_pair(right_child, distance_right);

        auto const &closer_pair =
            distance_left < distance_right ? left_pair : right_pair;
        auto const &further_pair =
            distance_left < distance_right ? right_pair : left_pair;

        if (!heap.empty() && heap.top().second < closer_pair.second)
        {
          node = heap.top().first;
          heap.pop();
          if (closer_pair.second < inf)
            heap.push(closer_pair);
        }
        else
          node = closer_pair.first;
        if (further_pair.second < inf)
          heap.push(further_pair);
      }
    }
  }
};

template <typename ExecutionSpace, typename BVH, typename Predicates,
          typename Callback>
void traverse(ExecutionSpace const &space, BVH const &bvh,
              Predicates const &predicates, Callback const &callback)
{
  using Tag = typename Predicates::value_type::Tag;
  TreeTraversal<BVH, Predicates, Callback, Tag>(space, bvh, predicates,
                                                callback);
}

} // namespace Details
} // namespace ArborX

#endif
