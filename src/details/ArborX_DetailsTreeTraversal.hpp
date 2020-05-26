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
#ifndef ARBORX_DETAILS_TREE_TRAVERSAL_HPP
#define ARBORX_DETAILS_TREE_TRAVERSAL_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsAlgorithms.hpp>
#include <ArborX_DetailsNode.hpp>
#include <ArborX_DetailsPriorityQueue.hpp>
#include <ArborX_DetailsStack.hpp>
#include <ArborX_Exception.hpp>
#include <ArborX_Macros.hpp>
#include <ArborX_Predicates.hpp>

namespace ArborX
{
namespace Details
{

template <typename BVH, typename Predicates, typename Callback, typename Tag>
struct TreeTraversal
{
};

template <typename BVH, typename Predicates, typename Callback>
struct TreeTraversal<BVH, Predicates, Callback, SpatialPredicateTag>
{
  BVH bvh_;
  Predicates predicates_;
  Callback callback_;

  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;

  template <typename ExecutionSpace>
  TreeTraversal(ExecutionSpace const &space, BVH const &bvh,
                Predicates const &predicates, Callback const &callback)
      : bvh_{bvh}
      , predicates_{predicates}
      , callback_{callback}
  {
    if (bvh_.empty())
    {
      // do nothing
    }
    else if (bvh_.size() == 1)
    {
      Kokkos::parallel_for(
          ARBORX_MARK_REGION("BVH:spatial_queries_degenerated_one_leaf_tree"),
          Kokkos::RangePolicy<ExecutionSpace, OneLeafTree>(
              space, 0, Access::size(predicates)),
          *this);
    }
    else
    {
      Kokkos::parallel_for(ARBORX_MARK_REGION("BVH:spatial_queries"),
                           Kokkos::RangePolicy<ExecutionSpace>(
                               space, 0, Access::size(predicates)),
                           *this);
    }
  }

  struct OneLeafTree
  {
  };

  KOKKOS_FUNCTION void operator()(OneLeafTree, int queryIndex) const
  {
    auto const &predicate = Access::get(predicates_, queryIndex);

    if (predicate(bvh_.getBoundingVolume(bvh_.getRoot())))
    {
      callback_(queryIndex, 0);
    }
  }

  KOKKOS_FUNCTION void operator()(int queryIndex) const
  {
    auto const &predicate = Access::get(predicates_, queryIndex);

    Node const *stack[64];
    Node const **stack_ptr = stack;
    *stack_ptr++ = nullptr;
    Node const *node = bvh_.getRoot();
    do
    {
      Node const *child_left = bvh_.getNodePtr(node->children.first);
      Node const *child_right = bvh_.getNodePtr(node->children.second);

      bool overlap_left = predicate(bvh_.getBoundingVolume(child_left));
      bool overlap_right = predicate(bvh_.getBoundingVolume(child_right));

      if (overlap_left && child_left->isLeaf())
      {
        callback_(queryIndex, child_left->getLeafPermutationIndex());
      }
      if (overlap_right && child_right->isLeaf())
      {
        callback_(queryIndex, child_right->getLeafPermutationIndex());
      }

      bool traverse_left = (overlap_left && !child_left->isLeaf());
      bool traverse_right = (overlap_right && !child_right->isLeaf());

      if (!traverse_left && !traverse_right)
      {
        node = *--stack_ptr;
      }
      else
      {
        node = traverse_left ? child_left : child_right;
        if (traverse_left && traverse_right)
          *stack_ptr++ = child_right;
      }
    } while (node != nullptr);
  }
};

// NOTE using tuple as a workaround
// Error: class template partial specialization contains a template parameter
// that cannot be deduced
template <typename BVH, typename Predicates, typename Callback>
struct TreeTraversal<BVH, Predicates, Callback, NearestPredicateTag>
{
  using MemorySpace = typename BVH::memory_space;

  BVH bvh_;
  Predicates predicates_;
  Callback callback_;

  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;

  using Buffer = Kokkos::View<Kokkos::pair<int, float> *, MemorySpace>;
  using Offset = Kokkos::View<int *, MemorySpace>;
  struct BufferProvider
  {
    Buffer buffer_;
    Offset offset_;

    KOKKOS_FUNCTION auto operator()(int i) const
    {
      auto const *offset_ptr = &offset_(i);
      return Kokkos::subview(buffer_,
                             Kokkos::make_pair(*offset_ptr, *(offset_ptr + 1)));
    }
  };

  BufferProvider buffer_;

  template <typename ExecutionSpace>
  void allocateBuffer(ExecutionSpace const &space)
  {
    using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
    auto const n_queries = Access::size(predicates_);

    Offset offset(Kokkos::ViewAllocateWithoutInitializing("offset"),
                  n_queries + 1);
    // NOTE workaround to avoid implicit capture of *this
    auto const &predicates = predicates_;
    Kokkos::parallel_for(
        ARBORX_MARK_REGION("scan_queries_for_numbers_of_nearest_neighbors"),
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
        KOKKOS_LAMBDA(int i) { offset(i) = getK(Access::get(predicates, i)); });
    exclusivePrefixSum(space, offset);
    int const buffer_size = lastElement(offset);
    // Allocate buffer over which to perform heap operations in
    // TreeTraversal::nearestQuery() to store nearest leaf nodes found so far.
    // It is not possible to anticipate how much memory to allocate since the
    // number of nearest neighbors k is only known at runtime.

    Buffer buffer(Kokkos::ViewAllocateWithoutInitializing("buffer"),
                  buffer_size);
    buffer_ = BufferProvider{buffer, offset};
  }

  template <typename ExecutionSpace>
  TreeTraversal(ExecutionSpace const &space, BVH const &bvh,
                Predicates const &predicates, Callback const &callback)
      : bvh_{bvh}
      , predicates_{predicates}
      , callback_{callback}
  {
    if (bvh_.empty())
    {
      // do nothing
    }
    else if (bvh_.size() == 1)
    {
      Kokkos::parallel_for(
          ARBORX_MARK_REGION("BVH:nearest_queries_degenerated_one_leaf_tree"),
          Kokkos::RangePolicy<ExecutionSpace, OneLeafTree>(
              space, 0, Access::size(predicates)),
          *this);
    }
    else
    {
      allocateBuffer(space);

      Kokkos::parallel_for(ARBORX_MARK_REGION("BVH:nearest_queries"),
                           Kokkos::RangePolicy<ExecutionSpace>(
                               space, 0, Access::size(predicates)),
                           *this);
    }
  }

  struct OneLeafTree
  {
  };

  KOKKOS_FUNCTION int operator()(OneLeafTree, int queryIndex) const
  {
    auto const &predicate = Access::get(predicates_, queryIndex);
    auto const k = getK(predicate);
    auto const distance = [geometry = getGeometry(predicate),
                           bvh = bvh_](Node const *node) {
      return Details::distance(geometry, bvh.getBoundingVolume(node));
    };

    // NOTE thinking about making this a precondition
    if (k < 1)
      return 0;

    callback_(queryIndex, 0, distance(bvh_.getRoot()));
    return 1;
  }

  KOKKOS_FUNCTION int operator()(int queryIndex) const
  {
    auto const &predicate = Access::get(predicates_, queryIndex);
    auto const k = getK(predicate);
    auto const distance = [geometry = getGeometry(predicate),
                           bvh = bvh_](Node const *node) {
      return Details::distance(geometry, bvh.getBoundingVolume(node));
    };
    auto const buffer = buffer_(queryIndex);

    // NOTE thinking about making this a precondition
    if (k < 1)
      return 0;

    // Nodes with a distance that exceed that radius can safely be
    // discarded. Initialize the radius to infinity and tighten it once k
    // neighbors have been found.
    auto radius = KokkosExt::ArithmeticTraits::infinity<float>::value;

    using PairIndexDistance = Kokkos::pair<int, float>;
    static_assert(
        std::is_same<typename decltype(buffer)::value_type,
                     PairIndexDistance>::value,
        "Type of the elements stored in the buffer passed as argument to "
        "TreeTraversal::nearestQuery is not right");
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
    assert(k == (int)buffer.size());
    PriorityQueue<PairIndexDistance, CompareDistance,
                  UnmanagedStaticVector<PairIndexDistance>>
        heap(UnmanagedStaticVector<PairIndexDistance>(buffer.data(),
                                                      buffer.size()));

    using PairNodePtrDistance = Kokkos::pair<Node const *, float>;
    PairNodePtrDistance stack[64];
    auto *stack_ptr = stack;
    *stack_ptr++ = {nullptr, 0.f};

    Node const *node = bvh_.getRoot();
    float node_distance = 0.f;
    do
    {
      if (node_distance < radius)
      {
        if (node->isLeaf())
        {
          int const leaf_index = node->getLeafPermutationIndex();
          auto const leaf_distance = node_distance;
          if ((int)heap.size() < k)
          {
            // Insert leaf node and update radius if it was the kth
            // one.
            heap.push(Kokkos::make_pair(leaf_index, leaf_distance));
            if ((int)heap.size() == k)
              radius = heap.top().second;
          }
          else
          {
            // Replace top element in the heap and update radius.
            heap.popPush(Kokkos::make_pair(leaf_index, leaf_distance));
            radius = heap.top().second;
          }
          auto const &pair = *--stack_ptr;
          node = pair.first;
          node_distance = pair.second;
        }
        else
        {
          // Insert children into the stack and make sure that the
          // closest one ends on top.
          Node const *left_child = bvh_.getNodePtr(node->children.first);
          Node const *right_child = bvh_.getNodePtr(node->children.second);
          auto const left_child_distance = distance(left_child);
          auto const right_child_distance = distance(right_child);
          if (left_child_distance < right_child_distance)
          {
            // NOTE not really sure why but it performed better with
            // the conditional insertion on the device and without
            // it on the host (~5% improvement for both)
#if defined(__CUDA_ARCH__)
            if (right_child_distance < radius)
#endif
              *stack_ptr++ = {right_child, right_child_distance};
            node = left_child;
            node_distance = left_child_distance;
          }
          else
          {
#if defined(__CUDA_ARCH__)
            if (left_child_distance < radius)
#endif
              *stack_ptr++ = {left_child, left_child_distance};
            node = right_child;
            node_distance = right_child_distance;
          }
        }
      }
      else
      {
        auto const &pair = *--stack_ptr;
        node = pair.first;
        node_distance = pair.second;
      }
    } while (node != nullptr);

    // Sort the leaf nodes and output the results.
    // NOTE: Do not try this at home.  Messing with the underlying container
    // invalidates the state of the PriorityQueue.
    sortHeap(heap.data(), heap.data() + heap.size(), heap.valueComp());
    for (decltype(heap.size()) i = 0; i < heap.size(); ++i)
    {
      int const leaf_index = (heap.data() + i)->first;
      auto const leaf_distance = (heap.data() + i)->second;
      callback_(queryIndex, leaf_index, leaf_distance);
    }
    return heap.size();
  }

  struct Deprecated
  {
  };

  // This is the older version of the nearest traversal that uses a priority
  // queue and that was deemed less performant than the newer version with a
  // stack.
  KOKKOS_FUNCTION int operator()(Deprecated, int queryIndex) const
  {
    auto const &predicate = Access::get(predicates_, queryIndex);
    auto const k = getK(predicate);
    auto const distance = [geometry = getGeometry(predicate),
                           bvh = bvh_](Node const *node) {
      return Details::distance(geometry, bvh.getBoundingVolume(node));
    };

    // NOTE thinking about making this a precondition
    if (k < 1)
      return 0;

    using PairNodePtrDistance = Kokkos::pair<Node const *, float>;
    struct CompareDistance
    {
      KOKKOS_INLINE_FUNCTION bool
      operator()(PairNodePtrDistance const &lhs,
                 PairNodePtrDistance const &rhs) const
      {
        // Reverse order (larger distance means lower priority)
        return lhs.second > rhs.second;
      }
    };
    PriorityQueue<PairNodePtrDistance, CompareDistance,
                  StaticVector<PairNodePtrDistance, 256>>
        queue;

    // Do not bother computing the distance to the root node since it is
    // immediately popped out of the stack and processed.
    queue.emplace(bvh_.getRoot(), 0.);
    decltype(k) count = 0;

    while (!queue.empty() && count < k)
    {
      // Get the node that is on top of the priority list (i.e. the
      // one that is closest to the query point)
      Node const *node = queue.top().first;
      auto const node_distance = queue.top().second;

      if (node->isLeaf())
      {
        queue.pop();
        callback_(queryIndex, node->getLeafPermutationIndex(), node_distance);
        ++count;
      }
      else
      {
        // Insert children into the priority queue
        Node const *left_child = bvh_.getNodePtr(node->children.first);
        Node const *right_child = bvh_.getNodePtr(node->children.second);
        auto const left_child_distance = distance(left_child);
        auto const right_child_distance = distance(right_child);
        queue.popPush(left_child, left_child_distance);
        queue.emplace(right_child, right_child_distance);
      }
    }
    return count;
  }
};

template <typename BVH>
using DeprecatedTreeTraversal = TreeTraversal<BVH, void, void, void>;

template <typename BVH>
struct TreeTraversal<BVH, void, void, void>
{
  // WARNING deprecated will be removed soon
  // still used in TreeVisualization
  template <typename Distance, typename Insert, typename Buffer>
  KOKKOS_FUNCTION static int
  nearestQuery(BVH const &bvh, Distance const &distance, std::size_t k,
               Insert const &insert, Buffer const &buffer)
  {
    if (bvh.empty() || k < 1)
      return 0;

    if (bvh.size() == 1)
    {
      insert(0, distance(bvh.getRoot()));
      return 1;
    }

    // Nodes with a distance that exceed that radius can safely be
    // discarded. Initialize the radius to infinity and tighten it once k
    // neighbors have been found.
    auto radius = KokkosExt::ArithmeticTraits::infinity<float>::value;

    using PairIndexDistance = Kokkos::pair<int, float>;
    static_assert(
        std::is_same<typename Buffer::value_type, PairIndexDistance>::value,
        "Type of the elements stored in the buffer passed as argument to "
        "TreeTraversal::nearestQuery is not right");
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
    assert(k == buffer.size());
    PriorityQueue<PairIndexDistance, CompareDistance,
                  UnmanagedStaticVector<PairIndexDistance>>
        heap(UnmanagedStaticVector<PairIndexDistance>(buffer.data(),
                                                      buffer.size()));

    using PairNodePtrDistance = Kokkos::pair<Node const *, float>;
    Stack<PairNodePtrDistance> stack;
    // Do not bother computing the distance to the root node since it is
    // immediately popped out of the stack and processed.
    stack.emplace(bvh.getRoot(), 0.);

    while (!stack.empty())
    {
      Node const *node = stack.top().first;
      auto const node_distance = stack.top().second;
      stack.pop();

      if (node_distance < radius)
      {
        if (node->isLeaf())
        {
          int const leaf_index = node->getLeafPermutationIndex();
          auto const leaf_distance = node_distance;
          if (heap.size() < k)
          {
            // Insert leaf node and update radius if it was the kth
            // one.
            heap.push(Kokkos::make_pair(leaf_index, leaf_distance));
            if (heap.size() == k)
              radius = heap.top().second;
          }
          else
          {
            // Replace top element in the heap and update radius.
            heap.popPush(Kokkos::make_pair(leaf_index, leaf_distance));
            radius = heap.top().second;
          }
        }
        else
        {
          // Insert children into the stack and make sure that the
          // closest one ends on top.
          Node const *left_child = bvh.getNodePtr(node->children.first);
          Node const *right_child = bvh.getNodePtr(node->children.second);
          auto const left_child_distance = distance(left_child);
          auto const right_child_distance = distance(right_child);
          if (left_child_distance < right_child_distance)
          {
            // NOTE not really sure why but it performed better with
            // the conditional insertion on the device and without
            // it on the host (~5% improvement for both)
#if defined(__CUDA_ARCH__)
            if (right_child_distance < radius)
#endif
              stack.emplace(right_child, right_child_distance);
            stack.emplace(left_child, left_child_distance);
          }
          else
          {
#if defined(__CUDA_ARCH__)
            if (left_child_distance < radius)
#endif
              stack.emplace(left_child, left_child_distance);
            stack.emplace(right_child, right_child_distance);
          }
        }
      }
    }
    // Sort the leaf nodes and output the results.
    // NOTE: Do not try this at home.  Messing with the underlying container
    // invalidates the state of the PriorityQueue.
    sortHeap(heap.data(), heap.data() + heap.size(), heap.valueComp());
    for (decltype(heap.size()) i = 0; i < heap.size(); ++i)
    {
      int const leaf_index = (heap.data() + i)->first;
      auto const leaf_distance = (heap.data() + i)->second;
      insert(leaf_index, leaf_distance);
    }
    return heap.size();
  }
};

template <typename ExecutionSpace, typename BVH, typename Predicates,
          typename Callback>
void traverse(ExecutionSpace const &space, BVH const &bvh,
              Predicates const &predicates, Callback const &callback)
{
  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
  using Tag = typename Traits::Helper<Access>::tag;
  TreeTraversal<BVH, Predicates, Callback, Tag>(space, bvh, predicates,
                                                callback);
}

} // namespace Details
} // namespace ArborX

#endif
