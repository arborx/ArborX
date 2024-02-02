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

#ifndef ARBORX_DETAILSDBSCANVERIFICATION_HPP
#define ARBORX_DETAILSDBSCANVERIFICATION_HPP

#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_LinearBVH.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>
#include <set>
#include <stack>
#include <vector>

namespace ArborX
{
namespace Details
{

// Check that core points have nonnegative indices
template <typename ExecutionSpace, typename IndicesView, typename OffsetView,
          typename LabelsView>
bool verifyCorePointsNonnegativeIndex(ExecutionSpace const &exec_space,
                                      IndicesView /*indices*/,
                                      OffsetView offset, LabelsView labels,
                                      int core_min_size)
{
  int n = labels.size();

  int num_incorrect;
  Kokkos::parallel_reduce(
      "ArborX::DBSCAN::verify_core_points_nonnegative",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int i, int &update) {
        bool self_is_core_point = (offset(i + 1) - offset(i) >= core_min_size);
        if (self_is_core_point && labels(i) < 0)
        {
#if KOKKOS_VERSION >= 40200
          using Kokkos::printf;
#elif defined(__SYCL_DEVICE_ONLY__)
          using sycl::ext::oneapi::experimental::printf;
#endif
          printf("Core point is marked as noise: %d [%d]\n", i, labels(i));
          update++;
        }
      },
      num_incorrect);
  return (num_incorrect == 0);
}

// Check that connected core points have same cluster indices
template <typename ExecutionSpace, typename IndicesView, typename OffsetView,
          typename LabelsView>
bool verifyConnectedCorePointsShareIndex(ExecutionSpace const &exec_space,
                                         IndicesView indices, OffsetView offset,
                                         LabelsView labels, int core_min_size)
{
  int n = labels.size();

  int num_incorrect;
  Kokkos::parallel_reduce(
      "ArborX::DBSCAN::verify_connected_core_points",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int i, int &update) {
        bool self_is_core_point = (offset(i + 1) - offset(i) >= core_min_size);
        if (self_is_core_point)
        {
          for (int jj = offset(i); jj < offset(i + 1); ++jj)
          {
            int j = indices(jj);
            bool neigh_is_core_point =
                (offset(j + 1) - offset(j) >= core_min_size);

            if (neigh_is_core_point && labels(i) != labels(j))
            {
#if KOKKOS_VERSION >= 40200
              using Kokkos::printf;
#elif defined(__SYCL_DEVICE_ONLY__)
              using sycl::ext::oneapi::experimental::printf;
#endif
              printf("Connected cores do not belong to the same cluster: "
                     "%d [%d] -> %d [%d]\n",
                     i, labels(i), j, labels(j));
              update++;
            }
          }
        }
      },
      num_incorrect);
  return (num_incorrect == 0);
}

// Check that border points share index with at least one core point, and
// that noise points have index -1
template <typename ExecutionSpace, typename IndicesView, typename OffsetView,
          typename LabelsView>
bool verifyBorderAndNoisePoints(ExecutionSpace const &exec_space,
                                IndicesView indices, OffsetView offset,
                                LabelsView labels, int core_min_size)
{
  int n = labels.size();

  int num_incorrect;
  Kokkos::parallel_reduce(
      "ArborX::DBSCAN::verify_connected_border_points",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int i, int &update) {
        bool self_is_core_point = (offset(i + 1) - offset(i) >= core_min_size);
        if (!self_is_core_point)
        {
          bool is_border = false;
          bool have_shared_core = false;
          for (int jj = offset(i); jj < offset(i + 1); ++jj)
          {
            int j = indices(jj);
            bool neigh_is_core_point =
                (offset(j + 1) - offset(j) >= core_min_size);

            if (neigh_is_core_point)
            {
              is_border = true;
              if (labels(i) == labels(j))
              {
                have_shared_core = true;
                break;
              }
            }
          }

#if KOKKOS_VERSION >= 40200
          using Kokkos::printf;
#elif defined(__SYCL_DEVICE_ONLY__)
          using sycl::ext::oneapi::experimental::printf;
#endif

          // Border point must be connected to a core point
          if (is_border && !have_shared_core)
          {
            printf("Border point does not belong to a cluster: %d [%d]\n", i,
                   labels(i));
            update++;
          }
          // Noise points must have index -1
          if (!is_border && labels(i) != -1)
          {
            printf("Noise point does not have index -1: %d [%d]\n", i,
                   labels(i));
            update++;
          }
        }
      },
      num_incorrect);
  return (num_incorrect == 0);
}

// Check that cluster indices are unique
template <typename ExecutionSpace, typename IndicesView, typename OffsetView,
          typename LabelsView>
bool verifyClustersAreUnique(ExecutionSpace const &exec_space,
                             IndicesView indices, OffsetView offset,
                             LabelsView labels, int core_min_size)
{
  int n = labels.size();

  // FIXME we don't want to modify the labels view in this check. What we
  // want here is to create a view on the host, and deep_copy into it.
  // create_mirror_view_and_copy won't work, because it is a no-op if labels
  // is already on the host.
  decltype(Kokkos::create_mirror_view(Kokkos::HostSpace{},
                                      std::declval<LabelsView>()))
      labels_host(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                     "ArborX::DBSCAN::labels_host"),
                  labels.size());
  Kokkos::deep_copy(exec_space, labels_host, labels);
  auto offset_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offset);
  auto indices_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices);

  auto is_core_point = [&](int i) {
    return offset_host(i + 1) - offset_host(i) >= core_min_size;
  };

  // Remove all border points from consideration (noise points are already -1)
  // The idea is that this way if labels were bridged through a border
  // point, we will count them as separate labels but with a shared cluster
  // index, which will fail the unique labels check
  for (int i = 0; i < n; ++i)
  {
    if (!is_core_point(i))
    {
      for (int jj = offset_host(i); jj < offset_host(i + 1); ++jj)
      {
        int j = indices_host(jj);
        if (is_core_point(j))
        {
          // The point is a border point
          labels_host(i) = -1;
          break;
        }
      }
    }
  }

  // Record all unique cluster indices
  std::set<int> unique_cluster_indices;
  for (int i = 0; i < n; ++i)
    if (labels_host(i) != -1)
      unique_cluster_indices.insert(labels_host(i));
  auto num_unique_cluster_indices = unique_cluster_indices.size();

  // Record all cluster indices, assigning a unique index to each (which is
  // different from the original cluster index). This will only use noise and
  // core points (see above).
  unsigned int num_clusters = 0;
  std::set<int> cluster_sets;
  for (int i = 0; i < n; ++i)
  {
    if (labels_host(i) >= 0)
    {
      auto id = labels_host(i);
      cluster_sets.insert(id);
      num_clusters++;

      // DFS search
      std::stack<int> stack;
      stack.push(i);
      while (!stack.empty())
      {
        auto k = stack.top();
        stack.pop();
        if (labels_host(k) >= 0)
        {
          labels_host(k) = -1;
          for (int jj = offset_host(k); jj < offset_host(k + 1); ++jj)
          {
            int j = indices_host(jj);
            if (is_core_point(j) || (labels_host(j) == id))
              stack.push(j);
          }
        }
      }
    }
  }
  if (cluster_sets.size() != num_unique_cluster_indices)
  {
    std::cerr << "Number of components does not match\n";
    return false;
  }
  if (num_clusters != num_unique_cluster_indices)
  {
    std::cerr << "Cluster IDs are not unique\n";
    return false;
  }

  return true;
}

template <typename ExecutionSpace, typename IndicesView, typename OffsetView,
          typename LabelsView>
bool verifyClusters(ExecutionSpace const &exec_space, IndicesView indices,
                    OffsetView offset, LabelsView labels, int core_min_size)
{
  int n = labels.size();
  if ((int)offset.size() != n + 1 ||
      KokkosExt::lastElement(exec_space, offset) != (int)indices.size())
    return false;

  using Verify = bool (*)(ExecutionSpace const &, IndicesView, OffsetView,
                          LabelsView, int);

  std::vector<Verify> verify{
      static_cast<Verify>(verifyCorePointsNonnegativeIndex),
      static_cast<Verify>(verifyConnectedCorePointsShareIndex),
      static_cast<Verify>(verifyBorderAndNoisePoints),
      static_cast<Verify>(verifyClustersAreUnique)};
  return std::all_of(verify.begin(), verify.end(), [&](Verify const &verify) {
    return verify(exec_space, indices, offset, labels, core_min_size);
  });
}

template <typename ExecutionSpace, typename Primitives, typename LabelsView>
bool verifyDBSCAN(ExecutionSpace exec_space, Primitives const &primitives,
                  float eps, int core_min_size, LabelsView const &labels)
{
  Kokkos::Profiling::pushRegion("ArborX::DBSCAN::verify");

  static_assert(Kokkos::is_view<LabelsView>{});

  using Points = Details::AccessValues<Primitives, PrimitivesTag>;
  using MemorySpace = typename Points::memory_space;

  static_assert(std::is_same<typename LabelsView::value_type, int>{});
  static_assert(std::is_same<typename LabelsView::memory_space, MemorySpace>{});

  ARBORX_ASSERT(eps > 0);
  ARBORX_ASSERT(core_min_size >= 2);

  Points points{primitives}; // NOLINT

  using Point = typename Points::value_type;
  static_assert(GeometryTraits::is_point_v<Point>);

  ArborX::BoundingVolumeHierarchy<MemorySpace, ArborX::PairValueIndex<Point>>
      bvh(exec_space, ArborX::Experimental::attach_indices(points));

  auto const predicates = Details::PrimitivesWithRadius<Points>{points, eps};

  Kokkos::View<int *, MemorySpace> indices("ArborX::DBSCAN::indices", 0);
  Kokkos::View<int *, MemorySpace> offset("ArborX::DBSCAN::offset", 0);
  ArborX::query(bvh, exec_space, predicates,
                ArborX::Details::LegacyDefaultCallback{}, indices, offset);

  auto passed = Details::verifyClusters(exec_space, indices, offset, labels,
                                        core_min_size);
  Kokkos::Profiling::popRegion();

  return passed;
}
} // namespace Details
} // namespace ArborX

#endif
