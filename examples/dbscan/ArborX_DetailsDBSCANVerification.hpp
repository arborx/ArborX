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

#ifndef ARBORX_DETAILSDBSCANVERIFICATION_HPP
#define ARBORX_DETAILSDBSCANVERIFICATION_HPP

#include <ArborX_DetailsUtils.hpp>

#include <Kokkos_View.hpp>

#include <set>
#include <stack>

namespace ArborX
{
namespace Details
{

// Check that connected core points have same cluster indices
// NOTE: if core_min_size = 1, all points are core points
template <typename ExecutionSpace, typename IndicesView, typename OffsetView,
          typename ClusterView>
bool verifyConnectedCorePointsShareIndex(ExecutionSpace const &exec_space,
                                         IndicesView indices, OffsetView offset,
                                         ClusterView clusters,
                                         int core_min_size)
{
  int n = clusters.size();

  int num_incorrect = 0;
  Kokkos::parallel_reduce(
      "ArborX::DBSCAN::verify_connected_core_points",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int i, int &update) {
        bool self_is_core_point =
            (offset(i + 1) - offset(i) - 1 >= core_min_size);
        if (self_is_core_point)
        {
          for (int jj = offset(i); jj < offset(i + 1); ++jj)
          {
            int j = indices(jj);
            bool neigh_is_core_point =
                (offset(j + 1) - offset(j) - 1 >= core_min_size);

            if (neigh_is_core_point && clusters(i) != clusters(j))
            {
              printf("Connected cores do not belong to the same cluster: "
                     "%d [%d] -> %d [%d]\n",
                     i, clusters(i), j, clusters(j));
              update++;
            }
          }
        }
      },
      num_incorrect);
  return (num_incorrect == 0);
}

// Check that boundary points share index with at least one core point
template <typename ExecutionSpace, typename IndicesView, typename OffsetView,
          typename ClusterView>
bool verifyBoundaryPointsConnectToCorePoints(ExecutionSpace const &exec_space,
                                             IndicesView indices,
                                             OffsetView offset,
                                             ClusterView clusters,
                                             int core_min_size)
{
  int n = clusters.size();

  int num_incorrect = 0;
  Kokkos::parallel_reduce(
      "ArborX::DBSCAN::verify_connected_boundary_points",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int i, int &update) {
        bool self_is_core_point =
            (offset(i + 1) - offset(i) - 1 >= core_min_size);
        if (!self_is_core_point)
        {
          bool is_boundary = false;
          bool have_shared_core = false;
          for (int jj = offset(i); jj < offset(i + 1); ++jj)
          {
            int j = indices(jj);
            bool neigh_is_core_point =
                (offset(j + 1) - offset(j) - 1 >= core_min_size);

            if (neigh_is_core_point)
            {
              is_boundary = true;
              if (clusters(i) == clusters(j))
              {
                have_shared_core = true;
                break;
              }
            }
          }

          if (is_boundary && !have_shared_core)
          {
            printf("Boundary point does not belong to a cluster: "
                   "%d [%d]\n",
                   i, clusters(i));
            update++;
          }
        }
      },
      num_incorrect);
  return (num_incorrect == 0);
}

// Check that cluster indices are unique
template <typename ExecutionSpace, typename IndicesView, typename OffsetView,
          typename ClusterView>
bool verifyClustersAreUnique(ExecutionSpace const &, IndicesView indices,
                             OffsetView offset, ClusterView clusters,
                             int core_min_size)
{
  int n = clusters.size();

  auto clusters_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, clusters);
  auto offset_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offset);
  auto indices_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices);

  auto is_core_point = [&](int i) {
    return offset_host(i + 1) - offset_host(i) - 1 >= core_min_size;
  };

  // Remove all boundary points from consideration
  // The idea is that this way if clusters were bridged through a boundary
  // point, we will count them as separate clusters but with a shared cluster
  // index, which will fail the unique clusters check
  for (int i = 0; i < n; ++i)
  {
    if (!is_core_point(i))
    {
      for (int jj = offset_host(i); jj < offset_host(i + 1); ++jj)
      {
        int j = indices_host(jj);
        if (is_core_point(j))
        {
          // The point is a boundary point
          clusters_host(i) = -1;
          break;
        }
      }
    }
  }

  // Record all unique cluster indices
  std::set<int> unique_cluster_indices;
  for (int i = 0; i < n; ++i)
    if (clusters_host(i) != -1)
      unique_cluster_indices.insert(clusters_host(i));
  auto num_unique_cluster_indices = unique_cluster_indices.size();

  // Record all cluster indices, assigning a unique index to each (which is
  // different from the original cluster index). This will only use noise and
  // core points (see above).
  unsigned int num_clusters = 0;
  std::set<int> cluster_sets;
  for (int i = 0; i < n; ++i)
  {
    if (clusters_host(i) >= 0)
    {
      auto id = clusters_host(i);
      cluster_sets.insert(id);
      num_clusters++;

      // DFS search
      std::stack<int> stack;
      stack.push(i);
      while (!stack.empty())
      {
        auto k = stack.top();
        stack.pop();
        if (clusters_host(k) >= 0)
        {
          clusters_host(k) = -1;
          for (int jj = offset_host(k); jj < offset_host(k + 1); ++jj)
          {
            int j = indices_host(jj);
            if (is_core_point(j) || (clusters_host(j) == id))
              stack.push(j);
          }
        }
      }
    }
  }
  if (cluster_sets.size() != num_unique_cluster_indices)
  {
    std::cerr << "Number of components does not match" << std::endl;
    return false;
  }
  if (num_clusters != num_unique_cluster_indices)
  {
    std::cerr << "Cluster IDs are not unique" << std::endl;
    return false;
  }

  return true;
}

template <typename ExecutionSpace, typename IndicesView, typename OffsetView,
          typename ClusterView>
bool verifyClusters(ExecutionSpace const &exec_space, IndicesView indices,
                    OffsetView offset, ClusterView clusters, int core_min_size)
{
  int n = clusters.size();
  if ((int)offset.size() != n + 1 ||
      ArborX::lastElement(offset) != (int)indices.size())
    return false;

  using Verify = bool (*)(ExecutionSpace const &, IndicesView, OffsetView,
                          ClusterView, int);

  for (auto verify :
       {static_cast<Verify>(verifyConnectedCorePointsShareIndex),
        static_cast<Verify>(verifyBoundaryPointsConnectToCorePoints),
        static_cast<Verify>(verifyClustersAreUnique)})
  {
    if (!verify(exec_space, indices, offset, clusters, core_min_size))
      return false;
  }

  return true;
}

} // namespace Details
} // namespace ArborX

#endif
