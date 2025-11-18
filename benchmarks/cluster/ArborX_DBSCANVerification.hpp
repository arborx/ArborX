/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
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

#include <ArborX_Config.hpp>

#include <ArborX_LinearBVH.hpp>
#ifdef ARBORX_ENABLE_MPI
#include <ArborX_DistributedTree.hpp>
#endif
#include <kokkos_ext/ArborX_KokkosExtViewHelpers.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include <Kokkos_Swap.hpp>

#include <iostream>
#include <set>
#include <stack>
#include <vector>

namespace ArborX::Details
{

namespace
{
struct PairIndexLabel
{
  unsigned index;
  int label;
  bool is_core;
};
#ifdef ARBORX_ENABLE_MPI
struct TupleIndexRankLabel
{
  unsigned index;
  int rank;
  long long label;
  bool is_core;
};
#endif

template <typename Neighbor>
KOKKOS_FUNCTION void dual_print(char const *msg, Neighbor self)
{
  Kokkos::printf(msg);
  if constexpr (std::is_same_v<Neighbor, PairIndexLabel>)
    Kokkos::printf(": %d [%d]\n", self.index, self.label);
  else
    Kokkos::printf(": %d (%d) [%lld]\n", self.index, self.rank, self.label);
}
template <typename Neighbor>
KOKKOS_FUNCTION void dual_print(char const *msg, Neighbor self, Neighbor neigh)
{
  Kokkos::printf(msg);
  if constexpr (std::is_same_v<Neighbor, PairIndexLabel>)
    Kokkos::printf(": %d [%d] -> %d [%d]\n", self.index, self.label,
                   neigh.index, neigh.label);
  else
    Kokkos::printf(": %d (%d) [%lld] -> %d (%d) [%lld]\n", self.index,
                   self.rank, self.label, neigh.index, neigh.rank, neigh.label);
}

template <typename Neighbors>
inline constexpr bool is_serial_v =
    std::is_same_v<typename Neighbors::value_type, PairIndexLabel>;

} // namespace

// Check that core points have nonnegative indices
template <typename ExecutionSpace, typename Offset, typename Neighbors>
bool verifyCorePointsNonnegativeIndex(ExecutionSpace const &exec_space,
                                      Offset offset, Neighbors neighbors,
                                      bool verbose)
{
  auto const n = offset.size() - 1;

  int num_incorrect;
  Kokkos::parallel_reduce(
      "ArborX::DBSCAN::verify_core_points_nonnegative",
      Kokkos::RangePolicy(exec_space, 0, n),
      KOKKOS_LAMBDA(int i, int &update) {
        auto self = neighbors(offset(i));
        if (self.is_core && self.label < 0)
        {
          ++update;
          if (verbose)
            dual_print("Core point is marked as noise", self);
        }
      },
      num_incorrect);
  return (num_incorrect == 0);
}

// Check that connected core points have same cluster indices
template <typename ExecutionSpace, typename Offset, typename Neighbors>
bool verifyConnectedCorePointsShareIndex(ExecutionSpace const &exec_space,
                                         Offset offset, Neighbors neighbors,
                                         bool verbose)
{
  auto const n = offset.size() - 1;

  int num_incorrect;
  Kokkos::parallel_reduce(
      "ArborX::DBSCAN::verify_connected_core_points",
      Kokkos::RangePolicy(exec_space, 0, n),
      KOKKOS_LAMBDA(int i, int &update) {
        auto const self = neighbors(offset(i));
        if (self.is_core)
        {
          for (int j = offset(i) + 1; j < offset(i + 1); ++j)
          {
            auto const neigh = neighbors(j);
            if (neigh.is_core && self.label != neigh.label)
            {
              ++update;
              if (verbose)
                dual_print("Connected cores do not belong to the same cluster",
                           self, neigh);
            }
          }
        }
      },
      num_incorrect);
  return (num_incorrect == 0);
}

// Check that noise points have index -1
template <typename ExecutionSpace, typename Offset, typename Neighbors>
bool verifyNoisePoints(ExecutionSpace const &exec_space, Offset offset,
                       Neighbors neighbors, bool verbose)
{
  auto const n = offset.size() - 1;

  int num_incorrect;
  Kokkos::parallel_reduce(
      "ArborX::DBSCAN::verify_connected_border_points",
      Kokkos::RangePolicy(exec_space, 0, n),
      KOKKOS_LAMBDA(int i, int &update) {
        auto const self = neighbors(offset(i));
        if (self.is_core)
          return;
        for (int j = offset(i) + 1; j < offset(i + 1); ++j)
        {
          auto const neigh = neighbors(j);
          if (neigh.is_core)
            return; // border point
        }

        // Noise points must have index -1
        if (self.label != -1)
        {
          update++;
          if (verbose)
            dual_print("Noise point does not have index -1", self);
        }
      },
      num_incorrect);
  return (num_incorrect == 0);
}

// Check that border points share index with at least one core point
template <typename ExecutionSpace, typename Offset, typename Neighbors>
bool verifyConnectedBorderPoints(ExecutionSpace const &exec_space,
                                 Offset offset, Neighbors neighbors,
                                 bool verbose)
{
  auto const n = offset.size() - 1;

  int num_incorrect;
  Kokkos::parallel_reduce(
      "ArborX::DBSCAN::verify_connected_border_points",
      Kokkos::RangePolicy(exec_space, 0, n),
      KOKKOS_LAMBDA(int i, int &update) {
        auto const self = neighbors(offset(i));
        if (self.is_core)
          return;
        bool is_border = false;
        bool have_shared_core = false;
        for (int j = offset(i) + 1; j < offset(i + 1); ++j)
        {
          auto const neigh = neighbors(j);
          if (neigh.is_core)
          {
            is_border = true;
            if (self.label == neigh.label)
            {
              have_shared_core = true;
              break;
            }
          }
        }

        // Border point must be connected to a core point
        if (is_border && !have_shared_core)
        {
          update++;
          if (verbose)
            dual_print("Border point does not belong to a cluster", self);
        }
      },
      num_incorrect);
  return (num_incorrect == 0);
}

// Check that border points are marked as noise
template <typename ExecutionSpace, typename Offset, typename Neighbors>
bool verifyIgnoredBorderPoints(ExecutionSpace const &exec_space, Offset offset,
                               Neighbors neighbors, bool verbose)
{
  auto const n = offset.size() - 1;

  int num_incorrect;
  Kokkos::parallel_reduce(
      "ArborX::DBSCAN::verify_connected_border_points",
      Kokkos::RangePolicy(exec_space, 0, n),
      KOKKOS_LAMBDA(int i, int &update) {
        auto const self = neighbors(offset(i));
        if (self.is_core)
          return;
        bool is_border = false;
        for (int j = offset(i) + 1; j < offset(i + 1); ++j)
        {
          auto const neigh = neighbors(j);
          if (neigh.is_core)
          {
            is_border = true;
            break;
          }
        }

        // Border point must be connected to a core point
        if (is_border && self.label != -1)
        {
          update++;
          if (verbose)
            dual_print("Border point does not have index -1", self);
        }
      },
      num_incorrect);
  return (num_incorrect == 0);
}

// Check that cluster indices are unique
template <typename ExecutionSpace, typename Offset, typename Neighbors>
bool verifyClustersAreUnique(ExecutionSpace const &, Offset offset_device,
                             Neighbors neighbors_device, bool verbose)
{
  int const n = offset_device.size() - 1;

  auto offset =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offset_device);
  auto neighbors = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                       neighbors_device);

  // Remove all border points from consideration (noise points are already -1)
  // The idea is that this way if labels were bridged through a border
  // point, we will count them as separate labels but with a shared cluster
  // index, which will fail the unique labels check
  for (int i = 0; i < n; ++i)
  {
    auto &self = neighbors(offset(i));
    if (!self.is_core)
    {
      for (int j = offset(i); j < offset(i + 1); ++j)
      {
        if (neighbors(j).is_core)
        {
          // The point is a border point
          self.label = -1;
          break;
        }
      }
    }
  }

  // Record all unique cluster indices
  std::set<int> unique_cluster_indices;
  for (int i = 0; i < n; ++i)
  {
    auto const self = neighbors(offset(i));
    if (self.label != -1)
      unique_cluster_indices.insert(self.label);
  }
  auto num_unique_cluster_indices = unique_cluster_indices.size();

  // Record all cluster indices, assigning a unique index to each (which is
  // different from the original cluster index). This will only use noise and
  // core points (see above).
  unsigned int num_clusters = 0;
  std::set<int> cluster_sets;
  for (int i = 0; i < n; ++i)
  {
    auto const self = neighbors(offset(i));
    if (self.label >= 0)
    {
      auto id = self.label;
      cluster_sets.insert(id);
      ++num_clusters;

      // DFS search
      std::stack<int> stack;
      stack.push(i);
      while (!stack.empty())
      {
        auto k = stack.top();
        stack.pop();
        if (neighbors(offset(k)).label >= 0)
        {
          neighbors(offset(k)).label = -1;
          for (int j = offset(k); j < offset(k + 1); ++j)
          {
            auto neigh = neighbors(j);
            if (neigh.is_core || neigh.label == id)
              stack.push(neigh.index);
          }
        }
      }
    }
  }
  if (cluster_sets.size() != num_unique_cluster_indices)
  {
    if (verbose)
      std::cerr << "Number of components does not match\n";
    return false;
  }
  if (num_clusters != num_unique_cluster_indices)
  {
    if (verbose)
      std::cerr << "Cluster IDs are not unique\n";
    return false;
  }

  return true;
}

template <typename Labels>
struct IndexLabelCallback
{
  Labels _labels;

  template <typename Query, typename Value, typename Output>
  KOKKOS_FUNCTION auto operator()(Query const &, Value const &value,
                                  Output const &out) const
  {
    out({value.index, _labels(value.index), 0 /*is_core*/});
  }
};

template <typename ExecutionSpace, typename Primitives, typename Labels,
          typename Coordinate>
bool verifyDBSCAN(ExecutionSpace exec_space, Primitives const &primitives,
                  Coordinate eps, int core_min_size, Labels const &labels,
                  std::string const &algorithm = "dbscan", bool verbose = false)
{
  Kokkos::Profiling::ScopedRegion guard("ArborX::DBSCAN::verify");

  static_assert(Kokkos::is_view<Labels>{});

  using Points = Details::AccessValues<Primitives>;
  using MemorySpace = typename Points::memory_space;

  static_assert(std::is_same_v<typename Labels::value_type, int>);
  static_assert(std::is_same_v<typename Labels::memory_space, MemorySpace>);

  ARBORX_ASSERT(eps > 0);
  ARBORX_ASSERT(core_min_size >= 2);
  ARBORX_ASSERT(algorithm == "dbscan" || algorithm == "dbscan*");

  Points points{primitives}; // NOLINT
  auto const n = points.size();

  using Point = typename Points::value_type;
  static_assert(GeometryTraits::is_point_v<Point>);

  ArborX::BoundingVolumeHierarchy index(
      exec_space, ArborX::Experimental::attach_indices(points));

  Kokkos::View<PairIndexLabel *, MemorySpace> neighbors(
      "ArborX::DBSCAN::neighbors", 0);
  Kokkos::View<int *, MemorySpace> offset("ArborX::DBSCAN::offset", 0);
  index.query(exec_space, ArborX::Experimental::make_intersects(points, eps),
              IndexLabelCallback<Labels>{labels}, neighbors, offset);

  Kokkos::parallel_for(
      "ArborX::DBSCAN::set_neighbors", Kokkos::RangePolicy(exec_space, 0, n),
      KOKKOS_LAMBDA(unsigned i) {
        int self_index = -1;
        for (int jj = offset(i); jj < offset(i + 1); ++jj)
        {
          auto j = neighbors(jj).index;
          neighbors(jj).is_core = (offset(j + 1) - offset(j) >= core_min_size);
          if (j == i)
            self_index = jj;
        }

        // Place self first
        if (self_index != offset(i))
          Kokkos::kokkos_swap(neighbors(offset(i)), neighbors(self_index));

        KOKKOS_ASSERT(neighbors(offset(i)).index == i);
      });

  using Verify = bool (*)(ExecutionSpace const &, decltype(offset),
                          decltype(neighbors), bool);

  std::vector<Verify> verify;
  if (algorithm == "dbscan")
    verify = {static_cast<Verify>(verifyCorePointsNonnegativeIndex),
              static_cast<Verify>(verifyConnectedCorePointsShareIndex),
              static_cast<Verify>(verifyNoisePoints),
              static_cast<Verify>(verifyConnectedBorderPoints),
              static_cast<Verify>(verifyClustersAreUnique)};
  else
    verify = {static_cast<Verify>(verifyCorePointsNonnegativeIndex),
              static_cast<Verify>(verifyConnectedCorePointsShareIndex),
              static_cast<Verify>(verifyNoisePoints),
              static_cast<Verify>(verifyIgnoredBorderPoints),
              static_cast<Verify>(verifyClustersAreUnique)};
  return std::all_of(verify.begin(), verify.end(), [&](Verify const &verify) {
    return verify(exec_space, offset, neighbors, verbose);
  });
}

#ifdef ARBORX_ENABLE_MPI
struct IndexCallback
{
  template <typename Query, typename Value, typename Output>
  KOKKOS_FUNCTION auto operator()(Query const &, Value const &value,
                                  Output const &out) const
  {
    out({value.index});
  }
};

template <typename Labels, typename IsCore>
struct IndexRankLabelCoreCallback
{
  int _rank;
  Labels _labels;
  IsCore _is_core;

  template <typename Query, typename Value, typename Output>
  KOKKOS_FUNCTION auto operator()(Query const &, Value const &value,
                                  Output const &out) const
  {
    auto const i = value.index;
    out({i, _rank, _labels(i), _is_core(i)});
  }
};

template <typename ExecutionSpace, typename Primitives, typename Labels,
          typename Coordinate>
bool verifyDBSCAN(MPI_Comm comm, ExecutionSpace exec_space,
                  Primitives const &primitives, Coordinate eps,
                  int core_min_size, Labels const &labels, bool verbose = false)
{
  Kokkos::Profiling::ScopedRegion guard("ArborX::DBSCAN::verify");

  static_assert(Kokkos::is_view<Labels>{});

  using Points = Details::AccessValues<Primitives>;
  using MemorySpace = typename Points::memory_space;

  static_assert(std::is_same_v<typename Labels::value_type, long long>);
  static_assert(std::is_same_v<typename Labels::memory_space, MemorySpace>);

  ARBORX_ASSERT(eps > 0);
  ARBORX_ASSERT(core_min_size >= 2);

  Points points{primitives}; // NOLINT
  auto const n = points.size();

  using Point = typename Points::value_type;
  static_assert(GeometryTraits::is_point_v<Point>);

  // FIXME_NVCC: nvcc 11.7 (and maybe others) cannot compile this without
  // template arguments.
  ArborX::DistributedTree<MemorySpace,
                          PairValueIndex<typename Points::value_type, unsigned>>
      index(comm, exec_space, ArborX::Experimental::attach_indices(points));

  // Phase 1: determine core points by getting offset
  Kokkos::View<bool *, MemorySpace> is_core(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::DBSCAN::is_core"),
      n);
  {
    Kokkos::View<int *, MemorySpace> offset("ArborX::DBSCAN::offset", 0);
    Kokkos::View<unsigned *, MemorySpace> indices("ArborX::DBSCAN::indices", 0);
    index.query(exec_space, ArborX::Experimental::make_intersects(points, eps),
                IndexCallback{}, indices, offset);

    Kokkos::parallel_for(
        "ArborX::DBSCAN::set_is_core", Kokkos::RangePolicy(exec_space, 0, n),
        KOKKOS_LAMBDA(int i) {
          is_core(i) = (offset(i + 1) - offset(i) >= core_min_size);
        });
  }

  int rank;
  MPI_Comm_rank(comm, &rank);

  // Phase 2: return point information (index, rank, label, core) from all ranks
  Kokkos::View<TupleIndexRankLabel *, MemorySpace> neighbors(
      "ArborX::DBSCAN::neighbors", 0);
  Kokkos::View<int *, MemorySpace> offset("ArborX::DBSCAN::offset", 0);
  index.query(exec_space, ArborX::Experimental::make_intersects(points, eps),
              IndexRankLabelCoreCallback<Labels, decltype(is_core)>{
                  rank, labels, is_core},
              neighbors, offset);
  Kokkos::parallel_for(
      "ArborX::DBSCAN::set_neighbors", Kokkos::RangePolicy(exec_space, 0, n),
      KOKKOS_LAMBDA(unsigned i) {
        int self_index = -1;
        for (int j = offset(i); j < offset(i + 1); ++j)
        {
          if (neighbors(j).index == i && neighbors(j).rank == rank)
            self_index = j;
        }

        // Place self first
        if (self_index != offset(i))
          Kokkos::kokkos_swap(neighbors(offset(i)), neighbors(self_index));

        KOKKOS_ASSERT(neighbors(offset(i)).index == i &&
                      neighbors(offset(i)).rank == rank);
      });

  using Verify = bool (*)(ExecutionSpace const &, decltype(offset),
                          decltype(neighbors), bool);

  // FIXME: we are skipping verifyClustersAreUnique check as no idea how to do
  // it in distributed setting right now
  std::vector<Verify> verify{
      static_cast<Verify>(verifyCorePointsNonnegativeIndex),
      static_cast<Verify>(verifyConnectedCorePointsShareIndex),
      static_cast<Verify>(verifyNoisePoints),
      static_cast<Verify>(verifyConnectedBorderPoints)};
  int local_success =
      std::all_of(verify.begin(), verify.end(), [&](Verify const &verify) {
        return verify(exec_space, offset, neighbors, verbose);
      });

  int global_success;
  MPI_Allreduce(&local_success, &global_success, 1, MPI_INT, MPI_MIN, comm);

  return global_success;
}
#endif

} // namespace ArborX::Details

#endif
