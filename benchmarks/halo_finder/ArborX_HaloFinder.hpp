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

#ifndef ARBORX_HALO_FINDER_HPP
#define ARBORX_HALO_FINDER_HPP

#include <ArborX_DetailsSortUtils.hpp>
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_Macros.hpp>

#include <Kokkos_Macros.hpp>
#include <Kokkos_View.hpp>

#include <set>
#include <stack>

#include "ECL.hpp"

namespace ArborX
{

namespace HaloFinder
{

template <typename ExecutionSpace, typename... P>
bool verifyCC(ExecutionSpace exec_space, Kokkos::View<int *, P...> offset,
              Kokkos::View<int *, P...> indices, Kokkos::View<int *, P...> ccs);

template <typename ExecutionSpace, typename... P>
void findHalos(ExecutionSpace exec_space, Kokkos::View<int *, P...> offset,
               Kokkos::View<int *, P...> indices,
               Kokkos::View<int *, P...> &halos_offset,
               Kokkos::View<int *, P...> &halos_indices, int min_size = 2,
               bool verify = false)
{
  using MemorySpace = typename Kokkos::View<int *, P...>::memory_space;
#if defined(KOKKOS_ENABLE_CUDA)
  static_assert(std::is_same<ExecutionSpace, Kokkos::Cuda>::value,
                "HaloFinder is only available with CUDA execution "
                "spaceimplemented only for CUDA");
#else
  static_assert(false,
                "HaloFinder is not available when Kokkos CUDA is not enabled");
#endif

  Kokkos::Profiling::pushRegion("ArborX:HaloFinder");

  int num_nodes = offset.size() - 1;
  int num_edges = indices.size();

  Kokkos::Profiling::pushRegion("ArborX:HaloFinder:ECL");

  Kokkos::View<int *, MemorySpace> ccs(
      Kokkos::ViewAllocateWithoutInitializing("ccs"), num_nodes);
  computeCC(num_nodes, num_edges, offset.data(), indices.data(), ccs.data());

  Kokkos::Profiling::popRegion();

  if (verify)
  {
    Kokkos::Profiling::pushRegion("ArborX:HaloFinder:verify_ccs");
    ARBORX_ASSERT(verifyCC(exec_space, offset, indices, ccs));
    Kokkos::Profiling::popRegion();
  }

  Kokkos::Profiling::pushRegion("ArborX:HaloFinder:sort_and_filter_ccs");

  // sort ccs and compute permutation
  auto permute = Details::sortObjects(exec_space, ccs);

  reallocWithoutInitializing(halos_offset, num_nodes + 1);
  Kokkos::View<int *, MemorySpace> halos_starts(
      Kokkos::ViewAllocateWithoutInitializing("halos_starts"), num_nodes);
  int num_halos = 0;
  // In the following scan, we locate the starting position (stored in
  // halos_starts) and size (stored in halos_offset) of each valid halo (i.e.,
  // connected component of size >= min_size). For every index i, we check
  // whether its CC index is different from the previous one (this indicates a
  // start of connected component) and whether the CC index of i + min_size is
  // the same (this indicates that this CC is at least of min_size size). If
  // those are true, we do a linear search from i + min_size till next CC index
  // change to find the CC size.
  Kokkos::parallel_scan(
      ARBORX_MARK_REGION("compute_halos_starts_and_sizes"),
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_nodes),
      KOKKOS_LAMBDA(int i, int &update, bool final_pass) {
        if ((i + min_size - 1 < num_nodes) &&
            (i == 0 || ccs(i) != ccs(i - 1)) &&
            (ccs(i + min_size - 1) == ccs(i)))
        {
          if (final_pass)
          {
            halos_starts(update) = i;
            int end = i + min_size - 1;
            while (++end < num_nodes && ccs(end) == ccs(i))
              ; // do nothing
            halos_offset(update) = end - i;
          }
          ++update;
        }
      },
      num_halos);
  Kokkos::resize(halos_offset, num_halos + 1);
  exclusivePrefixSum(exec_space, halos_offset);

  // Copy ccs indices to halos
  reallocWithoutInitializing(halos_indices, lastElement(halos_offset));
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("populate_halos"),
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int i) {
        for (int k = halos_offset(i); k < halos_offset(i + 1); ++k)
        {
          halos_indices(k) = permute(halos_starts(i) + (k - halos_offset(i)));
        }
      });

  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::popRegion();
}

template <typename ExecutionSpace, typename... P>
bool verifyCC(ExecutionSpace exec_space, Kokkos::View<int *, P...> offset,
              Kokkos::View<int *, P...> indices, Kokkos::View<int *, P...> ccs)
{
  int num_nodes = ccs.size();
  ARBORX_ASSERT((int)offset.size() == num_nodes + 1);
  ARBORX_ASSERT(ArborX::lastElement(offset) == (int)indices.size());

  // Check that there are no negative cc indices
  int num_incorrect = 0;
  Kokkos::parallel_reduce(
      ARBORX_MARK_REGION("verify_negative_indices"),
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_nodes),
      KOKKOS_LAMBDA(int i, int &update) {
        if (ccs(i) < 0)
        {
          printf("Negative cc index: %d\n", i);
          update++;
        }
      },
      num_incorrect);
  if (num_incorrect)
    return false;

  // Check that connected vertices have the same cc index
  num_incorrect = 0;
  Kokkos::parallel_reduce(
      ARBORX_MARK_REGION("verify_connected_indices"),
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_nodes),
      KOKKOS_LAMBDA(int i, int &update) {
        for (int j = offset(i); j < offset(i + 1); ++j)
        {
          if (ccs(i) != ccs(indices(j)))
          {
            printf("Non-matching cc indices: %d [%d] -> %d [%d]\n", i, ccs(i),
                   indices(j), ccs(indices(j)));
            update++;
          }
        }
      },
      num_incorrect);
  if (num_incorrect)
    return false;

  // Check that non-connected vertices have different cc indices
  auto ccs_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ccs);
  auto offset_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offset);
  auto indices_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices);

  std::set<int> unique_cc_indices;
  for (int i = 0; i < num_nodes; i++)
    unique_cc_indices.insert(ccs_host(i));
  auto num_unique_cc_indices = unique_cc_indices.size();

  unsigned int num_ccs = 0;
  std::set<int> cc_sets;
  for (int i = 0; i < num_nodes; i++)
  {
    if (ccs_host(i) >= 0)
    {
      auto id = ccs_host(i);
      cc_sets.insert(id);
      num_ccs++;

      // DFS search
      std::stack<int> stack;
      stack.push(i);
      while (!stack.empty())
      {
        auto k = stack.top();
        stack.pop();
        if (ccs_host(k) >= 0)
        {
          ARBORX_ASSERT(ccs_host(k) == id);
          ccs_host(k) = -1;
          for (int j = offset_host(k); j < offset_host(k + 1); j++)
            stack.push(indices_host(j));
        }
      }
    }
  }
  if (cc_sets.size() != num_unique_cc_indices)
  {
    // FIXME: Not sure how we can get here, but it was in the original verify
    // check in ECL
    std::cout << "Number of components does not match" << std::endl;
    return false;
  }
  if (num_ccs != num_unique_cc_indices)
  {
    std::cout << "Component IDs are not unique" << std::endl;
    return false;
  }

  return true;
}

} // namespace HaloFinder
} // namespace ArborX

#endif
