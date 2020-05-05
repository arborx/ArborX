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
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Macros.hpp>

#include <chrono>

namespace ArborX
{

template <typename View>
struct Wrapped
{
  View _M_view;
  double _r;
};

template <typename View>
auto wrap(View v, double r)
{
  return Wrapped<View>{v, r};
}

namespace Traits
{
template <typename View>
struct Access<Wrapped<View>, PredicatesTag>
{
  using memory_space = typename View::memory_space;
  static size_t size(Wrapped<View> const &w) { return w._M_view.extent(0); }
  static KOKKOS_FUNCTION auto get(Wrapped<View> const &w, size_t i)
  {
    return attach(intersects(Sphere{w._M_view(i), w._r}), (int)i);
  }
};
} // namespace Traits

namespace HaloFinder
{
template <typename MemorySpace>
struct CCSCallback
{
  using tag = ArborX::Details::InlineCallbackTag;
  Kokkos::View<int *, MemorySpace> stat_;

  KOKKOS_INLINE_FUNCTION
  int representative(int const i) const
  {
    // ##### ECL license (see LICENSE.ECL) #####
    int curr = stat_(i);
    if (curr != i)
    {
      int next, prev = i;
      while (curr > (next = stat_(curr)))
      {
        stat_(prev) = next;
        prev = curr;
        curr = next;
      }
    }
    return curr;
  }

  template <typename Query, typename Insert>
  KOKKOS_FUNCTION void operator()(Query const &query, int j,
                                  Insert const &) const
  {
    int const i = ArborX::getData(query);

    // only process edge in one direction
    if (i > j)
    {
      // initialize to the first neighbor that's smaller
      if (Kokkos::atomic_compare_exchange(&stat_(i), i, j) == i)
        return;

      {
        // ##### ECL license (see LICENSE.ECL) #####
        int vstat = representative(i);
        int ostat = representative(j);

        bool repeat;
        do
        {
          repeat = false;
          if (vstat != ostat)
          {
            int ret;
            if (vstat < ostat)
            {
              if ((ret = Kokkos::atomic_compare_exchange(&stat_(ostat), ostat,
                                                         vstat)) != ostat)
              {
                ostat = ret;
                repeat = true;
              }
            }
            else
            {
              if ((ret = Kokkos::atomic_compare_exchange(&stat_(vstat), vstat,
                                                         ostat)) != vstat)
              {
                vstat = ret;
                repeat = true;
              }
            }
          }
        } while (repeat);
      }
    }
  }
};

template <typename ExecutionSpace, typename Primitives,
          typename HalosIndicesView, typename HalosOffsetView>
void findHalos(ExecutionSpace exec_space, Primitives const &primitives,
               HalosIndicesView &halos_indices, HalosOffsetView &halos_offset,
               float linking_length, int min_size = 2)
{
  static_assert(Kokkos::is_view<HalosIndicesView>{}, "");
  static_assert(Kokkos::is_view<HalosOffsetView>{}, "");
  static_assert(std::is_same<typename HalosIndicesView::value_type, int>::value,
                "");
  static_assert(std::is_same<typename HalosOffsetView::value_type, int>::value,
                "");

  using MemorySpace = typename Primitives::memory_space;
  static_assert(
      std::is_same<typename HalosIndicesView::memory_space, MemorySpace>::value,
      "");
  static_assert(
      std::is_same<typename HalosOffsetView::memory_space, MemorySpace>::value,
      "");

  Kokkos::Profiling::pushRegion("ArborX:HaloFinder");

  using clock = std::chrono::high_resolution_clock;

  clock::time_point start_total, start;
  std::chrono::duration<double> elapsed_construction, elapsed_query,
      elapsed_halos, elapsed_total;

  start_total = clock::now();

  auto const predicates = wrap(primitives, linking_length);

  int const n = primitives.extent_int(0);

  // build the tree
  start = clock::now();
  Kokkos::Profiling::pushRegion("ArborX:HaloFinder:tree_construction");
  ArborX::BVH<MemorySpace> bvh(exec_space, primitives);
  Kokkos::Profiling::popRegion();
  elapsed_construction = clock::now() - start;

  // perform the queries and build ccs through callback
  // NOTE: indices and offfset are not going to be used as
  // insert() will not be called
  start = clock::now();
  Kokkos::Profiling::pushRegion("ArborX:HaloFinder:ccs");
  Kokkos::View<int *, MemorySpace> indices("indices", 0);
  Kokkos::View<int *, MemorySpace> offset("offset", 0);
  Kokkos::View<int *, MemorySpace> stat(
      Kokkos::ViewAllocateWithoutInitializing("stat"), n);
  ArborX::iota(exec_space, stat);
  Kokkos::Profiling::pushRegion("ArborX:HaloFinder:ccs:query");
  bvh.query(exec_space, predicates, CCSCallback<MemorySpace>{stat}, indices,
            offset);
  Kokkos::Profiling::popRegion();
  // flatten stat
  Kokkos::parallel_for(ARBORX_MARK_REGION("flatten_stat"),
                       Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                       KOKKOS_LAMBDA(int const i) {
                         // ##### ECL license (see LICENSE.ECL) #####
                         int next, vstat = stat(i);
                         int const old = vstat;
                         while (vstat > (next = stat(vstat)))
                         {
                           vstat = next;
                         }
                         if (vstat != old)
                           stat(i) = vstat;
                       });
  Kokkos::Profiling::popRegion();
  elapsed_query = clock::now() - start;

  // find halos
  start = clock::now();
  Kokkos::Profiling::pushRegion("ArborX:HaloFinder:sort_and_filter_ccs");

  // Use new name to clearly demonstrate the meaning of this view from now on
  auto ccs = stat;

  // sort ccs and compute permutation
  auto permute = Details::sortObjects(exec_space, ccs);

  reallocWithoutInitializing(halos_offset, n + 1);
  Kokkos::View<int *, MemorySpace> halos_starts(
      Kokkos::ViewAllocateWithoutInitializing("halos_starts"), n);
  int num_halos = 0;
  // In the following scan, we locate the starting position (stored in
  // halos_starts) and size (stored in halos_offset) of each valid halo (i.e.,
  // connected component of size >= min_size). For every index i, we check
  // whether its CC index is different from the previous one (this indicates a
  // start of connected component) and whether the CC index of i + min_size is
  // the same (this indicates that this CC is at least of min_size size). If
  // those are true, we do a linear search from i + min_size till next CC
  // index change to find the CC size.
  Kokkos::parallel_scan(ARBORX_MARK_REGION("compute_halos_starts_and_sizes"),
                        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                        KOKKOS_LAMBDA(int i, int &update, bool final_pass) {
                          if ((i + min_size - 1 < n) &&
                              (i == 0 || ccs(i) != ccs(i - 1)) &&
                              (ccs(i + min_size - 1) == ccs(i)))
                          {
                            if (final_pass)
                            {
                              halos_starts(update) = i;
                              int end = i + min_size - 1;
                              while (++end < n && ccs(end) == ccs(i))
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
  elapsed_halos = clock::now() - start;

  Kokkos::Profiling::popRegion();

  elapsed_total = clock::now() - start_total;

  printf("total time      : %10.3f\n", elapsed_total.count());
  printf("-> construction : %10.3f\n", elapsed_construction.count());
  printf("-> query+ccs    : %10.3f\n", elapsed_query.count());
  printf("-> halos        : %10.3f\n", elapsed_halos.count());
} // namespace HaloFinder

} // namespace HaloFinder
} // namespace ArborX

#endif
