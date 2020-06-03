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

#include <chrono>
#include <set>
#include <stack>

// The algorithm is based on the the following paper:
//
//   [1] Jaiganesh, Jayadharini, and Martin Burtscher. "A high-performance
//   connected components implementation for GPUs." In Proceedings of the 27th
//   International Symposium on High-Performance Parallel and Distributed
//   Computing, pp. 92-104. 2018.
//
// The general gist of the algorithm can be found on the second page and reads:
//
// ```
// ...the majority of the parallel CC algorithms are based on the following
// “label propagation” approach. Each vertex has a label to hold the component
// ID to which it belongs. Initially, this label is set to the vertex ID, that
// is, each vertex is considered a separate component, which can trivially be
// done in parallel. Then the vertices are iteratively processed in parallel to
// determine the connected components. For instance, each vertex’s label can be
// updated with the smallest label among its neighbors. This process repeats
// until there are no more updates, at which point all vertices in a connected
// component have the same label. In this example, the ID of the minimum vertex
// in each component serves as the component ID, which guarantees uniqueness. To
// speed up the computation, the labels are often replaced by “parent” pointers
// that form a union-find data structure.
// ```
//
// Union-find algorithm is sped up through path compression:
//
// ```
// Starting from any vertex, we can follow the parent pointers until we reach a
// vertex that points to itself. This final vertex is the “representative”.
// Every vertex’s implicit label is the ID of its representative and, as before,
// all vertices with the same label (aka representative) belong to the same
// component. Thus, making the representative u point to v indirectly changes
// the labels of all vertices whose representative used to be u. This makes
// union operations very fast. A chain of parent pointers leading to a
// representative is called a “path”. To also make the traversals of these
// paths, i.e., the find operations, very fast, the paths are sporadically
// shortened by making earlier elements skip over some elements and point
// directly to later elements. This process is referred to as “path
// compression”.
// ```
//
// ECL-CC makes use of an intermediate pointer jumping:
//
// ```
// Intermediate pointer jumping only requires a single traversal while still
// compressing the path of all elements encountered along the way, albeit not by
// as much as multiple pointer jumping. It accomplishes this by making every
// element skip over the next element, thus halving the path length in each
// traversal.
// ```
//
// This is encoded in `representative()` function.

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

template <typename ExecutionSpace, typename IndicesView, typename OffsetView,
          typename CCSView>
bool verifyCC(ExecutionSpace exec_space, IndicesView indices, OffsetView offset,
              CCSView ccs)
{
  int num_nodes = ccs.size();
  ARBORX_ASSERT((int)offset.size() == num_nodes + 1);
  ARBORX_ASSERT(ArborX::lastElement(offset) == (int)indices.size());

  // Check that connected vertices have the same cc index
  int num_incorrect = 0;
  Kokkos::parallel_reduce(
      "ArborX::HaloFinder::verify_connected_indices",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_nodes),
      KOKKOS_LAMBDA(int i, int &update) {
        for (int j = offset(i); j < offset(i + 1); ++j)
        {
          if (ccs(i) != ccs(indices(j)))
          {
            // Would like to do fprintf(stderr, ...), but fprintf is __host__
            // function in CUDA
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
    std::cerr << "Number of components does not match" << std::endl;
    return false;
  }
  if (num_ccs != num_unique_cc_indices)
  {
    std::cerr << "Component IDs are not unique" << std::endl;
    return false;
  }

  return true;
}

template <typename MemorySpace>
struct CCSCallback
{
  using tag = ArborX::Details::InlineCallbackTag;
  Kokkos::View<int *, MemorySpace> stat_;

  // Per [1]:
  //
  // Note that the [`representative()`] code is re-entrant and synchronization
  // free even though concurrent execution may cause data races on the parent
  // array. However, these races are guaranteed to be benign for the following
  // reasons. First, the only write to shared data is in [`parent[prev] = next;
  // `]. This write updates a single aligned machine word and is therefore
  // atomic. Moreover, it overwrites a valid entry with another valid entry.
  // Hence, it does not matter if other threads see the old or the new value as
  // either value will allow them to eventually reach the representative.
  // Similarly, all the reads of the parent array will either fetch the old or
  // new value, but both values are acceptable. The only problem that can occur
  // is that two threads try to update the same parent pointer at the same time.
  // In this case, one of the updates is lost. This reduces the code’s
  // performance as duplicate work is performed and the path is not shortened by
  // as much as it could have been, but it does not result in incorrect paths.
  // On average, the savings of not having to perform synchronization far
  // outweighs this small cost. Lastly, it should be noted that the rest of the
  // code either accesses the parent array via calls to the find_repres function
  // or changes the parent pointer of a representative vertex but never of a
  // vertex that is in the middle of a path. If the find_repres code already
  // sees the new representative, it will return it. Otherwise, it will return
  // the old representative. Either return value is handled correctly.
  KOKKOS_FUNCTION
  int representative(int const i) const
  {
    // ##### ECL license (see LICENSE.ECL) #####
    int curr = stat_(i);
    if (curr != i)
    {
      int next;
      int prev = i;
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
        // Per [1]:
        //
        // ```
        // ...checks if the representatives of the two endpoints of the edge are
        // the same. If so, nothing needs to be done. If not, the parent of the
        // larger of the two representatives is updated to point to the smaller
        // representative using an atomic CAS operation (<snip> depending on
        // which representative is smaller). The atomic CAS is required because
        // the representative may have been changed by another thread between
        // the call to `representative()` and the call to the CAS. If it has
        // been changed, `ostat` or `vstat` is updated with the latest value and
        // the do-while loop repeats the computation until it succeeds, i.e.,
        // until there is no data race on the parent.
        // ```

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
               float linking_length, int min_size = 2, bool verbose = false,
               bool verify = false)
{
  static_assert(Kokkos::is_view<HalosIndicesView>{}, "");
  static_assert(Kokkos::is_view<HalosOffsetView>{}, "");
  static_assert(std::is_same<typename HalosIndicesView::value_type, int>{}, "");
  static_assert(std::is_same<typename HalosOffsetView::value_type, int>{}, "");

  using MemorySpace = typename Primitives::memory_space;
  static_assert(
      std::is_same<typename HalosIndicesView::memory_space, MemorySpace>{}, "");
  static_assert(
      std::is_same<typename HalosOffsetView::memory_space, MemorySpace>{}, "");

  Kokkos::Profiling::pushRegion("ArborX::HaloFinder");

  using clock = std::chrono::high_resolution_clock;

  clock::time_point start_total;
  clock::time_point start;
  std::chrono::duration<double> elapsed_construction;
  std::chrono::duration<double> elapsed_query;
  std::chrono::duration<double> elapsed_halos;
  std::chrono::duration<double> elapsed_total;
  std::chrono::duration<double> elapsed_verify = clock::duration::zero();

  start_total = clock::now();

  auto const predicates = wrap(primitives, linking_length);

  int const n = primitives.extent_int(0);

  // build the tree
  start = clock::now();
  Kokkos::Profiling::pushRegion("ArborX::HaloFinder::tree_construction");
  ArborX::BVH<MemorySpace> bvh(exec_space, primitives);
  Kokkos::Profiling::popRegion();
  elapsed_construction = clock::now() - start;

  // perform the queries and build ccs through callback
  // NOTE: indices and offfset are not going to be used as
  // insert() will not be called
  start = clock::now();
  Kokkos::Profiling::pushRegion("ArborX::HaloFinder::ccs");
  Kokkos::View<int *, MemorySpace> indices("ArborX::HaloFinder::indices", 0);
  Kokkos::View<int *, MemorySpace> offset("ArborX::HaloFinder::offset", 0);
  Kokkos::View<int *, MemorySpace> stat(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::HaloFinder::stat"),
      n);
  ArborX::iota(exec_space, stat);
  Kokkos::Profiling::pushRegion("ArborX::HaloFinder::ccs::query");
  bvh.query(exec_space, predicates, CCSCallback<MemorySpace>{stat}, indices,
            offset);
  Kokkos::Profiling::popRegion();
  // Per [1]:
  //
  // ```
  // The finalization kernel will, ultimately, make all parents
  // point directly to the representative.
  // ```
  Kokkos::parallel_for("ArborX::HaloFinder::flatten_stat",
                       Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                       KOKKOS_LAMBDA(int const i) {
                         // ##### ECL license (see LICENSE.ECL) #####
                         int next;
                         int vstat = stat(i);
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

  // Use new name to clearly demonstrate the meaning of this view from now on
  auto ccs = stat;

  elapsed_total += clock::now() - start_total;
  if (verify)
  {
    start = clock::now();
    Kokkos::Profiling::pushRegion("ArborX::HaloFinder::verify");

    bvh.query(exec_space, predicates, indices, offset);
    auto passed = verifyCC(exec_space, indices, offset, ccs);
    printf("Verification %s\n", (passed ? "passed" : "failed"));

    Kokkos::Profiling::popRegion();
    elapsed_verify = clock::now() - start;
  }
  start_total = clock::now();

  // find halos
  start = clock::now();
  Kokkos::Profiling::pushRegion("ArborX::HaloFinder::sort_and_filter_ccs");

  // sort ccs and compute permutation
  auto permute = Details::sortObjects(exec_space, ccs);

  reallocWithoutInitializing(halos_offset, n + 1);
  Kokkos::View<int *, MemorySpace> halos_starts(
      Kokkos::ViewAllocateWithoutInitializing(
          "ArborX::HaloFinder::halos_starts"),
      n);
  int num_halos = 0;
  // In the following scan, we locate the starting position (stored in
  // halos_starts) and size (stored in halos_offset) of each valid halo (i.e.,
  // connected component of size >= min_size). For every index i, we check
  // whether its CC index is different from the previous one (this indicates a
  // start of connected component) and whether the CC index of i + min_size is
  // the same (this indicates that this CC is at least of min_size size). If
  // those are true, we do a linear search from i + min_size till next CC
  // index change to find the CC size.
  Kokkos::parallel_scan(
      "ArborX::HaloFinder::compute_halos_starts_and_sizes",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int i, int &update, bool final_pass) {
        bool const is_cc_first_index = (i == 0 || ccs(i) != ccs(i - 1));
        bool const is_cc_large_enough =
            ((i + min_size - 1 < n) && (ccs(i + min_size - 1) == ccs(i)));
        if (is_cc_first_index && is_cc_large_enough)
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
      "ArborX::HaloFinder::populate_halos",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int i) {
        for (int k = halos_offset(i); k < halos_offset(i + 1); ++k)
        {
          halos_indices(k) = permute(halos_starts(i) + (k - halos_offset(i)));
        }
      });
  Kokkos::Profiling::popRegion();
  elapsed_halos = clock::now() - start;

  elapsed_total += clock::now() - start_total;

  if (verbose)
  {
    printf("total time      : %10.3f\n", elapsed_total.count());
    printf("-> construction : %10.3f\n", elapsed_construction.count());
    printf("-> query+ccs    : %10.3f\n", elapsed_query.count());
    printf("-> halos        : %10.3f\n", elapsed_halos.count());
    if (verify)
      printf("verify          : %10.3f\n", elapsed_verify.count());
  }

  Kokkos::Profiling::popRegion();
}

} // namespace HaloFinder
} // namespace ArborX

#endif
