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

#ifndef ARBORX_DETAILSDBSCANCALLBACK_HPP
#define ARBORX_DETAILSDBSCANCALLBACK_HPP

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

#include <ArborX_Predicates.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{
namespace Details
{

template <typename MemorySpace, typename CorePointsType>
struct DBSCANCallback
{
  Kokkos::View<int *, MemorySpace> stat_;
  CorePointsType is_core_point_;

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

  KOKKOS_FUNCTION
  void combine_trees(int self, int neighbor) const
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
    auto i = self;
    auto j = neighbor;

    // initialize to the first neighbor that's smaller
    if (Kokkos::atomic_compare_exchange(&stat_(i), i, j) == i)
      return;

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

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int j) const
  {
    int const i = ArborX::getData(query);

    // NOTE: for halo finder/ccs algorithm (in which is_core_point(i) is always
    // true), the algorithm below will be simplified to
    //   if (i > j)

    if (!is_core_point_(j))
    {
      // The neighbor is not a core point, do nothing
      return;
    }

    bool is_boundary_point =
        !is_core_point_(i); // is_core_point_(j) is aready true

    if (is_boundary_point && stat_(i) == i)
    {
      // For a boundary point that was not processed before (stat_(i) == i),
      // set its representative to that of the core point. This way, when
      // another neighbor that is core point appears later, we won't process
      // this point.
      //
      // NOTE: DO NOT USE combine_trees(i, j) here. This may set this boundary
      // point as a representative for the whole cluster. This would mean that
      // a) stat_(i) == i still (so it would be processed later, and b) it may
      // be combined with a different cluster later forming a bridge.
      stat_(i) = representative(j);
    }
    else if (!is_boundary_point && i > j)
    {
      // For a core point that is connected to another core point, do the
      // standard CCS algorithm
      combine_trees(i, j);
    }
  }
};
} // namespace Details
} // namespace ArborX

#endif
