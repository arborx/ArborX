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

#ifndef ARBORX_DETAILSUNIONFIND_HPP
#define ARBORX_DETAILSUNIONFIND_HPP

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

#include <Kokkos_Core.hpp>

namespace ArborX::Details
{

template <typename MemorySpace, bool DoSerial = false>
struct UnionFind
{
  using memory_space = MemorySpace;

  Kokkos::View<int *, MemorySpace> _labels;

  UnionFind(Kokkos::View<int *, MemorySpace> labels)
      : _labels(labels)
  {}

  KOKKOS_FUNCTION
  auto size() const { return _labels.size(); }

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
    int curr = _labels(i);
    if (curr != i)
    {
      int next;
      int prev = i;
      while (curr > (next = _labels(curr)))
      {
        _labels(prev) = next;
        prev = curr;
        curr = next;
      }
    }
    return curr;
  }

  // In some situations it is necessary to make sure that the a particular
  // label is assigned to a point. As a regular merge() does not guarantee
  // that, an extra function is introduced, which assigns the label of the
  // second point (or, rather, the label of its representative) to the first.
  KOKKOS_FUNCTION
  void merge_into(int i, int j) const { _labels(i) = representative(j); }

  KOKKOS_FUNCTION
  void merge(int i, int j) const
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

    int vstat = representative(i);
    int ostat = representative(j);

    if constexpr (DoSerial)
    {
      if (vstat < ostat)
        _labels(ostat) = vstat;
      else
        _labels(vstat) = ostat;
    }
    else
    {
      // Note: the code does one extra iteration even when the labels array was
      // updated. However, it does not show up in any of the performance
      // studies, and the code is cleaner and more compact.
      while (vstat != ostat)
      {
        if (vstat < ostat)
          ostat =
              Kokkos::atomic_compare_exchange(&_labels(ostat), ostat, vstat);
        else
          vstat =
              Kokkos::atomic_compare_exchange(&_labels(vstat), vstat, ostat);
      }
    }
  }
};

} // namespace ArborX::Details

#endif
