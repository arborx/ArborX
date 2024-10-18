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

#ifndef ARBORX_DISTRIBUTED_TREE_HPP
#define ARBORX_DISTRIBUTED_TREE_HPP

#include <ArborX_Box.hpp>
#include <ArborX_LinearBVH.hpp>
#include <detail/ArborX_AccessTraits.hpp>
#include <detail/ArborX_DistributedTreeNearest.hpp>
#include <detail/ArborX_DistributedTreeSpatial.hpp>
#include <detail/ArborX_PairValueIndex.hpp>
#include <kokkos_ext/ArborX_KokkosExtStdAlgorithms.hpp>

#include <Kokkos_Core.hpp>

#include <memory>

#include <mpi.h>

namespace ArborX
{

template <typename BottomTree>
class DistributedTreeBase
{
private:
  using MemorySpace = typename BottomTree::memory_space;
  using BoundingVolume = typename BottomTree::bounding_volume_type;
  using TopTree = BoundingVolumeHierarchy<
      MemorySpace, PairValueIndex<BoundingVolume, int>,
      Experimental::Indexable<PairValueIndex<BoundingVolume, int>>,
      BoundingVolume>;

  using bottom_tree_type = BottomTree;
  using top_tree_type = TopTree;

public:
  using memory_space = MemorySpace;
  static_assert(Kokkos::is_memory_space<MemorySpace>::value);
  using size_type = typename MemorySpace::size_type;
  using bounding_volume_type = BoundingVolume;
  using value_type = typename BottomTree::value_type;

  DistributedTreeBase() = default; // build an empty tree

  template <typename ExecutionSpace, typename... Args>
  DistributedTreeBase(MPI_Comm comm, ExecutionSpace const &space,
                      Args &&...args);

  // Return the smallest axis-aligned box able to contain all the objects
  // stored in the tree or an invalid box if the tree is empty.
  bounding_volume_type bounds() const noexcept { return _top_tree.bounds(); }

  // Return the global number of objects stored in the tree
  size_type size() const noexcept { return _top_tree_size; }

  // Indicate whether the tree is empty on all processes
  bool empty() const noexcept { return size() == 0; }

  // Find objects satisfying the passed predicates (e.g. nearest to some point
  // or intersecting with some box)
  //
  // This query function performs a batch of spatial or k-nearest neighbors
  // searches.  The results give indices of the objects that satisfy predicates
  // (as given to the constructor).  They are organized in a distributed
  // compressed row storage format.
  //
  // `indices` stores the indices of the objects that satisfy the predicates.
  // `offset` stores the locations in the `indices` view that start a
  // predicate, that is, "queries(q)" is satisfied by `indices(o)` for
  // `objects(q) <= o < objects(q+1)` that live on processes `ranks(o)`
  // respectively.  Following the usual convention, `offset(n) = nnz`, where
  // `n` is the number of queries that were performed and `nnz` is the total
  // number of collisions.
  template <typename ExecutionSpace, typename UserPredicates, typename... Args>
  void query(ExecutionSpace const &space, UserPredicates const &user_predicates,
             Args &&...args) const
  {
    static_assert(
        Details::KokkosExt::is_accessible_from<MemorySpace,
                                               ExecutionSpace>::value);

    using Predicates = Details::AccessValues<UserPredicates, PredicatesTag>;
    static_assert(Details::KokkosExt::is_accessible_from<
                      typename Predicates::memory_space, ExecutionSpace>::value,
                  "Predicates must be accessible from the execution space");

    Predicates predicates{user_predicates}; // NOLINT

    using Tag = typename Predicates::value_type::Tag;
    Details::DistributedTreeImpl::queryDispatch(Tag{}, *this, space, predicates,
                                                std::forward<Args>(args)...);
  }

  auto const &indexable_get() const { return _bottom_tree.indexable_get(); }

protected:
  MPI_Comm getComm() const { return *_comm_ptr; }

private:
  friend struct Details::DistributedTreeImpl;

  std::shared_ptr<MPI_Comm> _comm_ptr{
      std::make_unique<MPI_Comm>(MPI_COMM_NULL)};
  BottomTree _bottom_tree; // local
  TopTree _top_tree;       // replicated
  size_type _top_tree_size{0};
  Kokkos::View<size_type *, MemorySpace> _bottom_tree_sizes;
};

// NOTE: query() must be called as collective over all processes in the
// communicator passed to the constructor
template <typename MemorySpace, typename Value,
          typename IndexableGetter = Experimental::Indexable<Value>,
          typename BoundingVolume = Box<
              GeometryTraits::dimension_v<
                  std::decay_t<std::invoke_result_t<IndexableGetter, Value>>>,
              typename GeometryTraits::coordinate_type_t<
                  std::decay_t<std::invoke_result_t<IndexableGetter, Value>>>>>
class DistributedTree
    : public DistributedTreeBase<BoundingVolumeHierarchy<
          MemorySpace, Value, IndexableGetter, BoundingVolume>>

{
  using base_type = DistributedTreeBase<BoundingVolumeHierarchy<
      MemorySpace, Value, IndexableGetter, BoundingVolume>>;

public:
  using memory_space = MemorySpace;
  static_assert(Kokkos::is_memory_space<MemorySpace>::value);
  using bounding_volume_type = BoundingVolume;
  using value_type = Value;

  DistributedTree() = default; // build an empty tree

  template <typename ExecutionSpace, typename Values>
  DistributedTree(MPI_Comm comm, ExecutionSpace const &space,
                  Values const &values,
                  IndexableGetter const &indexable_getter = IndexableGetter())
      : base_type(comm, space, values, indexable_getter)
  {}
};

template <typename ExecutionSpace, typename Values>
#if KOKKOS_VERSION >= 40400
KOKKOS_DEDUCTION_GUIDE
#else
KOKKOS_FUNCTION
#endif
    DistributedTree(MPI_Comm, ExecutionSpace, Values) -> DistributedTree<
        typename Details::AccessValues<Values, PrimitivesTag>::memory_space,
        typename Details::AccessValues<Values, PrimitivesTag>::value_type>;

template <typename BottomTree>
template <typename ExecutionSpace, typename... Args>
DistributedTreeBase<BottomTree>::DistributedTreeBase(
    MPI_Comm comm, ExecutionSpace const &space, Args &&...args)
{
  Kokkos::Profiling::pushRegion("ArborX::DistributedTree::DistributedTree");

  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value);

  // Create new context for the library to isolate library's communication from
  // user's
  _comm_ptr.reset(
      // duplicate the communicator and store it in a std::shared_ptr so that
      // all copies of the distributed tree point to the same object
      [comm]() {
        auto p = std::make_unique<MPI_Comm>();
        MPI_Comm_dup(comm, p.get());
        return p.release();
      }(),
      // custom deleter to mark the communicator object for deallocation
      [](MPI_Comm *p) {
        MPI_Comm_free(p);
        delete p;
      });

  Kokkos::Profiling::pushRegion("ArborX::DistributedTree::DistributedTree::"
                                "bottom_tree_construction");

  _bottom_tree = BottomTree(space, std::forward<Args>(args)...);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::DistributedTree::DistributedTree::"
                                "top_tree_construction");

  int comm_rank;
  MPI_Comm_rank(getComm(), &comm_rank);
  int comm_size;
  MPI_Comm_size(getComm(), &comm_size);

  Kokkos::View<BoundingVolume *, MemorySpace> volumes(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::DistributedTree::DistributedTree::"
                         "rank_bounding_volumes"),
      comm_size);

  Kokkos::DefaultHostExecutionSpace host_exec;
#ifdef ARBORX_ENABLE_GPU_AWARE_MPI
  Kokkos::deep_copy(space, Kokkos::subview(volumes, comm_rank),
                    _bottom_tree.bounds());
  space.fence("ArborX::DistributedTree::DistributedTree"
              " (fill on device done before MPI_Allgather)");

  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                static_cast<void *>(volumes.data()), sizeof(BoundingVolume),
                MPI_BYTE, getComm());
#else
  auto volumes_host = Kokkos::create_mirror_view(
      Kokkos::view_alloc(host_exec, Kokkos::WithoutInitializing), volumes);
  host_exec.fence();
  volumes_host(comm_rank) = _bottom_tree.bounds();

  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                static_cast<void *>(volumes_host.data()),
                sizeof(BoundingVolume), MPI_BYTE, getComm());

  Kokkos::deep_copy(space, volumes, volumes_host);
#endif

  // Build top tree with attached ranks
  _top_tree = TopTree{space, Experimental::attach_indices<int>(volumes)};

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::DistributedTree::DistributedTree::"
                                "size_calculation");

  _bottom_tree_sizes = Kokkos::View<size_type *, MemorySpace>(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborX::DistributedTree::"
                         "leave_count_in_local_trees"),
      comm_size);
  auto bottom_tree_sizes_host = Kokkos::create_mirror_view(
      Kokkos::view_alloc(host_exec, Kokkos::WithoutInitializing),
      _bottom_tree_sizes);
  host_exec.fence();
  bottom_tree_sizes_host(comm_rank) = _bottom_tree.size();
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                static_cast<void *>(bottom_tree_sizes_host.data()),
                sizeof(size_type), MPI_BYTE, getComm());
  Kokkos::deep_copy(space, _bottom_tree_sizes, bottom_tree_sizes_host);

  _top_tree_size = Details::KokkosExt::reduce(space, _bottom_tree_sizes, 0);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
}

} // namespace ArborX

#endif
