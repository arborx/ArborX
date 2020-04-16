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

#ifndef ARBORX_DISTRIBUTED_SEARCH_TREE_HPP
#define ARBORX_DISTRIBUTED_SEARCH_TREE_HPP

#include <ArborX_Box.hpp>
#include <ArborX_DetailsDistributedSearchTreeImpl.hpp>
#include <ArborX_DetailsUtils.hpp> // accumulate
#include <ArborX_LinearBVH.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

namespace ArborX
{

/** \brief Distributed search tree
 *
 *  \note query() must be called as collective over all processes in the
 *  communicator passed to the constructor.
 */
template <typename MemorySpace, typename Enable = void>
class DistributedSearchTree
{
public:
  using memory_space = MemorySpace;
  static_assert(Kokkos::is_memory_space<MemorySpace>::value, "");
  using size_type = typename BVH<MemorySpace>::size_type;
  using bounding_volume_type = typename BVH<MemorySpace>::bounding_volume_type;

  template <typename ExecutionSpace, typename Primitives>
  DistributedSearchTree(MPI_Comm comm, ExecutionSpace const &space,
                        Primitives const &primitives);

  ~DistributedSearchTree() { MPI_Comm_free(&_comm); }

  /** Returns the smallest axis-aligned box able to contain all the objects
   *  stored in the tree or an invalid box if the tree is empty.
   */
  bounding_volume_type bounds() const noexcept { return _top_tree.bounds(); }

  /** Returns the global number of objects stored in the tree.
   */
  size_type size() const noexcept { return _top_tree_size; }

  /** Indicates whether the tree is empty on all processes.
   */
  bool empty() const noexcept { return size() == 0; }

  /** \brief Finds object satisfying the passed predicates (e.g. nearest to
   *  some point or intersecting with some box)
   *
   *  This query function performs a batch of spatial or k-nearest neighbors
   *  searches.  The results give indices of the objects that satisfy
   *  predicates (as given to the constructor).  They are organized in a
   *  distributed compressed row storage format.
   *
   *  \c indices stores the indices of the objects that satisfy the
   *  predicates.  \c offset stores the locations in the \c indices view that
   *  start a predicate, that is, \c queries(q) is satisfied by \c indices(o)
   *  for <code>objects(q) <= o < objects(q+1)</code> that live on processes
   *  \c ranks(o) respectively.  Following the usual convention,
   *  <code>offset(n) = nnz</code>, where \c n is the number of queries that
   *  were performed and \c nnz is the total number of collisions.
   *
   *  \note The views \c indices, \c offset, and \c ranks are passed by
   *  reference because \c Kokkos::realloc() calls the assignment operator.
   *
   *  \param[in] predicates Collection of predicates of the same type.  These
   *  may be spatial predicates or nearest predicates.
   *  \param[out] args
   *     - \c indices Object local indices that satisfy the predicates.
   *     - \c offset Array of predicate offsets for one-dimensional
   *       storage.
   *     - \c ranks Process ranks that own objects.
   *     - \c distances Computed distances (optional and only for nearest
   *       predicates).
   */
  template <typename ExecutionSpace, typename Predicates, typename... Args>
  void query(ExecutionSpace const &space, Predicates const &predicates,
             Args &&... args) const
  {
    static_assert(Kokkos::is_execution_space<ExecutionSpace>::value, "");
    using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
    using Tag =
        typename Details::Tag<Details::decay_result_of_get_t<Access>>::type;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
    Details::DistributedSearchTreeImpl<DeviceType>::queryDispatch(
        Tag{}, *this, space, predicates, std::forward<Args>(args)...);
  }

private:
  template <typename DeviceType>
  friend struct Details::DistributedSearchTreeImpl;
  MPI_Comm _comm;
  BVH<MemorySpace> _top_tree;    // replicated
  BVH<MemorySpace> _bottom_tree; // local
  size_type _top_tree_size;
  Kokkos::View<size_type *, MemorySpace> _bottom_tree_sizes;
};

template <typename MemorySpace, typename Enable>
template <typename ExecutionSpace, typename Primitives>
DistributedSearchTree<MemorySpace, Enable>::DistributedSearchTree(
    MPI_Comm comm, ExecutionSpace const &space, Primitives const &primitives)
    : _bottom_tree{space, primitives}
{
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value, "");

  // Create new context for the library to isolate library's communication from
  // user's
  MPI_Comm_dup(comm, &_comm);

  int comm_rank;
  MPI_Comm_rank(_comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(_comm, &comm_size);

  Kokkos::View<Box *, MemorySpace> boxes(
      Kokkos::ViewAllocateWithoutInitializing("rank_bounding_boxes"),
      comm_size);
  // FIXME when we move to MPI with CUDA-aware support, we will not need to
  // copy from the device to the host
  auto boxes_host = Kokkos::create_mirror_view(boxes);
  boxes_host(comm_rank) = _bottom_tree.bounds();
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                static_cast<void *>(boxes_host.data()), sizeof(Box), MPI_BYTE,
                _comm);
  Kokkos::deep_copy(space, boxes, boxes_host);

  _top_tree = BVH<MemorySpace>{space, boxes};

  _bottom_tree_sizes = Kokkos::View<size_type *, MemorySpace>(
      Kokkos::ViewAllocateWithoutInitializing("leave_count_in_local_trees"),
      comm_size);
  auto bottom_tree_sizes_host = Kokkos::create_mirror_view(_bottom_tree_sizes);
  bottom_tree_sizes_host(comm_rank) = _bottom_tree.size();
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                static_cast<void *>(bottom_tree_sizes_host.data()),
                sizeof(size_type), MPI_BYTE, _comm);
  Kokkos::deep_copy(space, _bottom_tree_sizes, bottom_tree_sizes_host);

  _top_tree_size = accumulate(space, _bottom_tree_sizes, 0);
}

template <typename DeviceType>
class DistributedSearchTree<
    DeviceType, std::enable_if_t<Details::is_device_type<DeviceType>::value>>
    : public DistributedSearchTree<typename DeviceType::memory_space>
{
public:
  using device_type = DeviceType;
  template <typename Primitives>
  DistributedSearchTree(MPI_Comm comm, Primitives const &primitives)
      : DistributedSearchTree<typename DeviceType::memory_space>(
            comm, typename DeviceType::execution_space{}, primitives)
  {
  }
  template <typename... Args>
  void query(Args &&... args) const
  {
    DistributedSearchTree<typename DeviceType::memory_space>::query(
        typename DeviceType::execution_space{}, std::forward<Args>(args)...);
  }
};

} // namespace ArborX

#endif
