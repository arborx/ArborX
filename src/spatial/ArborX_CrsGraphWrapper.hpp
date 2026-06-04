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

#ifndef ARBORX_CRS_GRAPH_WRAPPER_HPP
#define ARBORX_CRS_GRAPH_WRAPPER_HPP

#include <ArborX_Config.hpp>

#include <detail/ArborX_Iota.hpp>

#include <Kokkos_Core.hpp>

#ifdef ARBORX_ENABLE_MPI
#include <mpi.h>
#endif

namespace ArborX
{

template <typename Tree, typename ExecutionSpace, typename Predicates,
          typename CallbackOrView, typename View, typename... Args>
inline void query(Tree const &tree, ExecutionSpace const &space,
                  Predicates const &predicates,
                  CallbackOrView &&callback_or_view, View &&view,
                  Args &&...args)
{
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value);

  tree.query(space, predicates, std::forward<CallbackOrView>(callback_or_view),
             std::forward<View>(view), std::forward<Args>(args)...);
}

template <template <typename...> class Index, typename MemorySpace,
          typename ExecutionSpace, typename IndexableGetter>
auto create_index(ExecutionSpace const &space, int size,
                  IndexableGetter &&indexable_getter)
{
  return Index<MemorySpace, int, IndexableGetter>(
      space, Details::Iota<MemorySpace>(size),
      std::forward<IndexableGetter>(indexable_getter));
}

template <template <typename...> class Index, typename ExecutionSpace,
          typename IndexableGetter>
auto create_index(ExecutionSpace const &space, int size,
                  IndexableGetter &&indexable_getter)
{
  return create_index<Index, typename ExecutionSpace::memory_space>(
      space, size, std::forward<IndexableGetter>(indexable_getter));
}

template <typename Index, typename ExecutionSpace, typename IndexableGetter>
auto create_index(ExecutionSpace const &space, int size,
                  IndexableGetter &&indexable_getter)
{
  return Index(space, Details::Iota<typename Index::memory_space>(size),
               std::forward<IndexableGetter>(indexable_getter));
}

#ifdef ARBORX_ENABLE_MPI
template <template <typename...> class Index, typename MemorySpace,
          typename ExecutionSpace, typename IndexableGetter>
auto create_index(MPI_Comm comm, ExecutionSpace const &space, int size,
                  IndexableGetter &&indexable_getter)
{
  return Index<MemorySpace, int, IndexableGetter>(
      comm, space, Details::Iota<MemorySpace>(size),
      std::forward<IndexableGetter>(indexable_getter));
}

template <template <typename...> class Index, typename ExecutionSpace,
          typename IndexableGetter>
auto create_index(MPI_Comm comm, ExecutionSpace const &space, int size,
                  IndexableGetter &&indexable_getter)
{
  return create_index<Index, typename ExecutionSpace::memory_space>(
      comm, space, size, std::forward<IndexableGetter>(indexable_getter));
}

template <typename Index, typename ExecutionSpace, typename IndexableGetter>
auto create_index(MPI_Comm comm, ExecutionSpace const &space, int size,
                  IndexableGetter &&indexable_getter)
{
  return Index(comm, space, Details::Iota<typename Index::memory_space>(size),
               std::forward<IndexableGetter>(indexable_getter));
}
#endif

} // namespace ArborX

#endif
