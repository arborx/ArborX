/****************************************************************************
 * Copyright (c) 2017-2024 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_DETAILS_KOKKOS_EXT_DISTRIBUTED_COMM_HPP
#define ARBORX_DETAILS_KOKKOS_EXT_DISTRIBUTED_COMM_HPP

#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>

#include <type_traits>
#include <utility>

#include <mpi.h>

namespace ArborX::Details::KokkosExt
{

template <typename View>
inline constexpr bool is_valid_mpi_view_v =
    (View::rank == 1 &&
     (std::is_same_v<typename View::array_layout, Kokkos::LayoutLeft> ||
      std::is_same_v<typename View::array_layout, Kokkos::LayoutRight>));

template <typename View>
void mpi_isend(MPI_Comm comm, View const &view, int destination, int tag,
               MPI_Request &request)
{
  static_assert(Kokkos::is_view_v<View>);
  static_assert(is_valid_mpi_view_v<View>);

  auto const n = view.size();
  ARBORX_ASSERT(n > 0);

  using ValueType = typename View::value_type;
  MPI_Isend(view.data(), n * sizeof(ValueType), MPI_BYTE, destination, tag,
            comm, &request);
}

template <typename View>
void mpi_irecv(MPI_Comm comm, View const &view, int source, int tag,
               MPI_Request &request)
{
  static_assert(Kokkos::is_view_v<View>);
  static_assert(is_valid_mpi_view_v<View>);

  auto const n = view.size();
  ARBORX_ASSERT(n > 0);

  using ValueType = typename View::value_type;
  MPI_Irecv(view.data(), n * sizeof(ValueType), MPI_BYTE, source, tag, comm,
            &request);
}

template <typename View>
void mpi_allgather(MPI_Comm comm, View const &view)
{
  static_assert(Kokkos::is_view_v<View>);
  static_assert(is_valid_mpi_view_v<View>);

  auto const n = view.size();
  ARBORX_ASSERT(n > 0);

  using ValueType = typename View::value_type;
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, view.data(),
                sizeof(ValueType), MPI_BYTE, comm);
}

template <typename View>
void mpi_alltoall(MPI_Comm comm, View const &view)
{
  static_assert(Kokkos::is_view_v<View>);
  static_assert(is_valid_mpi_view_v<View>);

  auto const n = view.size();
  ARBORX_ASSERT(n > 0);

  using ValueType = typename View::value_type;
  MPI_Alltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, view.data(),
               sizeof(ValueType), MPI_BYTE, comm);
}

} // namespace ArborX::Details::KokkosExt

#endif
