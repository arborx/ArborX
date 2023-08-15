/****************************************************************************
 * Copyright (c) 2023 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#pragma once

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

#include <cassert>
#include <memory>
#include <optional>

#include "common.hpp"
#include <mpi.h>

template <typename MemorySpace>
class MPIComms
{
public:
  MPIComms() = default;

  template <typename ExecutionSpace>
  MPIComms(ExecutionSpace const &space, MPI_Comm comm,
           Kokkos::View<int *, MemorySpace> indices,
           Kokkos::View<int *, MemorySpace> ranks)
  {
    assert(indices.extent(0) == ranks.extent(0));
    std::size_t data_len = indices.extent(0);

    _comm.reset(
        [comm]() {
          auto p = new MPI_Comm;
          MPI_Comm_dup(comm, p);
          return p;
        }(),
        [](MPI_Comm *p) {
          int mpi_finalized;
          MPI_Finalized(&mpi_finalized);
          if (!mpi_finalized)
            MPI_Comm_free(p);
          delete p;
        });

    int rank;
    MPI_Comm_rank(*_comm, &rank);

    Kokkos::View<int *, MemorySpace> mpi_tmp(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::MPI::tmp"),
        data_len);

    // Computes what will be common to every exchange. Every time
    // someone wants to get the value from the same set of elements,
    // they will use the same list of recv and send indices.
    // The rank data will be saved inside the back distributor,
    // as the front one is not relevant once the recv indices
    // are computed.

    // This builds for each process a local array indicating how much
    // informatiom will be gathered
    ArborX::Details::Distributor<MemorySpace> distributor_forth(*_comm);
    _num_requests = distributor_forth.createFromSends(space, ranks);

    // This creates the temporary buffer that will help when producing the
    // array that rebuilds the output
    Kokkos::View<int *, MemorySpace> mpi_rev_indices(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::MPI::rev_indices"),
        _num_requests);
    ArborX::iota(space, mpi_tmp);
    ArborX::Details::DistributedTreeImpl<MemorySpace>::sendAcrossNetwork(
        space, distributor_forth, mpi_tmp, mpi_rev_indices);

    // This retrieves which source index a process wants and gives it to
    // the process owning the source
    _mpi_send_indices = Kokkos::View<int *, MemorySpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::MPI::send_indices"),
        _num_requests);
    ArborX::Details::DistributedTreeImpl<MemorySpace>::sendAcrossNetwork(
        space, distributor_forth, indices, _mpi_send_indices);

    // This builds the temporary buffer that will create the reverse
    // distributor to dispatch the values
    Kokkos::View<int *, MemorySpace> mpi_rev_ranks(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::MPI::rev_ranks"),
        _num_requests);
    Kokkos::deep_copy(space, mpi_tmp, rank);
    ArborX::Details::DistributedTreeImpl<MemorySpace>::sendAcrossNetwork(
        space, distributor_forth, mpi_tmp, mpi_rev_ranks);

    // This will create the reverse of the previous distributor
    _distributor_back = ArborX::Details::Distributor<MemorySpace>(*_comm);
    _num_responses = _distributor_back->createFromSends(space, mpi_rev_ranks);

    // There should be enough responses to perfectly fill what was requested
    assert(_num_responses == data_len);

    // The we send back the requested indices so that each process can rebuild
    // the output
    _mpi_recv_indices = Kokkos::View<int *, MemorySpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::MPI::recv_indices"),
        _num_responses);
    ArborX::Details::DistributedTreeImpl<MemorySpace>::sendAcrossNetwork(
        space, *_distributor_back, mpi_rev_indices, _mpi_recv_indices);
  }

  template <typename ExecutionSpace, typename Values>
  Kokkos::View<Details::inner_value_t<Values> *, MemorySpace>
  distributeArborX(ExecutionSpace const &space, Values const &source)
  {
    using value_t = Details::inner_value_t<Values>;
    using access = ArborX::AccessTraits<Values, ArborX::PrimitivesTag>;
    assert(_distributor_back.has_value());

    // We know what each process want so we prepare the data to be sent
    Kokkos::View<value_t *, MemorySpace> data_to_send(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::MPI::data_to_send"),
        _num_requests);
    Kokkos::parallel_for(
        "Example::MPI::data_to_send_fill",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, _num_requests),
        KOKKOS_CLASS_LAMBDA(int const i) {
          data_to_send(i) = access::get(source, _mpi_send_indices(i));
        });

    return distribute(space, data_to_send);
  }

  template <typename ExecutionSpace, typename ValueType>
  Kokkos::View<ValueType *, MemorySpace>
  distributeView(ExecutionSpace const &space,
                 Kokkos::View<ValueType *, MemorySpace> const &source)
  {
    assert(_distributor_back.has_value());

    // We know what each process want so we prepare the data to be sent
    Kokkos::View<ValueType *, MemorySpace> data_to_send(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::MPI::data_to_send"),
        _num_requests);
    Kokkos::parallel_for(
        "Example::MPI::data_to_send_fill",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, _num_requests),
        KOKKOS_CLASS_LAMBDA(int const i) {
          data_to_send(i) = source(_mpi_send_indices(i));
        });

    return distribute(space, data_to_send);
  }

private:
  std::shared_ptr<MPI_Comm> _comm;
  Kokkos::View<int *, MemorySpace> _mpi_send_indices;
  Kokkos::View<int *, MemorySpace> _mpi_recv_indices;
  std::optional<ArborX::Details::Distributor<MemorySpace>> _distributor_back;
  std::size_t _num_requests;
  std::size_t _num_responses;

  template <typename ExecutionSpace, typename ValueType>
  Kokkos::View<ValueType *, MemorySpace>
  distribute(ExecutionSpace const &space,
             Kokkos::View<ValueType *, MemorySpace> const &data_to_send)
  {
    // We properly send the data, and each process has what it wants, but in the
    // wrong order
    Kokkos::View<ValueType *, MemorySpace> data_to_recv(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::MPI::data_to_recv"),
        _num_responses);
    ArborX::Details::DistributedTreeImpl<MemorySpace>::sendAcrossNetwork(
        space, *_distributor_back, data_to_send, data_to_recv);

    // So we fix this by moving everything
    Kokkos::View<ValueType *, MemorySpace> output(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::MPI::output"),
        _num_responses);
    Kokkos::parallel_for(
        "Example::MPI::output_fill",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, _num_responses),
        KOKKOS_CLASS_LAMBDA(int const i) {
          output(_mpi_recv_indices(i)) = data_to_recv(i);
        });

    return output;
  }
};