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

#include <memory>
#include <optional>

#include <mpi.h>

namespace Details
{

template <typename MemorySpace>
class DistributedTreePostQueryComms
{
public:
  DistributedTreePostQueryComms() = default;

  template <typename ExecutionSpace, typename IndicesAndRanks>
  DistributedTreePostQueryComms(MPI_Comm comm, ExecutionSpace const &space,
                                IndicesAndRanks const &indices_and_ranks)
  {
    std::size_t data_len = indices_and_ranks.extent(0);

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
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "Example::DTPQC::tmp"),
        data_len);

    // Split indices/ranks
    Kokkos::View<int *, MemorySpace> indices(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::DTPQC::indices"),
        data_len);
    Kokkos::View<int *, MemorySpace> ranks(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::DTPQC::ranks"),
        data_len);
    Kokkos::parallel_for(
        "Example::DTPQC::indices_and_ranks_split",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, data_len),
        KOKKOS_LAMBDA(int const i) {
          indices(i) = indices_and_ranks(i).index;
          ranks(i) = indices_and_ranks(i).rank;
        });

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
                           "Example::DTPQC::rev_indices"),
        _num_requests);
    ArborX::iota(space, mpi_tmp);
    ArborX::Details::DistributedTreeImpl<MemorySpace>::sendAcrossNetwork(
        space, distributor_forth, mpi_tmp, mpi_rev_indices);

    // This retrieves which source index a process wants and gives it to
    // the process owning the source
    _mpi_send_indices = Kokkos::View<int *, MemorySpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::DTPQC::send_indices"),
        _num_requests);
    ArborX::Details::DistributedTreeImpl<MemorySpace>::sendAcrossNetwork(
        space, distributor_forth, indices, _mpi_send_indices);

    // This builds the temporary buffer that will create the reverse
    // distributor to dispatch the values
    Kokkos::View<int *, MemorySpace> mpi_rev_ranks(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::DTPQC::rev_ranks"),
        _num_requests);
    Kokkos::deep_copy(space, mpi_tmp, rank);
    ArborX::Details::DistributedTreeImpl<MemorySpace>::sendAcrossNetwork(
        space, distributor_forth, mpi_tmp, mpi_rev_ranks);

    // This will create the reverse of the previous distributor
    _distributor = ArborX::Details::Distributor<MemorySpace>(*_comm);
    _num_responses = _distributor->createFromSends(space, mpi_rev_ranks);

    // There should be enough responses to perfectly fill what was requested
    // i.e. _num_responses == data_len

    // The we send back the requested indices so that each process can rebuild
    // their output
    _mpi_recv_indices = Kokkos::View<int *, MemorySpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::DTPQC::recv_indices"),
        _num_responses);
    ArborX::Details::DistributedTreeImpl<MemorySpace>::sendAcrossNetwork(
        space, *_distributor, mpi_rev_indices, _mpi_recv_indices);
  }

  template <typename ExecutionSpace, typename Values>
  Kokkos::View<typename ArborX::Details::AccessTraitsHelper<
                   ArborX::AccessTraits<Values, ArborX::PrimitivesTag>>::type *,
               typename ArborX::AccessTraits<
                   Values, ArborX::PrimitivesTag>::memory_space>
  distribute(ExecutionSpace const &space, Values const &source)
  {
    using src_acc = ArborX::AccessTraits<Values, ArborX::PrimitivesTag>;
    using value_t = typename ArborX::Details::AccessTraitsHelper<src_acc>::type;
    using memory_space = typename src_acc::memory_space;

    // We know what each process want so we prepare the data to be sent
    Kokkos::View<value_t *, MemorySpace> data_to_send(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::DTPQC::data_to_send"),
        _num_requests);
    Kokkos::parallel_for(
        "Example::DTPQC::data_to_send_fill",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, _num_requests),
        KOKKOS_CLASS_LAMBDA(int const i) {
          data_to_send(i) = src_acc::get(source, _mpi_send_indices(i));
        });

    // We properly send the data, and each process has what it wants, but in the
    // wrong order
    Kokkos::View<value_t *, MemorySpace> data_to_recv(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::DTPQC::data_to_recv"),
        _num_responses);
    ArborX::Details::DistributedTreeImpl<MemorySpace>::sendAcrossNetwork(
        space, *_distributor, data_to_send, data_to_recv);

    // So we fix this by moving everything
    Kokkos::View<value_t *, memory_space> output(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "Example::DTPQC::output"),
        _num_responses);
    Kokkos::parallel_for(
        "Example::DTPQC::output_fill",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, _num_responses),
        KOKKOS_CLASS_LAMBDA(int const i) {
          output(_mpi_recv_indices(i)) = data_to_recv(i);
        });

    return output;
  }

private:
  std::shared_ptr<MPI_Comm> _comm;
  Kokkos::View<int *, MemorySpace> _mpi_send_indices;
  Kokkos::View<int *, MemorySpace> _mpi_recv_indices;
  std::optional<ArborX::Details::Distributor<MemorySpace>> _distributor;
  std::size_t _num_requests;
  std::size_t _num_responses;
};

} // namespace Details
