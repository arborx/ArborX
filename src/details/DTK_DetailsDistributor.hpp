/****************************************************************************
 * Copyright (c) 2012-2018 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef DTK_DETAILS_DISTRIBUTOR_HPP
#define DTK_DETAILS_DISTRIBUTOR_HPP

#include <Teuchos_ArrayView.hpp>
#include <Tpetra_Distributor.hpp>

namespace DataTransferKit
{
namespace Details
{

class Distributor
{
  public:
    Distributor( Teuchos::RCP<Teuchos::Comm<int> const> comm )
        : _distributor( comm )
    {
    }
    size_t createFromSends( Teuchos::ArrayView<int const> const &exportProcIDs )
    {
        return _distributor.createFromSends( exportProcIDs );
    }
    template <typename Packet>
    void doPostsAndWaits( Teuchos::ArrayView<Packet const> const &exports,
                          size_t numPackets,
                          Teuchos::ArrayView<Packet> const &imports )
    {
        _distributor.doPostsAndWaits( exports, numPackets, imports );
    }
    Teuchos::ArrayView<size_t const> getLengthsFrom() const
    {
        return _distributor.getLengthsFrom();
    }
    Teuchos::ArrayView<size_t const> getLengthsTo() const
    {
        return _distributor.getLengthsTo();
    }
    size_t getTotalReceiveLength() const
    {
        return _distributor.getTotalReceiveLength();
    }

  private:
    Tpetra::Distributor _distributor;
};

} // namespace Details
} // namespace DataTransferKit

#endif
