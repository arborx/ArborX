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

#ifndef DTK_LINEAR_BVH_DECL_HPP
#define DTK_LINEAR_BVH_DECL_HPP

#include <DTK_Box.hpp>
#include <DTK_DetailsBoundingVolumeHierarchyImpl.hpp>
#include <DTK_DetailsNode.hpp>

#include <Kokkos_View.hpp>

namespace DataTransferKit
{

template <typename DeviceType>
class BoundingVolumeHierarchy
{
  public:
    using bounding_volume_type = Box;
    using size_type = typename Kokkos::View<int *, DeviceType>::size_type;

    BoundingVolumeHierarchy() = default; // build an empty tree

    BoundingVolumeHierarchy(
        Kokkos::View<Box const *, DeviceType> bounding_boxes );

    KOKKOS_INLINE_FUNCTION
    size_type size() const { return _leaf_nodes.extent( 0 ); }

    KOKKOS_INLINE_FUNCTION
    bool empty() const { return size() == 0; }

    KOKKOS_INLINE_FUNCTION
    bounding_volume_type bounds() const
    {
        // NOTE should default constructor initialize to an invalid geometry?
        if ( empty() )
            return bounding_volume_type();
        return ( size() > 1 ? _internal_nodes : _leaf_nodes )[0].bounding_box;
    }

    template <typename Query, typename... Args>
    inline void query( Kokkos::View<Query *, DeviceType> queries,
                       Args &&... args ) const
    {
        using Tag = typename Query::Tag;
        Details::BoundingVolumeHierarchyImpl<DeviceType>::queryDispatch(
            Tag{}, *this, queries, std::forward<Args>( args )... );
    }

  private:
    friend struct Details::TreeTraversal<DeviceType>;

    Kokkos::View<Node *, DeviceType> _leaf_nodes;
    Kokkos::View<Node *, DeviceType> _internal_nodes;
};

template <typename DeviceType>
using BVH = BoundingVolumeHierarchy<DeviceType>;

} // namespace DataTransferKit

#endif
