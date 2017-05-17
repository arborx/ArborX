/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#ifndef DTK_LINEAR_BVH_DECL_HPP
#define DTK_LINEAR_BVH_DECL_HPP

#include <Kokkos_ArithTraits.hpp>
#include <Kokkos_Array.hpp>
#include <Kokkos_View.hpp>

#include <DTK_DetailsAlgorithms.hpp>
#include <DTK_DetailsBox.hpp>
#include <DTK_DetailsNode.hpp>
#include <DTK_DetailsPredicate.hpp>

#include "DTK_ConfigDefs.hpp"

namespace DataTransferKit
{
/**
 * Bounding Volume Hierarchy.
 */
template <typename NO>
struct BVH
{
  public:
    using DeviceType = typename NO::device_type;

    BVH( Kokkos::View<Box const *, DeviceType> bounding_boxes );

    Kokkos::View<Node *, DeviceType> leaf_nodes;
    Kokkos::View<Node *, DeviceType> internal_nodes;
    /**
     *  Array of indices that sort the boxes used to construct the hierarchy.
     *  The leaf nodes are ordered so we need these to identify objects that
     * meet
     *  a predicate.
     */
    Kokkos::View<int *, DeviceType> indices;
};

} // end namespace DataTransferKit

#endif
