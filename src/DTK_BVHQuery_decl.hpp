/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/
#ifndef DTK_BVHQUERY_DECL_HPP
#define DTK_BVHQUERY_DECL_HPP

#include "DTK_ConfigDefs.hpp"

#include "DTK_LinearBVH.hpp"

namespace DataTransferKit
{
template <typename NO>
class BVHQuery
{
  public:
    using DeviceType = typename NO::device_type;
    /**
     * Fill out with the indices of the leaf nodes that satisfy the
     * Nearest predicate.
     */
    static int query( BVH<NO> const bvh, Details::Nearest const &predicates,
                      Kokkos::View<int *, DeviceType> out );

    /**
     * Fill out with the indices of the leaf nodes that satisfy the
     * Within predicate.
     */
    static int query( BVH<NO> const bvh, Details::Within const &predicates,
                      Kokkos::View<int *, DeviceType> out );
};
}

#endif
