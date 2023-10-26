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

#ifndef ARBORX_TEST_LEGACY_TREE_HPP
#define ARBORX_TEST_LEGACY_TREE_HPP

#include <ArborX_DetailsLegacy.hpp>

template <typename Tree>
class LegacyTree : public Tree
{
public:
  LegacyTree() = default;

  template <typename ExecutionSpace, typename Primitives>
  LegacyTree(ExecutionSpace const &space, Primitives const &primitives)
      : Tree(space,
             ArborX::Details::LegacyValues<Primitives,
                                           typename Tree::bounding_volume_type>{
                 primitives})
  {}
};

#endif
