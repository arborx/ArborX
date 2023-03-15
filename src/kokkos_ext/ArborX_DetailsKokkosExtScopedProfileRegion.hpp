/****************************************************************************
 * Copyright (c) 2017-2023 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DETAILS_KOKKOS_EXT_SCOPED_PROFILE_REGION_HPP
#define ARBORX_DETAILS_KOKKOS_EXT_SCOPED_PROFILE_REGION_HPP

#include <Kokkos_Macros.hpp>

#if KOKKOS_VERSION >= 40099
#include <Kokkos_Profiling_ScopedRegion.hpp>
namespace KokkosExt
{
using ScopedProfileRegion = Kokkos::Profiling::ScopedRegion;
}
#else
#include <Kokkos_Core.hpp>

#include <string>

namespace KokkosExt
{

class ScopedProfileRegion
{
public:
  ScopedProfileRegion(ScopedProfileRegion const &) = delete;
  ScopedProfileRegion &operator=(ScopedProfileRegion const &) = delete;

  explicit ScopedProfileRegion(std::string const &name)
  {
    Kokkos::Profiling::pushRegion(name);
  }
  ~ScopedProfileRegion() { Kokkos::Profiling::popRegion(); }
};

} // namespace KokkosExt
#endif

#endif
