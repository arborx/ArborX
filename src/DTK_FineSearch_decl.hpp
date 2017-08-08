/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#ifndef DTK_FINE_SEARCH_DECL_HPP
#define DTK_FINE_SEARCH_DECL_HPP

#include "DTK_ConfigDefs.hpp"
#include <Kokkos_Core.hpp>
#include <Shards_CellTopology.hpp>

namespace DataTransferKit
{
template <typename DeviceType>
class FineSearch
{
  public:
    /**
     * Performs the local search .
     *    @param[out] reference_points The coordinates of the points in the
     * reference space (n_phys_pts, dim)
     *    @param[out] point_in_cell Booleans with value true if the point is in
     * the cell and false otherwise (n_ref_pts)
     *    @param[in] physical_points The coordinates of the points in the
     * physical space (n_phys_pts, dim)
     *    @param[in] cells Cells owned by the processor (n_cells, n_nodes, dim)
     *    @param[in] coarse_search_output_cells Indices of local cells from the
     * coarse search (coarse_output_size)
     *    @param[in] cell_topo Topology of the cells in @param cells
     */
    static void
    search( Kokkos::View<Coordinate **, DeviceType> reference_points,
            Kokkos::View<bool *, DeviceType> point_in_cell,
            Kokkos::View<Coordinate **, DeviceType> physical_points,
            Kokkos::View<Coordinate ***, DeviceType> cells,
            Kokkos::View<unsigned int *, DeviceType> coarse_search_output_cells,
            shards::CellTopology cell_topo );

    /**
     * Same function as above. However, the function is virtual so that the user
     * can provide their own implementation. If the function is not overriden,
     * it throws an exception.
     *    @param[out] reference_points The coordinates of the points in the
     * reference space (n_phys_pts, dim)
     *    @param[out] point_in_cell Booleans with value true if the point is in
     * the cell and false otherwise (n_ref_pts)
     *    @param[in] physical_points The coordinates of the points in the
     * physical space (n_phys_pts, dim)
     *    @param[in] cells Cells owned by the processor (n_cells, n_nodes, dim)
     *    @param[in] coarse_search_output_cells Indices of local cells from the
     * coarse search (coarse_output_size)
     *    @param[in] cell_topo Topology of the cells in @param cells
     */
    virtual void
    search( Kokkos::View<Coordinate **, DeviceType> reference_points,
            Kokkos::View<bool *, DeviceType> point_in_cell,
            Kokkos::View<Coordinate **, DeviceType> physical_points,
            Kokkos::View<Coordinate ***, DeviceType> cells,
            Kokkos::View<unsigned int *, DeviceType> coarse_search_output_cells,
            std::string cell_topo )
    {
        throw std::runtime_error( "Not implemented" );
    }

    static double threshold;
};

// Default value for threshold matches the inclusion tolerance in DTK-2.0 which
// is arbitrary and might need adjustement in client code. See
// https://github.com/ORNL-CEES/DataTransferKit/blob/dtk-2.0/packages/Adapters/Libmesh/src/DTK_LibmeshEntityLocalMap.cpp#L58
template <typename DeviceType>
double FineSearch<DeviceType>::threshold = 1e-6;
}

#endif
