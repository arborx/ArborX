/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#ifndef DTK_FINE_SEARCH_FUNCTOR_HPP
#define DTK_FINE_SEARCH_FUNCTOR_HPP

#include <Intrepid2_CellTools_Serial.hpp>
#include <Intrepid2_HGRAD_HEX_C1_FEM.hpp>
#include <Intrepid2_HGRAD_HEX_C2_FEM.hpp>
#include <Intrepid2_HGRAD_PYR_C1_FEM.hpp>
#include <Intrepid2_HGRAD_QUAD_C1_FEM.hpp>
#include <Intrepid2_HGRAD_QUAD_C2_FEM.hpp>
#include <Intrepid2_HGRAD_TET_C1_FEM.hpp>
#include <Intrepid2_HGRAD_TET_C2_FEM.hpp>
#include <Intrepid2_HGRAD_TRI_C1_FEM.hpp>
#include <Intrepid2_HGRAD_TRI_C2_FEM.hpp>
#include <Intrepid2_HGRAD_WEDGE_C1_FEM.hpp>
#include <Intrepid2_HGRAD_WEDGE_C2_FEM.hpp>
#include <Kokkos_Core.hpp>

namespace DataTransferKit
{
namespace CellType
{
class Hexahedron_8
{
  public:
    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION void mapToReferenceFrame(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            physical_point,
        Kokkos::View<Coordinate **, Kokkos::LayoutStride, ExecutionSpace>
            nodes ) const
    {
        typedef Intrepid2::Impl::Basis_HGRAD_HEX_C1_FEM basis_type;
        Intrepid2::Impl::CellTools::Serial::mapToReferenceFrame<basis_type>(
            reference_point, physical_point, nodes );
    }

    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION bool checkPointInclusion(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        double threshold ) const
    {
        return Intrepid2::Impl::Hexahedron<8>::checkPointInclusion(
            reference_point, threshold );
    }
};

class Hexahedron_27
{
  public:
    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION void mapToReferenceFrame(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            physical_point,
        Kokkos::View<Coordinate **, Kokkos::LayoutStride, ExecutionSpace>
            nodes ) const
    {
        typedef Intrepid2::Impl::Basis_HGRAD_HEX_C2_FEM basis_type;
        Intrepid2::Impl::CellTools::Serial::mapToReferenceFrame<basis_type>(
            reference_point, physical_point, nodes );
    }

    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION bool checkPointInclusion(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        double threshold ) const
    {
        return Intrepid2::Impl::Hexahedron<27>::checkPointInclusion(
            reference_point, threshold );
    }
};

class Pyramid_5
{
  public:
    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION void mapToReferenceFrame(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            physical_point,
        Kokkos::View<Coordinate **, Kokkos::LayoutStride, ExecutionSpace>
            nodes ) const
    {
        typedef Intrepid2::Impl::Basis_HGRAD_PYR_C1_FEM basis_type;
        Intrepid2::Impl::CellTools::Serial::mapToReferenceFrame<basis_type>(
            reference_point, physical_point, nodes );
    }

    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION bool checkPointInclusion(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        double threshold ) const
    {
        return Intrepid2::Impl::Pyramid<5>::checkPointInclusion(
            reference_point, threshold );
    }
};

class Quadrilateral_4
{
  public:
    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION void mapToReferenceFrame(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            physical_point,
        Kokkos::View<Coordinate **, Kokkos::LayoutStride, ExecutionSpace>
            nodes ) const
    {
        typedef Intrepid2::Impl::Basis_HGRAD_QUAD_C1_FEM basis_type;
        Intrepid2::Impl::CellTools::Serial::mapToReferenceFrame<basis_type>(
            reference_point, physical_point, nodes );
    }

    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION bool checkPointInclusion(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        double threshold ) const
    {
        return Intrepid2::Impl::Quadrilateral<4>::checkPointInclusion(
            reference_point, threshold );
    }
};

class Quadrilateral_9
{
  public:
    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION void mapToReferenceFrame(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            physical_point,
        Kokkos::View<Coordinate **, Kokkos::LayoutStride, ExecutionSpace>
            nodes ) const
    {
        typedef Intrepid2::Impl::Basis_HGRAD_QUAD_C2_FEM basis_type;
        Intrepid2::Impl::CellTools::Serial::mapToReferenceFrame<basis_type>(
            reference_point, physical_point, nodes );
    }

    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION bool checkPointInclusion(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        double threshold ) const
    {
        return Intrepid2::Impl::Quadrilateral<9>::checkPointInclusion(
            reference_point, threshold );
    }
};

class Tetrahedron_4
{
  public:
    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION void mapToReferenceFrame(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            physical_point,
        Kokkos::View<Coordinate **, Kokkos::LayoutStride, ExecutionSpace>
            nodes ) const
    {
        typedef Intrepid2::Impl::Basis_HGRAD_TET_C1_FEM basis_type;
        Intrepid2::Impl::CellTools::Serial::mapToReferenceFrame<basis_type>(
            reference_point, physical_point, nodes );
    }

    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION bool checkPointInclusion(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        double threshold ) const
    {
        return Intrepid2::Impl::Tetrahedron<4>::checkPointInclusion(
            reference_point, threshold );
    }
};

class Tetrahedron_10
{
  public:
    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION void mapToReferenceFrame(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            physical_point,
        Kokkos::View<Coordinate **, Kokkos::LayoutStride, ExecutionSpace>
            nodes ) const
    {
        typedef Intrepid2::Impl::Basis_HGRAD_TET_C2_FEM basis_type;
        Intrepid2::Impl::CellTools::Serial::mapToReferenceFrame<basis_type>(
            reference_point, physical_point, nodes );
    }

    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION bool checkPointInclusion(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        double threshold ) const
    {
        return Intrepid2::Impl::Tetrahedron<10>::checkPointInclusion(
            reference_point, threshold );
    }
};

class Triangle_3
{
  public:
    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION void mapToReferenceFrame(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            physical_point,
        Kokkos::View<Coordinate **, Kokkos::LayoutStride, ExecutionSpace>
            nodes ) const
    {
        typedef Intrepid2::Impl::Basis_HGRAD_TRI_C1_FEM basis_type;
        Intrepid2::Impl::CellTools::Serial::mapToReferenceFrame<basis_type>(
            reference_point, physical_point, nodes );
    }

    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION bool checkPointInclusion(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        double threshold ) const
    {
        return Intrepid2::Impl::Triangle<3>::checkPointInclusion(
            reference_point, threshold );
    }
};

class Triangle_6
{
  public:
    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION void mapToReferenceFrame(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            physical_point,
        Kokkos::View<Coordinate **, Kokkos::LayoutStride, ExecutionSpace>
            nodes ) const
    {
        typedef Intrepid2::Impl::Basis_HGRAD_TRI_C2_FEM basis_type;
        Intrepid2::Impl::CellTools::Serial::mapToReferenceFrame<basis_type>(
            reference_point, physical_point, nodes );
    }

    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION bool checkPointInclusion(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        double threshold ) const
    {
        return Intrepid2::Impl::Triangle<6>::checkPointInclusion(
            reference_point, threshold );
    }
};

class Wedge_6
{
  public:
    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION void mapToReferenceFrame(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            physical_point,
        Kokkos::View<Coordinate **, Kokkos::LayoutStride, ExecutionSpace>
            nodes ) const
    {
        typedef Intrepid2::Impl::Basis_HGRAD_WEDGE_C1_FEM basis_type;
        Intrepid2::Impl::CellTools::Serial::mapToReferenceFrame<basis_type>(
            reference_point, physical_point, nodes );
    }

    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION bool checkPointInclusion(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        double threshold ) const
    {
        return Intrepid2::Impl::Wedge<6>::checkPointInclusion( reference_point,
                                                               threshold );
    }
};

class Wedge_18
{
  public:
    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION void mapToReferenceFrame(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            physical_point,
        Kokkos::View<Coordinate **, Kokkos::LayoutStride, ExecutionSpace>
            nodes ) const
    {
        typedef Intrepid2::Impl::Basis_HGRAD_WEDGE_C2_FEM basis_type;
        Intrepid2::Impl::CellTools::Serial::mapToReferenceFrame<basis_type>(
            reference_point, physical_point, nodes );
    }

    template <typename ExecutionSpace>
    KOKKOS_INLINE_FUNCTION bool checkPointInclusion(
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            reference_point,
        double threshold ) const
    {
        return Intrepid2::Impl::Wedge<18>::checkPointInclusion( reference_point,
                                                                threshold );
    }
};
}

namespace Functor
{
template <typename CellType, typename DeviceType>
class FineSearch
{
  public:
    FineSearch(
        double threshold,
        Kokkos::View<Coordinate **, DeviceType> reference_points,
        Kokkos::View<bool *, DeviceType> point_in_cell,
        Kokkos::View<Coordinate **, DeviceType> physical_points,
        Kokkos::View<Coordinate ***, DeviceType> cells,
        Kokkos::View<unsigned int *, DeviceType> coarse_search_output_cells )
        : _threshold( threshold )
        , _reference_points( reference_points )
        , _point_in_cell( point_in_cell )
        , _physical_points( physical_points )
        , _cells( cells )
        , _coarse_search_output_cells( coarse_search_output_cells )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( unsigned int const i ) const
    {
        // Extract the indices computed by the coarse search
        unsigned int const cell_index = _coarse_search_output_cells( i );
        // Get the subviews corresponding the reference point (dim), the
        // physical point (dim), the current cell (nodes, dim)
        using ExecutionSpace = typename DeviceType::execution_space;
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            ref_point( _reference_points, i, Kokkos::ALL() );
        Kokkos::View<Coordinate *, Kokkos::LayoutStride, ExecutionSpace>
            phys_point( _physical_points, i, Kokkos::ALL() );
        Kokkos::View<Coordinate **, Kokkos::LayoutStride, ExecutionSpace> nodes(
            _cells, cell_index, Kokkos::ALL(), Kokkos::ALL() );

        // Compute the reference point and return true if the
        // point is inside the cell
        CellType cell_type;
        cell_type.mapToReferenceFrame( ref_point, phys_point, nodes );
        _point_in_cell[i] =
            cell_type.checkPointInclusion( ref_point, _threshold );
    }

  private:
    double _threshold;
    Kokkos::View<Coordinate **, DeviceType> _reference_points;
    Kokkos::View<bool *, DeviceType> _point_in_cell;
    Kokkos::View<Coordinate **, DeviceType> _physical_points;
    Kokkos::View<Coordinate ***, DeviceType> _cells;
    Kokkos::View<unsigned int *, DeviceType> _coarse_search_output_cells;
};
}
}

#endif
