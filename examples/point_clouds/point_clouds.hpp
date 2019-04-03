/****************************************************************************
 * Copyright (c) 2012-2019 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <DTK_Search_Exception.hpp>

#include <Kokkos_View.hpp>

#include <fstream>
#include <random>

enum PointCloudType { filled_box, hollow_box, filled_sphere, hollow_sphere };

template <typename Layout, typename DeviceType>
void writePointCloud(
    Kokkos::View<DataTransferKit::Point *, Layout, DeviceType> random_points,
    std::string const &filename )

{
    static_assert(
        Kokkos::Impl::MemorySpaceAccess<
            Kokkos::HostSpace, typename DeviceType::memory_space>::accessible,
        "The View should be accessible on the Host" );
    std::ofstream file( filename );
    if ( file.is_open() )
    {
        unsigned int const n = random_points.extent( 0 );
        for ( unsigned int i = 0; i < n; ++i )
            file << random_points( i )[0] << " " << random_points( i )[1] << " "
                 << random_points( i )[2] << "\n";
        file.close();
    }
}

template <typename Layout, typename DeviceType>
void filledBoxCloud(
    double const half_edge,
    Kokkos::View<DataTransferKit::Point *, Layout, DeviceType> random_points )
{
    static_assert(
        Kokkos::Impl::MemorySpaceAccess<
            Kokkos::HostSpace, typename DeviceType::memory_space>::accessible,
        "The View should be accessible on the Host" );
    std::uniform_real_distribution<double> distribution( -half_edge,
                                                         half_edge );
    std::default_random_engine generator;
    auto random = [&distribution, &generator]() {
        return distribution( generator );
    };
    unsigned int const n = random_points.extent( 0 );
    for ( unsigned int i = 0; i < n; ++i )
        random_points( i ) = {{random(), random(), random()}};
}

template <typename Layout, typename DeviceType>
void hollowBoxCloud(
    double const half_edge,
    Kokkos::View<DataTransferKit::Point *, Layout, DeviceType> random_points )
{
    static_assert(
        Kokkos::Impl::MemorySpaceAccess<
            Kokkos::HostSpace, typename DeviceType::memory_space>::accessible,
        "The View should be accessible on the Host" );
    std::uniform_real_distribution<double> distribution( -half_edge,
                                                         half_edge );
    std::default_random_engine generator;
    auto random = [&distribution, &generator]() {
        return distribution( generator );
    };
    unsigned int const n = random_points.extent( 0 );
    for ( unsigned int i = 0; i < n; ++i )
    {
        unsigned int face = i % 6;
        switch ( face )
        {
        case 0:
        {
            random_points( i ) = {{-half_edge, random(), random()}};

            break;
        }
        case 1:
        {
            random_points( i ) = {{half_edge, random(), random()}};

            break;
        }
        case 2:
        {
            random_points( i ) = {{random(), -half_edge, random()}};

            break;
        }
        case 3:
        {
            random_points( i ) = {{random(), half_edge, random()}};

            break;
        }
        case 4:
        {
            random_points( i ) = {{random(), random(), -half_edge}};

            break;
        }
        case 5:
        {
            random_points( i ) = {{random(), random(), half_edge}};

            break;
        }
        default:
        {
            throw std::runtime_error( "Your compiler is broken" );
        }
        }
    }
}

template <typename Layout, typename DeviceType>
void filledSphereCloud(
    double const radius,
    Kokkos::View<DataTransferKit::Point *, Layout, DeviceType> random_points )
{
    static_assert(
        Kokkos::Impl::MemorySpaceAccess<
            Kokkos::HostSpace, typename DeviceType::memory_space>::accessible,
        "The View should be accessible on the Host" );
    std::default_random_engine generator;

    std::uniform_real_distribution<double> distribution( -radius, radius );
    auto random = [&distribution, &generator]() {
        return distribution( generator );
    };

    unsigned int const n = random_points.extent( 0 );
    for ( unsigned int i = 0; i < n; ++i )
    {
        bool point_accepted = false;
        while ( !point_accepted )
        {
            double const x = random();
            double const y = random();
            double const z = random();

            // Only accept points that are in the sphere
            if ( std::sqrt( x * x + y * y + z * z ) <= radius )
            {
                random_points( i ) = {{x, y, z}};
                point_accepted = true;
            }
        }
    }
}

template <typename Layout, typename DeviceType>
void hollowSphereCloud(
    double const radius,
    Kokkos::View<DataTransferKit::Point *, Layout, DeviceType> random_points )
{
    static_assert(
        Kokkos::Impl::MemorySpaceAccess<
            Kokkos::HostSpace, typename DeviceType::memory_space>::accessible,
        "The View should be accessible on the Host" );
    std::default_random_engine generator;

    std::uniform_real_distribution<double> distribution( -1., 1. );
    auto random = [&distribution, &generator]() {
        return distribution( generator );
    };

    unsigned int const n = random_points.extent( 0 );
    for ( unsigned int i = 0; i < n; ++i )
    {
        double const x = random();
        double const y = random();
        double const z = random();
        double const norm = std::sqrt( x * x + y * y + z * z );

        random_points( i ) = {
            {radius * x / norm, radius * y / norm, radius * z / norm}};
    }
}

template <typename DeviceType>
void generatePointCloud(
    PointCloudType const point_cloud_type, double const length,
    Kokkos::View<DataTransferKit::Point *, DeviceType> random_points )
{
    auto random_points_host = Kokkos::create_mirror_view( random_points );
    if ( point_cloud_type == PointCloudType::filled_box )
    {
        filledBoxCloud( length, random_points_host );
    }
    else if ( point_cloud_type == PointCloudType::hollow_box )
    {
        hollowBoxCloud( length, random_points_host );
    }
    else if ( point_cloud_type == PointCloudType::filled_sphere )
    {
        filledSphereCloud( length, random_points_host );
    }
    else if ( point_cloud_type == PointCloudType::hollow_sphere )
    {
        hollowSphereCloud( length, random_points_host );
    }
    else
    {
        throw DataTransferKit::SearchException( "not implemented" );
    }
    Kokkos::deep_copy( random_points, random_points_host );
}

template <typename DeviceType>
void loadPointCloud(
    std::string const &filename,
    Kokkos::View<DataTransferKit::Point *, DeviceType> &random_points )
{
    std::ifstream file( filename );
    if ( file.is_open() )
    {
        int size = -1;
        file >> size;
        DTK_SEARCH_ASSERT( size > 0 );
        Kokkos::realloc( random_points, size );
        auto random_points_host = Kokkos::create_mirror_view( random_points );
        for ( int i = 0; i < size; ++i )
            for ( int j = 0; j < 3; ++j )
                file >> random_points( i )[j];
        Kokkos::deep_copy( random_points, random_points_host );
    }
    else
    {
        throw std::runtime_error( "Cannot open file" );
    }
}
