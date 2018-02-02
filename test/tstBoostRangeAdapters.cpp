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

#include <DTK_DetailsAlgorithms.hpp>
#include <DTK_DetailsBox.hpp>
#include <DTK_DetailsPoint.hpp>

#include <Teuchos_UnitTestHarness.hpp>

#include <boost/range/algorithm.hpp> // reverse_copy, replace_if, count, generate, count_if
#include <boost/range/algorithm_ext.hpp> // iota
#include <boost/range/numeric.hpp>       // accumulate

#include "DTK_BoostRangeAdapters.hpp"

#include <random>
#include <sstream>

TEUCHOS_UNIT_TEST( BoostGeometryAdapters, Range )
{
    Kokkos::View<int[4], Kokkos::HostSpace> w( "w" );

    boost::iota( w, 0 );
    std::stringstream ss;
    boost::reverse_copy( w, std::ostream_iterator<int>( ss, " " ) );
    TEST_EQUALITY( ss.str(), "3 2 1 0 " );

    boost::replace_if( w, []( int i ) { return ( i > 1 ); }, -1 );
    TEST_COMPARE_ARRAYS( w, std::vector<int>( {0, 1, -1, -1} ) );

    TEST_EQUALITY( boost::count( w, -1 ), 2 );
    TEST_EQUALITY( boost::accumulate( w, 5 ), 4 );
}

TEUCHOS_UNIT_TEST( BoostGeometryAdapters, PointCloud )
{
    using DataTransferKit::Details::distance;
    using DataTransferKit::Point;
    double const seed = 3.14;
    std::default_random_engine generator( seed );
    std::uniform_real_distribution<double> distribution( -1., 1. );
    int const n = 10000;
    Kokkos::View<Point *, Kokkos::HostSpace> cloud( "cloud", n );
    boost::generate( cloud, [&distribution, &generator]() {
        Point p;
        p[0] = distribution( generator );
        p[1] = distribution( generator );
        p[2] = distribution( generator );
        return p;
    } );

    Point const origin = {0., 0., 0.};
    double const radius = 1.;
    // 4/3 pi 1^3 / 2^3
    double const pi = 6. *
                      static_cast<double>( boost::count_if(
                          cloud,
                          [origin, radius]( Point point ) {
                              return ( distance( point, origin ) <= radius );
                          } ) ) /
                      static_cast<double>( n );

    double const relative_tolerance = .05;
    TEST_FLOATING_EQUALITY( pi, 3.14, relative_tolerance );
}
