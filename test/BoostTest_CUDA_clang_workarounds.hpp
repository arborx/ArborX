#ifndef BOOSTTEST_CUDA_CLANG_WORKAROUNDS_HPP
#define BOOSTTEST_CUDA_CLANG_WORKAROUNDS_HPP

#include <boost/fusion/adapted/std_tuple.hpp>
#include <boost/fusion/include/std_tuple.hpp>
#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/add_pointer.hpp>

#ifndef BOOST_STATIC_ASSERT
#define BOOST_STATIC_ASSERT(m) static_assert(m, "");
#endif

#endif
