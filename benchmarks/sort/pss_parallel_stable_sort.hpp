#ifndef PSS_HPP
#define PSS_HPP

/*
  Copyright (C) 2014 Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
  * Neither the name of Intel Corporation nor the names of its
    contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
  AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
  WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.
*/

#include "pss_common.hpp"

namespace pss {

namespace internal {

// Merge sequences [xs,xe) and [ys,ye) to output sequence [zs,zs+(xe-xs)+(ye-ys))
// Destroy input sequence iff destroy==true
template <typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename RandomAccessIterator3,
          typename Compare>
void parallel_move_merge(RandomAccessIterator1 xs, RandomAccessIterator1 xe,
                         RandomAccessIterator2 ys, RandomAccessIterator2 ye,
                         RandomAccessIterator3 zs,
                         bool destroy, Compare comp,
                         ssize_t cutoff) {
  while( (xe-xs) + (ye-ys) > cutoff ) {
    RandomAccessIterator1 xm;
    RandomAccessIterator2 ym;
    if( xe-xs < ye-ys  ) {
      ym = ys+(ye-ys)/2;
      xm = std::upper_bound(xs,xe,*ym,comp);
    } else {
      xm = xs+(xe-xs)/2;
      ym = std::lower_bound(ys,ye,*xm,comp);
    }
#pragma omp task untied mergeable firstprivate(xs,xm,ys,ym,zs,destroy,comp)
    parallel_move_merge( xs, xm, ys, ym, zs, destroy, comp, cutoff );
    zs += (xm-xs) + (ym-ys);
    xs = xm;
    ys = ym;
  }
  serial_move_merge( xs, xe, ys, ye, zs, comp );
  if( destroy ) {
    serial_destroy( xs, xe );
    serial_destroy( ys, ye );
  }
#pragma omp taskwait
}

// Sorts [xs,xe), where zs[0:xe-xs) is temporary buffer supplied by caller.
// Result is in [xs,xe) if inplace==true, otherwise in [zs,zs+(xe-xs))
template <typename RandomAccessIterator1,
          typename RandomAccessIterator2,
          typename Compare>
void parallel_stable_sort_aux(RandomAccessIterator1 xs, RandomAccessIterator1 xe,
                              RandomAccessIterator2 zs,
                              int inplace, Compare comp,
                              ssize_t cutoff) {
  if((xe - xs) <= cutoff) {
    stable_sort_base_case(xs, xe, zs, inplace, comp);
  } else {
    RandomAccessIterator1 xm = xs + (xe-xs)/2;
    RandomAccessIterator2 zm = zs + (xm-xs);
    RandomAccessIterator2 ze = zs + (xe-xs);
#pragma omp task
    parallel_stable_sort_aux( xs, xm, zs, !inplace, comp, cutoff );
    parallel_stable_sort_aux( xm, xe, zm, !inplace, comp, cutoff );
#pragma omp taskwait
    if( inplace )
      parallel_move_merge( zs, zm, zm, ze, xs, inplace==2, comp, cutoff );
    else
      parallel_move_merge( xs, xm, xm, xe, zs, false, comp, cutoff );
  }
}

} // namespace internal

template<typename RandomAccessIterator, typename Compare>
void parallel_stable_sort(RandomAccessIterator xs, RandomAccessIterator xe,
                          Compare comp) {
  auto n = xe - xs;
  auto t = omp_get_max_threads();
  auto cutoff = n / t;
  if (cutoff < 2) cutoff = 2;
  typedef typename std::iterator_traits<RandomAccessIterator>::value_type T;
  internal::raw_buffer z(size_t(n) * sizeof(T));
#pragma omp parallel
#pragma omp master
  internal::parallel_stable_sort_aux( xs, xe, static_cast<T*>(z.get()), 2, comp, cutoff );
}

} // namespace pss

#endif
