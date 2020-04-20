/*
ECL-CC code: ECL-CC is a connected components algorithm. The CUDA
implementation thereof is very fast. It operates on graphs stored in
binary CSR format.

Copyright (c) 2017, Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted for academic, research, experimental, or personal use provided
that the following conditions are met:

   * Redistributions of source code must retain the above copyright notice,
     this list of conditions, and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions, and the following disclaimer in the documentation
     and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

For all other uses, please contact the Office for Commercialization and Industry
Relations at Texas State University <http://www.txstate.edu/ocir/>.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Jayadharini Jaiganesh and Martin Burtscher
*/


#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

#include "ECL.hpp"

static const int Device = 0;
static const int ThreadsPerBlock = 256;
static const int warpsize = 32;

static __device__ int topL, posL, topH, posH;

/* initialize with first smaller neighbor ID */

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void init(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  for (int v = from; v < nodes; v += incr) {
    const int beg = nidx[v];
    const int end = nidx[v + 1];
    int m = v;
    int i = beg;
    while ((m == v) && (i < end)) {
      m = min(m, nlist[i]);
      i++;
    }
    nstat[v] = m;
  }

  if (from == 0) {topL = 0; posL = 0; topH = nodes - 1; posH = nodes - 1;}
}

/* intermediate pointer jumping */

static inline __device__ int representative(const int idx, int* const __restrict__ nstat)
{
  int curr = nstat[idx];
  if (curr != idx) {
    int next, prev = idx;
    while (curr > (next = nstat[curr])) {
      nstat[prev] = next;
      prev = curr;
      curr = next;
    }
  }
  return curr;
}

/* process low-degree vertices at thread granularity and fill worklists */

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void compute1(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat, int* const __restrict__ wl)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  for (int v = from; v < nodes; v += incr) {
    const int vstat = nstat[v];
    if (v != vstat) {
      const int beg = nidx[v];
      const int end = nidx[v + 1];
      int deg = end - beg;
      if (deg > 16) {
        int idx;
        if (deg <= 352) {
          idx = atomicAdd(&topL, 1);
        } else {
          idx = atomicAdd(&topH, -1);
        }
        wl[idx] = v;
      } else {
        int vstat = representative(v, nstat);
        for (int i = beg; i < end; i++) {
          const int nli = nlist[i];
          if (v > nli) {
            int ostat = representative(nli, nstat);
            bool repeat;
            do {
              repeat = false;
              if (vstat != ostat) {
                int ret;
                if (vstat < ostat) {
                  if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) {
                    ostat = ret;
                    repeat = true;
                  }
                } else {
                  if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) {
                    vstat = ret;
                    repeat = true;
                  }
                }
              }
            } while (repeat);
          }
        }
      }
    }
  }
}

/* process medium-degree vertices at warp granularity */

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void compute2(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat, const int* const __restrict__ wl)
{
  const int lane = threadIdx.x % warpsize;

  int idx;
  if (lane == 0) idx = atomicAdd(&posL, 1);
  idx = __shfl_sync(0xFFFFFFFF, idx, 0);
  while (idx < topL) {
    const int v = wl[idx];
    int vstat = representative(v, nstat);
    for (int i = nidx[v] + lane; i < nidx[v + 1]; i += warpsize) {
      const int nli = nlist[i];
      if (v > nli) {
        int ostat = representative(nli, nstat);
        bool repeat;
        do {
          repeat = false;
          if (vstat != ostat) {
            int ret;
            if (vstat < ostat) {
              if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) {
                ostat = ret;
                repeat = true;
              }
            } else {
              if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) {
                vstat = ret;
                repeat = true;
              }
            }
          }
        } while (repeat);
      }
    }
    if (lane == 0) idx = atomicAdd(&posL, 1);
    idx = __shfl_sync(0xFFFFFFFF, idx, 0);
  }
}

/* process high-degree vertices at block granularity */

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void compute3(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat, const int* const __restrict__ wl)
{
  __shared__ int vB;
  if (threadIdx.x == 0) {
    const int idx = atomicAdd(&posH, -1);
    vB = (idx > topH) ? wl[idx] : -1;
  }
  __syncthreads();
  while (vB >= 0) {
    const int v = vB;
    __syncthreads();
    int vstat = representative(v, nstat);
    for (int i = nidx[v] + threadIdx.x; i < nidx[v + 1]; i += ThreadsPerBlock) {
      const int nli = nlist[i];
      if (v > nli) {
        int ostat = representative(nli, nstat);
        bool repeat;
        do {
          repeat = false;
          if (vstat != ostat) {
            int ret;
            if (vstat < ostat) {
              if ((ret = atomicCAS(&nstat[ostat], ostat, vstat)) != ostat) {
                ostat = ret;
                repeat = true;
              }
            } else {
              if ((ret = atomicCAS(&nstat[vstat], vstat, ostat)) != vstat) {
                vstat = ret;
                repeat = true;
              }
            }
          }
        } while (repeat);
      }
    }
    if (threadIdx.x == 0) {
      const int idx = atomicAdd(&posH, -1);
      vB = (idx > topH) ? wl[idx] : -1;
    }
    __syncthreads();
  }
}

/* link all vertices to sink */

static __global__ __launch_bounds__(ThreadsPerBlock, 2048 / ThreadsPerBlock)
void flatten(const int nodes, const int* const __restrict__ nidx, const int* const __restrict__ nlist, int* const __restrict__ nstat)
{
  const int from = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  const int incr = gridDim.x * ThreadsPerBlock;

  for (int v = from; v < nodes; v += incr) {
    int next, vstat = nstat[v];
    const int old = vstat;
    while (vstat > (next = nstat[vstat])) {
      vstat = next;
    }
    if (old != vstat) nstat[v] = vstat;
  }
}

void computeCC(const int nodes, const int edges, const int* const __restrict__ nidx_d, const int* const __restrict__ nlist_d, int* const __restrict__ nstat_d)
{
  cudaSetDevice(Device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, Device);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: there is no CUDA capable device\n\n");  exit(-1);}
  const int SMs = deviceProp.multiProcessorCount;
  const int mTSM = deviceProp.maxThreadsPerMultiProcessor;

  int* wl_d;

  if (cudaSuccess != cudaMalloc((void **)&wl_d, nodes * sizeof(int))) {fprintf(stderr, "ERROR: could not allocate wl_d,\n\n");  exit(-1);}

  cudaFuncSetCacheConfig(init, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(compute1, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(compute2, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(compute3, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(flatten, cudaFuncCachePreferL1);

  const int blocks = SMs * mTSM / ThreadsPerBlock;
  init<<<blocks, ThreadsPerBlock>>>(nodes, nidx_d, nlist_d, nstat_d);
  compute1<<<blocks, ThreadsPerBlock>>>(nodes, nidx_d, nlist_d, nstat_d, wl_d);
  compute2<<<blocks, ThreadsPerBlock>>>(nodes, nidx_d, nlist_d, nstat_d, wl_d);
  compute3<<<blocks, ThreadsPerBlock>>>(nodes, nidx_d, nlist_d, nstat_d, wl_d);
  flatten<<<blocks, ThreadsPerBlock>>>(nodes, nidx_d, nlist_d, nstat_d);

  cudaFree(wl_d);
}
