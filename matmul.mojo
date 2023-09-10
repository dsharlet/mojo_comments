# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# This sample demonstrates how various systems optimizations can be
# applied to a naive matmul implementation in Mojo to gain significant
# performance speedups

from benchmark import Benchmark
from sys.intrinsics import strided_load
from utils.list import VariadicList
from math import div_ceil, min
from memory import memset_zero
from random import rand, random_float64
from sys.info import simdwidthof
from time import now
from algorithm import vectorize, parallelize, vectorize_unroll
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from python.object import PythonObject
from python.python import Python, _destroy_python, _init_python
from runtime.llcl import Runtime


struct Matrix:
    var data: DTypePointer[DType.float32]
    var rows: Int
    var cols: Int

    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[DType.float32].alloc(rows * cols)
        rand(self.data, rows * cols)
        self.rows = rows
        self.cols = cols

    fn __del__(owned self):
        self.data.free()

    fn zero(inout self):
        memset_zero(self.data, self.rows * self.cols)

    @always_inline
    fn __getitem__(self, y: Int, x: Int) -> Float32:
        return self.load[1](y, x)

    @always_inline
    fn __setitem__(self, y: Int, x: Int, val: Float32):
        return self.store[1](y, x, val)

    @always_inline
    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[DType.float32, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    @always_inline
    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[DType.float32, nelts]):
        return self.data.simd_store[nelts](y * self.cols + x, val)


fn matmul_naive(C: Matrix, A: Matrix, B: Matrix, _rt: Runtime):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]


# Mojo has SIMD vector types, we can vectorize the Matmul code as follows.
alias nelts = simdwidthof[DType.float32]()  # The SIMD vector width.


fn matmul_vectorized_0(C: Matrix, A: Matrix, B: Matrix, _rt: Runtime):
    for m in range(C.rows):
        for k in range(A.cols):
            for nv in range(0, C.cols, nelts):
                C.store[nelts](
                    m, nv, C.load[nelts](m, nv) + A[m, k] * B.load[nelts](k, nv)
                )

            # Handle remaining elements with scalars.
            for n in range(nelts * (C.cols // nelts), C.cols):
                C[m, n] += A[m, k] * B[k, n]


# Simplify the code by using the builtin vectorize function
# from Functional import vectorize
fn matmul_vectorized_1(C: Matrix, A: Matrix, B: Matrix, _rt: Runtime):
    for m in range(C.rows):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[nelts, dot](C.cols)
            
# Perform 2D tiling on the iteration space defined by end_x and end_y.
fn dont_parallelize[f: fn(Int) capturing -> None](end_i: Int):
    # Note: this assumes that ends are multiples of the tiles.
    for i in range(0, end_i):
        f(i);

# Parallelize the code by using the builtin parallelize function
# from Functional import parallelize
fn matmul_not_parallelized(C: Matrix, A: Matrix, B: Matrix, rt: Runtime):
    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[nelts, dot](C.cols)

    dont_parallelize[calc_row](C.rows)

    
# Perform 2D tiling on the iteration space defined by end_x and end_y.
fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    # Note: this assumes that ends are multiples of the tiles.
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)

# Perform 2D tiling on the iteration space defined by end_x and end_y.
fn tile_parallel[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    # Note: this assumes that ends are multiples of the tiles.
    @parameter
    fn row(yo: Int):
        let y = tile_y * yo
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)

            
    dont_parallelize[row](end_y // tile_y)
            


# Use the above tile function to perform tiled matmul.
fn matmul_tiled_not_parallelized(C: Matrix, A: Matrix, B: Matrix, rt: Runtime):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):

                @parameter
                fn dot[
                    nelts: Int,
                ](n: Int):
                    C.store[nelts](
                        m,
                        n + x,
                        C.load[nelts](m, n + x)
                        + A[m, k] * B.load[nelts](k, n + x),
                    )

                vectorize[nelts, dot](tile_x)

        # We hardcode the tile factor to be 4.
        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, tile_size](A.cols, C.cols)

    dont_parallelize[calc_row](C.rows)


# Unroll the vectorized loop by a constant factor.
# from Functional import vectorize_unroll
fn matmul_tiled_unrolled_not_parallelized(
    C: Matrix, A: Matrix, B: Matrix, rt: Runtime
):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):

                @parameter
                fn dot[
                    nelts: Int,
                ](n: Int):
                    C.store[nelts](
                        m,
                        n + x,
                        C.load[nelts](m, n + x) + A[m, k] * B.load[nelts](k, n + x),
                    )

                # Vectorize by nelts and unroll by tile_x/nelts
                # Here unroll factor is 4
                vectorize_unroll[nelts, tile_x // nelts, dot](tile_x)

        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, tile_size](A.cols, C.cols)

    dont_parallelize[calc_row](C.rows)
    
# Tile the output, without using any higher order functions.
fn matmul_tile_output_raw(
    C: Matrix, A: Matrix, B: Matrix, rt: Runtime
):
    alias tile_i = 4
    alias tile_j = nelts*4
    for io in range(0, C.rows, tile_i):
        for jo in range(0, C.cols, tile_j):
            # Zero the output tile.
            for i in range(io, io + tile_i):
                for j in range(jo, jo + tile_j):
                    C[i, j] = 0

            for k in range(0, A.cols):
                for i in range(io, io + tile_i):
                    for j in range(jo, jo + tile_j):
                        C[i, j] = C[i, j] + A[i, k] * B[k, j]
  

# Tile the output, using higher order functions, with vectorization and unrolling
fn matmul_tile_output(
    C: Matrix, A: Matrix, B: Matrix, rt: Runtime
):

  @parameter
  fn calc_tile[tile_j: Int, tile_i: Int](jo: Int, io: Int):
    # Zero the output tile.
    for i in range(io, io + tile_i):
      for j in range(jo, jo + tile_j):
        C.store[1](i, j, 0)

    for k in range(0, A.cols):
      @parameter
      fn calc_tile_row[i: Int]():
        @parameter
        fn calc_tile_cols[nelts: Int](j: Int):
          C.store[nelts](io + i, jo + j, C.load[nelts](io + i, jo + j) + A[io + i, k] * B.load[nelts](k, jo + j))

        vectorize_unroll[nelts, tile_j // nelts, calc_tile_cols](tile_j)

      unroll[tile_i, calc_tile_row]()

  alias tile_i = 4
  alias tile_j = nelts*4
  tile[calc_tile, tile_j, tile_i](C.cols, C.rows)

# Tile the output, and parallelize rows of tiles
fn matmul_tile_output_parallel(
    C: Matrix, A: Matrix, B: Matrix, rt: Runtime
):
  @parameter
  fn calc_tile[tile_j: Int, tile_i: Int](jo: Int, io: Int):
    # Zero the output tile.
    for i in range(io, io + tile_i):
      for j in range(jo, jo + tile_j):
        C.store[1](i, j, 0)

    for k in range(0, A.cols):
      @parameter
      fn calc_tile_row[i: Int]():
        @parameter
        fn calc_tile_cols[nelts: Int](j: Int):
          C.store[nelts](io + i, jo + j, C.load[nelts](io + i, jo + j) + A[io + i, k] * B.load[nelts](k, jo + j))

        vectorize_unroll[nelts, tile_j // nelts, calc_tile_cols](tile_j)

      unroll[tile_i, calc_tile_row]()

  alias tile_i = 4
  alias tile_j = nelts*4
  tile_parallel[calc_tile, tile_j, tile_i](C.cols, C.rows)

# Try using a temporary tile instead of accumulating directly in the output.
fn matmul_tile_output_temp_tile(
  C: Matrix, A: Matrix, B: Matrix, rt: Runtime
):

  @parameter
  fn calc_tile[tile_j: Int, tile_i: Int](jo: Int, io: Int):

    var temp = Matrix(tile_i, tile_j)
    temp.zero()

    for k in range(0, A.cols):
      @parameter
      fn calc_tile_row[i: Int]():

        @parameter
        fn calc_tile_cols[nelts: Int](j: Int):
          temp.store[nelts](i, j, temp.load[nelts](i, j) + A[io + i, k] * B.load[nelts](k, jo + j))

        vectorize_unroll[nelts, tile_j // nelts, calc_tile_cols](tile_j)

      unroll[tile_i, calc_tile_row]()
      
    # Copy the local tile to the output
    for i in range(tile_i):
      for j in range(tile_j):
        C[io + i, jo + j] = temp[i, j]

  alias tile_i = 4
  alias tile_j = nelts*4
  tile[calc_tile, tile_j, tile_i](C.cols, C.rows)

# Try lifting the temporary out of the inner tile loop.
fn matmul_tile_output_temp_lifted(
  C: Matrix, A: Matrix, B: Matrix, rt: Runtime
):
  # calc_tile is now dependent on this outside state.
  # Can't parallelize this trivially any more...
  alias tile_i = 4
  alias tile_j = nelts*4
  var temp_tile = Matrix(tile_i, tile_j)

  @parameter
  fn calc_tile[tile_j: Int, tile_i: Int](jo: Int, io: Int):

    temp_tile.zero()

    for k in range(0, A.cols):
      @parameter
      fn calc_tile_row[i: Int]():

        @parameter
        fn calc_tile_cols[nelts: Int](j: Int):
          temp_tile.store[nelts](i, j, temp_tile.load[nelts](i, j) + A[io + i, k] * B.load[nelts](k, jo + j))

        vectorize_unroll[nelts, tile_j // nelts, calc_tile_cols](tile_j)

      unroll[tile_i, calc_tile_row]()
      
    # Copy the local tile to the output
    for i in range(tile_i):
      for j in range(tile_j):
        C[io + i, jo + j] = temp_tile[i, j]

  tile[calc_tile, tile_j, tile_i](C.cols, C.rows)

fn fill_non_zero(A: Matrix):
    for i in range(0, A.rows):
        for j in range(0, A.cols):
            A[i, j] = i + j

@always_inline
fn benchmark[
    func: fn (Matrix, Matrix, Matrix, Runtime) -> None
](M: Int, N: Int, K: Int, base_gflops: Float64, str: String):
    var C = Matrix(M, N)
    C.zero()
    var A = Matrix(M, K)
    var B = Matrix(K, N)
    fill_non_zero(A)
    fill_non_zero(B)

    with Runtime() as rt:

        @always_inline
        @parameter
        fn test_fn():
            _ = func(C, A, B, rt)

        func(C, A, B, rt)
        var C_ref = Matrix(M, N)
        C_ref.zero()
        matmul_naive(C_ref, A, B, rt)
        for i in range(0, M):
            for j in range(0, N):
                if C[i, j] != C_ref[i, j]:
                    print("Mismatch: ", i, ", ", j, ": ", C[i, j], " != ", C_ref[i, j])
                    return

        let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
        # Prevent the matrices from being freed before the benchmark run
        _ = (A, B, C)
        let gflops = ((2 * M * N * K) / secs) / 1e9
        let speedup: Float64 = gflops / base_gflops
        # print(gflops, "GFLOP/s ", secs, " s, ", speedup, " speedup")
        print(str)
        print(gflops, "GFLOP/s, ", secs * 1000, " ms")


fn main():
    # Python
    print("Throughput of a 128x128 matrix multiplication in Python: ")
    let python_gflops = 1
    alias M = 512
    alias N = 512
    alias K = 512
    # Mojo variants
    benchmark[matmul_naive](
        M, N, K,
        python_gflops,
        (
            "Throughput of a 512x512 matrix multiplication in Mojo using a"
            " naive algorithm: "
        ),
    )
    benchmark[matmul_vectorized_0](
        M, N, K,
        python_gflops,
        (
            "Throughput of a 512x512 matrix multiplication in Mojo using"
            " vectorization: "
        ),
    )
    benchmark[matmul_vectorized_1](
        M, N, K,
        python_gflops,
        (
            "Throughput of a 512x512 matrix multiplication in Mojo using the"
            " stdlib `vectorize`: "
        ),
    )
    benchmark[matmul_not_parallelized](
        M, N, K,
        python_gflops,
        (
            "Throughput of a 512x512 {vectorized + not_parallelized} matrix"
            " multiplication in Mojo: "
        ),
    )
    if (K == 512):
        # Crashes for larger K
        benchmark[matmul_tiled_not_parallelized](
            M, N, K,
            python_gflops,
            (
                "Throughput of a 512x512 {tiled + vectorized + not_parallelized} matrix"
                " multiplication in Mojo: "
            ),
        )
        benchmark[matmul_tiled_unrolled_not_parallelized](
            M, N, K,
            python_gflops,
            (
                "Throughput of a 512x512 {tiled + unrolled + vectorized +"
                " not_parallelized} matrix multiplication in Mojo: "
            ),
        )
    benchmark[matmul_tile_output](
        M, N, K,
        python_gflops,
        (
            "Throughput of a 512x512 {tiled output} matrix multiplication in Mojo: "
        ),
    )
    benchmark[matmul_tile_output_parallel](
        M, N, K,
        python_gflops,
        (
            "Throughput of a 512x512 {tiled output, parallelized tiles} matrix multiplication in Mojo: "
        ),
    )
    benchmark[matmul_tile_output_temp_tile](
        M, N, K,
        python_gflops,
        (
            "Throughput of a 512x512 {tiled output, temporary tile} matrix multiplication in Mojo: "
        ),
    )
    benchmark[matmul_tile_output_temp_lifted](
        M, N, K,
        python_gflops,
        (
            "Throughput of a 512x512 {tiled output, temporary tile outside the inner loop} matrix multiplication in Mojo: "
        ),
    )