# Mojo's matmul example
Programming languages targeting fast numerical code are interesting to me. So of course, I was interested in mojo when it was announced. My first thought was "show me some disassembly!". My second thought was "show me a comparison to something other than python!".

When the SDK was released, I started playing with it. If you're like me and wanted to see some generated code and comparisons to high performance languages, this doc of notes is for you. I understand the language is very new and still in the early stages. You should understand this too before reading this document.

I started with the [matmul.mojo](https://github.com/modularml/mojo/blob/main/examples/matmul.mojo) example.

Update 9/12/2023: I was able to get `perf` working (I'm on Windows Subsystem for Linux, it was non-obvious). It allowed me to find and resolve a number of distracting issues. I've updated the doc accordingly.

## Differences from matrix.mojo

### Random initialization
I moved the `rand` initialization of `Matrix.data`, to avoid this overhead when using `Matrix` for temporary tile data.

### Parallelism
I removed thread parallelism from all of the implementations. Thread parallelism isn't very interesting. Of course, it is very useful, but it is mostly orthogonal to programming language and code quality, and in my opinion, should be the last optimization to make, after we've maximized the utilization of one core.

This gives the following results:
```
Throughput of a 512x512 matrix multiplication in Mojo using a naive algorithm:
7.8623165681034024 GFLOP/s,  34.142031000000003  ms
Throughput of a 512x512 matrix multiplication in Mojo using vectorization:
33.97651373601407 GFLOP/s,  7.9006179999999997  ms
Throughput of a 512x512 matrix multiplication in Mojo using the stdlib `vectorize`:
39.004071057077041 GFLOP/s,  6.8822420000000006  ms
Throughput of a 512x512 {vectorized + not_parallelized} matrix multiplication in Mojo:
38.475479493228477 GFLOP/s,  6.9767929999999998  ms
Throughput of a 512x512 {tiled + vectorized + not_parallelized} matrix multiplication in Mojo:
23.756592579311008 GFLOP/s,  11.299409000000001  ms
Throughput of a 512x512 {tiled + unrolled + vectorized + not_parallelized} matrix multiplication in Mojo:
26.118186111669495 GFLOP/s,  10.277721999999999  ms
```
It's a bit interesting, it looks like the overhead from the various tiling and unrolling splits actually slow things down a little, without parallelism to hide it.

## Optimization strategy
The next thing I wanted to do was try a different strategy. In order to describe the strategies, let's first describe the naive algorithm with pseudocode:
```
for i:  // C.rows
  for j:  // C.cols
    for k:  // A.cols
      C[i, j] += A[i, k] * B[k, j]
```
In words: starting from a textbook definition of matrix multiplication, reorder the loops over j and k, which allows j to vectorize cleanly, and avoid accessing columns of memory in the inner loop.

The strategy for fast computation used in the mojo example is to:
- Parallelize i (disabled as mentioned above)
- Tile `[j, k]` into tiles of `[nelts * tile_size, tile_size]`
- Vectorize and unroll within each row of the tile

Or in pseudo-code:
```
for i:                           // C.rows
  for jo:                        // C.cols / (nelts * tile_size)
    for ko:                      // A.cols / tile_size
      unrolled for k:            // tile_size
        vectorize_unroll for j:  // nelts * tile_size
          C[i, jo + j] += A[i, ko + k] * B[ko + k, jo + j]
```
Let's call this the "tiling j-k" approach

In my experience, the best strategy for a simple but fast matrix multiply is something more like this:
- Tile `[i, j]` into tiles of `[tile_rows, tile_cols]`, where `tile_cols` is a multiple of the SIMD width, and `tile_rows * tile_cols` is tuned to avoid using too many registers.
- Vectorize and unroll the inner loops over i and j.

Or in psuedo-code:
```
for io:                   // C.rows / tile_rows
  for jo:                 // C.cols / tile_cols
    for k:                // A.cols
      unroll for i:       // tile_rows
        vectorize for j:  // tile_cols
          C[io + i, jo + j] += A[io + i, k] * B[k, jo + j]
```
This strategy is designed such that the accumulators for `C` can be kept in registers, and we only need to read `tile_rows + tile_cols` values of input to compute `tile_rows * tile_cols` values of output. Let's call this the "tiling i-j" approach

### C++ version

Both of these can be nicely implemented in C++ with some helpers from [my array library](https://github.com/dsharlet/array). First, the mojo strategy:
```
template <typename T>
NOINLINE void mojo_tiling_jk(const_matrix_ref<T> A, const_matrix_ref<T> B, matrix_ref<T> C) {
  // Adjust this depending on the target architecture. For AVX2,
  // vectors are 256-bit.
  constexpr index_t vector_size = 32 / sizeof(T);

  // We want the tiles to be as big as possible without spilling any
  // of the accumulator registers to the stack.
  constexpr index_t tile_rows = 4;
  constexpr index_t tile_cols = vector_size * 4;

  for (auto i : C.i()) {
    for (auto jo : split<tile_cols>(C.j())) {
      auto C_ijo = C(i, jo);

      T buffer[tile_cols] = {0};
      auto accumulator = make_array_ref(buffer, make_compact(C_ijo.shape()));

      for (auto ko : split<tile_rows>(A.j()))
        for (auto k : ko)
          for (auto j : jo)
            accumulator(j) += A(i, k) * B(k, j);

      for (auto j : jo)
        C_ijo(j) = accumulator(j);
    }
  }
}
```
And my preferred strategy:
```
template <typename T>
NOINLINE void tiling_ij(const_matrix_ref<T> A, const_matrix_ref<T> B, matrix_ref<T> C) {
  // Adjust this depending on the target architecture. For AVX2,
  // vectors are 256-bit.
  constexpr index_t vector_size = 32 / sizeof(T);

  // We want the tiles to be as big as possible without spilling any
  // of the accumulator registers to the stack.
  constexpr index_t tile_rows = 4;
  constexpr index_t tile_cols = vector_size * 3;

  for (auto io : split<tile_rows>(C.i())) {
    for (auto jo : split<tile_cols>(C.j())) {
      // Make a reference to this tile of the output.
      auto C_ijo = C(io, jo);
      // Define an accumulator buffer.
      T buffer[tile_rows * tile_cols] = {0};
      auto accumulator = make_array_ref(buffer, make_compact(C_ijo.shape()));

      // Perform the matrix multiplication for this tile.
      for (index_t k : A.j())
        for (index_t i : C_ijo.i())
          for (index_t j : C_ijo.j())
            accumulator(i, j) += A(i, k) * B(k, j);

      // Copy the accumulators to the output.
      for (index_t i : C_ijo.i())
        for (index_t j : C_ijo.j())
          C_ijo(i, j) = accumulator(i, j);
    }
  }
}
```

This runs in 3.4ms on my machine (for 512x512 matrices, the same as the mojo examples). Recall the best mojo example ran in 6.9ms, about 2x slower.

For reference, the mojo strategy implemented in C++ runs in 10.6ms. This is very close to the mojo version of the same strategy!

### Mojo output tiling
Here is my attempt at replicating this strategy in mojo:
```
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
```
Unfortunately, this runs in ~24ms. Looking at the disassembly, the inner loop looks like this:
```
   ...
   1a43a:       48 8b 5a 10             mov    0x10(%rdx),%rbx
   1a43e:       4c 8b 2a                mov    (%rdx),%r13
   1a441:       48 0f af d9             imul   %rcx,%rbx
   1a445:       4c 01 f3                add    %r14,%rbx
   1a448:       c4 c1 7c 10 44 9d 00    vmovups 0x0(%r13,%rbx,4),%ymm0
   1a44f:       48 8b 5e 10             mov    0x10(%rsi),%rbx
   1a453:       4c 8b 2e                mov    (%rsi),%r13
   1a456:       49 0f af d8             imul   %r8,%rbx
   1a45a:       48 01 cb                add    %rcx,%rbx
   1a45d:       48 ff c1                inc    %rcx
   1a460:       c4 c2 7d 18 4c 9d 00    vbroadcastss 0x0(%r13,%rbx,4),%ymm1
   1a467:       48 8b 5f 10             mov    0x10(%rdi),%rbx
   1a46b:       4c 8b 2f                mov    (%rdi),%r13
   1a46e:       49 0f af d8             imul   %r8,%rbx
   1a472:       4c 01 f3                add    %r14,%rbx
   1a475:       c4 c2 7d a8 4c 9d 00    vfmadd213ps 0x0(%r13,%rbx,4),%ymm0,%ymm1
   1a47c:       c4 c1 7c 11 4c 9d 00    vmovups %ymm1,0x0(%r13,%rbx,4)
   1a483:       4c 8b 6c 24 f8          mov    -0x8(%rsp),%r13
   1a488:       49 39 cd                cmp    %rcx,%r13
   1a48b:       0f 85 8f fb ff ff       jne    1a020 <$matrix::matmul_tile_output($matrix::Matrix,$matrix::Matrix,$matrix::Matrix,$runtime::$llcl::Runtime)+0x110>
```
This is cropped for size, there are 4x4 = 16 of these `vfmadd213ps` sequences, as we should expect, i.e. the inner loop is 16x bigger than shown here. There's simply a ton of overhead. It looks like there's a lot of address arithmetic that isn't being simplified and lifted out of these inner loops.

In contrast, here is the C++ version from above:
```
    8240:       c4 62 7d 18 24 9e       vbroadcastss (%rsi,%rbx,4),%ymm12
    8246:       c4 21 7c 10 6c aa c0    vmovups -0x40(%rdx,%r13,4),%ymm13
    824d:       c4 21 7c 10 74 aa e0    vmovups -0x20(%rdx,%r13,4),%ymm14
    8254:       c4 21 7c 10 3c aa       vmovups (%rdx,%r13,4),%ymm15
    825a:       4d 01 f5                add    %r14,%r13
    825d:       c4 42 15 b8 dc          vfmadd231ps %ymm12,%ymm13,%ymm11
    8262:       c4 42 0d b8 d4          vfmadd231ps %ymm12,%ymm14,%ymm10
    8267:       c4 42 05 b8 cc          vfmadd231ps %ymm12,%ymm15,%ymm9
    826c:       c4 42 7d 18 24 9c       vbroadcastss (%r12,%rbx,4),%ymm12
    8272:       c4 42 15 b8 c4          vfmadd231ps %ymm12,%ymm13,%ymm8
    8277:       c4 c2 0d b8 fc          vfmadd231ps %ymm12,%ymm14,%ymm7
    827c:       c4 c2 05 b8 f4          vfmadd231ps %ymm12,%ymm15,%ymm6
    8281:       c4 62 7d 18 24 9f       vbroadcastss (%rdi,%rbx,4),%ymm12
    8287:       c4 c2 15 b8 ec          vfmadd231ps %ymm12,%ymm13,%ymm5
    828c:       c4 c2 0d b8 e4          vfmadd231ps %ymm12,%ymm14,%ymm4
    8291:       c4 c2 05 b8 dc          vfmadd231ps %ymm12,%ymm15,%ymm3
    8296:       c4 62 7d 18 64 9d 00    vbroadcastss 0x0(%rbp,%rbx,4),%ymm12
    829d:       48 ff c3                inc    %rbx
    82a0:       c4 c2 1d b8 d5          vfmadd231ps %ymm13,%ymm12,%ymm2
    82a5:       c4 c2 1d b8 ce          vfmadd231ps %ymm14,%ymm12,%ymm1
    82aa:       c4 c2 05 b8 c4          vfmadd231ps %ymm12,%ymm15,%ymm0
    82af:       49 39 da                cmp    %rbx,%r10
    82b2:       75 8c                   jne    8240 <_Z21multiply_reduce_tilesIfEvN3nda9array_refIKT_NS0_5shapeIJNS0_3dimILln9ELln9ELln9EEENS5_ILln9ELln9ELl1EEEEEEEES9_NS1_IS2_S8_EE+0x350>
```
This is the *entire* inner loop, I didn't need to truncate it for size as I did with the mojo code.

### Why the difference?
So what is going on, can we fix this? Some possible issues:
1. In C++, I had to store the tile of accumulators in a local temporary in order to get the compiler to keep them in registers.
2. In C++, I use a 3x4 tile of registers, in mojo I'm using 4x4. Using 4x3 crashes the mojo example, because `tile[]` can't handle tile sizes that don't divide the dimensions. (The C++ version uses some tricks to avoid this, see [array's README.md) for more information](https://github.com/dsharlet/array#slicing-cropping-and-splitting).

(2) is easy to test, we can just tweak the C++ version to use 4x4 tiles. As expected, it spills a few of the accumulator registers, and gets a bit slower, running in 5.5ms (vs. 3.4ms before), still far from 20ms in mojo.

For (1), here's my attempt to use a local temporary matrix to store the accumulators:
```
fn matmul_tile_output(
  C: Matrix, A: Matrix, B: Matrix, rt: Runtime
):

  @parameter
  fn calc_tile[tile_j: Int, tile_i: Int](jo: Int, io: Int):

    var tile = Matrix(tile_i, tile_j)
    tile.zero()

    for k in range(0, A.cols):
      @parameter
      fn calc_tile_row[i: Int]():

        @parameter
        fn calc_tile_cols[nelts: Int](j: Int):
          tile.store[nelts](i, j, tile.load[nelts](i, j) + A[io + i, k] * B.load[nelts](k, jo + j))

        vectorize_unroll[nelts, tile_j // nelts, calc_tile_cols](tile_j)

      unroll[tile_i, calc_tile_row]()
      
    # Copy the local tile to the output
    for i in range(tile_i):
      for j in range(tile_j):
        C[io + i, jo + j] = tile[i, j]

  alias tile_i = 4
  alias tile_j = nelts*4
  tile[calc_tile, tile_j, tile_i](C.cols, C.rows)
```
Unfortunately, this runs slower, in 26ms. The inner loop actually does look much better now:
```
   ...
   1a277:       c5 fc 28 f5             vmovaps %ymm5,%ymm6
   1a27b:       c4 a2 7d 18 04 96       vbroadcastss (%rsi,%r10,4),%ymm0
   1a281:       c4 c2 7d b8 21          vfmadd231ps (%r9),%ymm0,%ymm4
   1a286:       c5 fc 11 a5 60 01 00    vmovups %ymm4,0x160(%rbp)
   1a28d:       00
   1a28e:       c5 fc 28 ec             vmovaps %ymm4,%ymm5
   1a292:       c4 a2 7d 18 04 92       vbroadcastss (%rdx,%r10,4),%ymm0
   1a298:       c4 c2 7d b8 59 a0       vfmadd231ps -0x60(%r9),%ymm0,%ymm3
   1a29e:       c5 fc 11 9d 80 01 00    vmovups %ymm3,0x180(%rbp)
   1a2a5:       00
   1a2a6:       c5 fc 28 e3             vmovaps %ymm3,%ymm4
   1a2aa:       c4 a2 7d 18 04 92       vbroadcastss (%rdx,%r10,4),%ymm0
   1a2b0:       c4 c2 7d b8 51 c0       vfmadd231ps -0x40(%r9),%ymm0,%ymm2
   1a2b6:       c5 fc 11 95 a0 01 00    vmovups %ymm2,0x1a0(%rbp)
   1a2bd:       00
   1a2be:       c5 fc 28 da             vmovaps %ymm2,%ymm3
   1a2c2:       c4 a2 7d 18 04 92       vbroadcastss (%rdx,%r10,4),%ymm0
   1a2c8:       c4 c2 7d b8 49 e0       vfmadd231ps -0x20(%r9),%ymm0,%ymm1
   1a2ce:       c5 fc 11 8d c0 01 00    vmovups %ymm1,0x1c0(%rbp)
   1a2d5:       00
   1a2d6:       c5 fc 28 d1             vmovaps %ymm1,%ymm2
   1a2da:       c5 fc 10 4c 24 70       vmovups 0x70(%rsp),%ymm1
   1a2e0:       c4 a2 7d 18 04 92       vbroadcastss (%rdx,%r10,4),%ymm0
   1a2e6:       4d 89 da                mov    %r11,%r10
   1a2e9:       c4 c2 7d b8 09          vfmadd231ps (%r9),%ymm0,%ymm1
   1a2ee:       49 01 c9                add    %rcx,%r9
   1a2f1:       c5 fc 11 4c 24 70       vmovups %ymm1,0x70(%rsp)
   1a2f7:       c5 fc 28 ca             vmovaps %ymm2,%ymm1
   1a2fb:       c5 fc 28 d3             vmovaps %ymm3,%ymm2
   1a2ff:       c5 fc 28 dc             vmovaps %ymm4,%ymm3
   1a303:       c5 fc 28 e5             vmovaps %ymm5,%ymm4
   1a307:       c5 fc 28 ee             vmovaps %ymm6,%ymm5
   1a30b:       c5 fc 28 f7             vmovaps %ymm7,%ymm6
   1a30f:       c5 7c 29 c7             vmovaps %ymm8,%ymm7
   1a313:       c4 41 7c 28 c1          vmovaps %ymm9,%ymm8
   1a318:       c4 41 7c 28 ca          vmovaps %ymm10,%ymm9
   1a31d:       c4 41 7c 28 d3          vmovaps %ymm11,%ymm10
   1a322:       c4 41 7c 28 dc          vmovaps %ymm12,%ymm11
   1a327:       c4 41 7c 28 e5          vmovaps %ymm13,%ymm12
   1a32c:       c4 41 7c 28 ee          vmovaps %ymm14,%ymm13
   1a331:       c5 7c 10 b4 24 90 00    vmovups 0x90(%rsp),%ymm14
   1a338:       00 00
   1a33a:       c5 fc 10 44 24 70       vmovups 0x70(%rsp),%ymm0
   1a340:       c5 fc 11 85 e0 01 00    vmovups %ymm0,0x1e0(%rbp)
   1a347:       00
   1a348:       4c 39 d8                cmp    %r11,%rax
   1a34b:       0f 85 1f fe ff ff       jne    1a170 <$matrix::matmul_tile_output($matrix::Matrix,$matrix::Matrix,$matrix::Matrix,$runtime::$llcl::Runtime)+0x260>
```
As before, I've truncated this for size. It looks like storing the accumulators in a local temporary eliminated a lot of the overhead due to address computations, but it is still storing and reloading them on every iteration of k. It's also doing something really weird, rotating all of the registers by one at the end of the inner loop. If not for these two (pretty big) issues, this inner loop would be pretty good!

It's also allocating and freeing the tile on the heap in every tile of output. Moving the tile outside the `calc_tile` helper fixes this:
```
fn matmul_tile_output(
  C: Matrix, A: Matrix, B: Matrix, rt: Runtime
):
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
```
This runs in 7.5ms, finally (roughly) matching the original matmul.mojo examples! But still more than 2x slower than C++. However, this requires poking holes in the `tile` abstraction, which makes the code not nearly as nice. We really need a way to make small cheap stack allocations in mojo to avoid this.

## Update 9/11/2023: using `memory.stack_allocation`
On https://github.com/modularml/mojo/discussions/735, a user pointed me to `memory.stack_allocation`, which seems like what I was looking for. I attempted to use this for the temporary tile. To do this, we need to add a new `MatrixView` type that accepts a pointer to data as input, instead of allocating. Here is this implementation:
```
fn matmul_tile_output_temp_tile_stack(
  C: Matrix, A: Matrix, B: Matrix, rt: Runtime
):
  @parameter
  fn calc_tile[tile_j: Int, tile_i: Int](jo: Int, io: Int):

    var temp = MatrixView(tile_i, tile_j, stack_allocation[tile_i * tile_j, DType.float32]())
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
```
This gives the best performance yet, at 5.7ms, within 2x of my C++ code, and 25-50% faster than any of the official mojo matmul examples. The inner loop assembly is looking decent at this point, it seems like the compiler is just not lifting some (or all) of the loads and stores out of the loop over rows:
```
   1e9d9:       c4 e2 7d 18 24 8a       vbroadcastss (%rdx,%rcx,4),%ymm4
   1e9df:       48 89 d9                mov    %rbx,%rcx
   1e9e2:       c5 7c 11 94 24 a0 00    vmovups %ymm10,0xa0(%rsp)
   1e9e9:       00 00
   1e9eb:       c4 41 7c 28 d3          vmovaps %ymm11,%ymm10
   1e9f0:       c4 41 7c 28 dc          vmovaps %ymm12,%ymm11
   1e9f5:       c4 41 7c 28 e5          vmovaps %ymm13,%ymm12
   1e9fa:       c4 41 7c 28 ee          vmovaps %ymm14,%ymm13
   1e9ff:       c4 41 7c 28 f7          vmovaps %ymm15,%ymm14
   1ea04:       c5 7c 28 f8             vmovaps %ymm0,%ymm15
   1ea08:       c5 fc 10 84 24 20 01    vmovups 0x120(%rsp),%ymm0
   1ea0f:       00 00
   1ea11:       c4 e2 5d b8 fa          vfmadd231ps %ymm2,%ymm4,%ymm7
   1ea16:       c4 e2 75 b8 f4          vfmadd231ps %ymm4,%ymm1,%ymm6
   1ea1b:       c4 62 5d b8 cd          vfmadd231ps %ymm5,%ymm4,%ymm9
   1ea20:       c4 62 5d b8 c3          vfmadd231ps %ymm3,%ymm4,%ymm8
   1ea25:       c5 fc 11 7c 24 40       vmovups %ymm7,0x40(%rsp)
   1ea2b:       c5 fc 11 b4 24 c0 00    vmovups %ymm6,0xc0(%rsp)
   1ea32:       00 00
   1ea34:       c5 fc 10 bc 24 00 01    vmovups 0x100(%rsp),%ymm7
   1ea3b:       00 00
   1ea3d:       c5 fc 10 b4 24 e0 00    vmovups 0xe0(%rsp),%ymm6
   1ea44:       00 00
   1ea46:       c5 fc 10 54 24 40       vmovups 0x40(%rsp),%ymm2
   1ea4c:       c5 fc 10 8c 24 c0 00    vmovups 0xc0(%rsp),%ymm1
   1ea53:       00 00
   1ea55:       48 39 dd                cmp    %rbx,%rbp
   1ea58:       0f 85 92 fe ff ff       jne    1e8f0 <$matmul::matmul_tile_output_temp_tile_stack_lifted($matmul::Matrix,$matmul::Matrix,$matmul::Matrix,$runtime::$llcl::Runtime)+0x230>
```
There are 4 more of these sequences, one for each row.

## Summary and performance observations
To summarize the data above, here are 3 basic implementations, measured in both Mojo and C++ (all times in milliseconds, 512x512 matrices):

| Strategy           | Mojo | C++  |
|--------------------|------|------|
| "Naive"            | 33   | 8.7  |
| Tiling j-k         | 10   | 10.6 |
| Tiling i-j         | 5.7  | 3.4  |

# Thoughts on Mojo so far

When I first saw mojo, I liked the idea. I want to be able to write code expressively, but still get good performance, without relying on too much automatic compiler magic. Being able to be explicit about tiling, splitting loops, unrolling and vectorizing, etc. looked like a big step forward that previously only existed in niche languages like [Halide](https://halide-lang.org) or in messy ways like SIMD intrinsics.

After getting hands on with it for a few days, here are my observations and thoughts so far:
- The `vectorize` (and `vectorize_unroll`) abstractions are leaky, you still have to write things like `C.load[nelts](...)` to get a vector load of `nelts`. 
  - This is bug prone, it assumes that the stride of the elements in the dimension being vectorized is one. This shouldn't be an assumption made on the part of the programmer.
  - If `nelts` isn't one of the precise SIMD widths available in the instruction set, the code will fail to compile.
    It should be easy to vectorize by any number of elements, and the compiler should deal with generating multiple vectors worth of code (or partial vectors). In my C++ code above, we just present scalar code to the compiler that is readily vectorized, and it handles dispatching to multiple vectors, or a mix of SSE and AVX vectors, or maybe something more exotic on other architectures.
- There needs to be more explicit control over where allocations go. There needs to be an easy way to put allocations on the stack, and the compiler needs to be good at promoting those to registers when appropriate. Maybe this exists already, I can't find it in the docs (or the [roadmap](https://docs.modular.com/mojo/roadmap.html)). I understand priorities may be different right now, but this one really seems fundamental to making mojo a useful language, and might be very difficult to support while remaining faithful to python.
  - Update: This exists, as described in the above updated section.
- I don't understand why things like address arithmetic aren't being lifted out of inner loops. Assuming mojo is using LLVM as a backend, mojo should be getting optimizations like this for "free". It would be shocking if Chris Lattner *didn't* use his own wildly successful project here...
- Mojo is matching C++ in one case, despite the generated code looking terrible. Interestingly, I can't find anything at all wrong with the C++ generated code, it's very clean and vectorized, profiling does not show time being spent in any unexpected places. I'm not sure what Mojo is doing in this case to do so well.
- The inner loop of the i-j tiling version in mojo just needs a little work from LLVM, and I think it would match the C++ code.
- Composing the higher order functions for vectorization, parallelism, etc. seems difficult. What if I want to tile a function, and then parallelize the outer loop over rows of tiles? This can be made a little less messy, e.g. modifying the `tile` function to be `tile_parallel`, but this still requires a new function for each higher order function to apply to.

Some of these issues seem readily fixable, and it seems like after they are fixed, the inner loop code quality should be comparable to the C++ code.

The last issue of composability of the higher order functions seems more fundamental. I find that when I'm trying to experiment with the strategy for optimization, I need to rewrite significant parts of my code in ways that aren't intuitive to read and understand. And when I do this, I introduce bugs that take me a while to find, and this seems like a more inherent part of the language that can't easily be fixed. I find it much easier to modify the strategy used for code written in C++ using the array library's helpers like `split`. Languages like Halide go even further to enable exploring the schedule space without needing to modify the program (and introduce bugs).

Basically, the following needs to be automated (like the C++ code above) so the programmer doesn't have to fill in all this logic about the bounds of tiles:
```
@parameter
fn calc_tile[tile_w: Int, tile_h: Int](xo: Int, yo: Int)
  ...

parallelize[lambda yo: tile[calc_tile, tile_w, tile_h](a.width, range(yo*tile_h, (yo + 1)*tile_h))](a.height // tile_h)
```
Maybe some kind of partial function application with named variables can handle this. It seems like it will require redesigning the higher order functions to understand `range`s and not just sizes.

## Expressiveness/productivity of mojo vs. alternatives

Mojo really does not seem very expressive. When I compare my mojo code (or the official matmul.mojo examples), I feel it is quite a bit harder to read, understand, and modify than my C++ example above. But I have been happily using C++ for decades, and mojo only for a few days. Perhaps there are ways to use mojo more expressively that I haven't thought of yet.
