# Mojo's matmul example
Programming languages targeting fast numerical code are interesting to me.
So of course, I was interested in mojo when it was announced. My first thought was "show me some disassembly!".
My second thought was "show me a comparison to something other than python!".

When the SDK was released, I started playing with it. 
If you're like me and wanted to see some generated code and comparisons to high performance languages, this doc of notes is for you. I understand the language is very new and still in the early stages. You should understand this too before reading this document.

I started with the [matmul.mojo](https://github.com/modularml/mojo/blob/main/examples/matmul.mojo) example.

## Thoughts on parallelism
I removed thread parallelism from all of the implementations. 
Thread parallelism isn't very interesting. 
Of course, it is very useful, but it is mostly orthogonal to programming language and code quality, and in my opinion, should be the
last optimization to make, after we've maximized the utilization of one core.

This gives the following results:
```
Throughput of a 512x512 matrix multiplication in Mojo using a naive algorithm:
7.9945532238111028 GFLOP/s,  33.577293000000004  ms
Throughput of a 512x512 matrix multiplication in Mojo using vectorization:
32.997195977578151 GFLOP/s,  8.1350990000000003  ms
Throughput of a 512x512 matrix multiplication in Mojo using the stdlib `vectorize`:
33.68763712815938 GFLOP/s,  7.9683670000000006  ms
Throughput of a 512x512 {vectorized + not_parallelized} matrix multiplication in Mojo:
33.038384818528883 GFLOP/s,  8.1249570000000002  ms
Throughput of a 512x512 {tiled + vectorized + not_parallelized} matrix multiplication in Mojo:
24.335731440087024 GFLOP/s,  11.030507  ms
Throughput of a 512x512 {tiled + unrolled + vectorized + not_parallelized} matrix multiplication in Mojo:
26.228654146516149 GFLOP/s,  10.234435  ms
```
It's a bit interesting, it looks like the overhead from the various tiling and unrolling splits actually slow things down a little,
without parallelism to hide it.

## Optimization strategy
The next thing I wanted to do was try a different strategy. In order to describe the strategies, let's first describe the naive algorithm with pseudocode:
```
for i:  // C.rows
  for j:  // C.cols
    for k:  // A.cols
      C[i, j] += A[i, k] * B[k, j]
```
The strategy for fast computation used in the mojo example is to:
- Parallelize i
- Tile `[j, k]` into tiles of `[nelts * tile_size, tile_size]`
- Vectorize and unroll within each row of the tile

In my experience, the best strategy for a simple but fast matrix multiply is something more like this:
- Tile `[i, j]` into tiles of `[tile_rows, tile_cols]`, where `tile_cols` is a multiple of the SIMD width, and `tile_rows * tile_cols` is tuned to avoid using too many registers.
- Vectorize and unroll the inner loops over i and j.

Or in psuedo-code:
```
for io:  // C.rows
  for jo:  // C.cols
    for k:  // A.cols
      for i:
        for j:
          C[io + i, jo + j] += A[io + i, k] * B[k, jo + j]
```
This strategy is designed such that the accumulators for `C` can be kept in registers, and we only need to read `tile_rows + tile_cols` values of input to compute `tile_rows * tile_cols` values of output.

### C++ version of tiling

This can be nicely implemented in C++ with some helpers from [my array library](https://github.com/dsharlet/array):
```
template <typename T>
NOINLINE void multiply_reduce_tiles(const_matrix_ref<T> A, const_matrix_ref<T> B, matrix_ref<T> C) {
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
      for (index_t k : A.j()) {
        for (index_t i : C_ijo.i()) {
          for (index_t j : C_ijo.j()) {
            accumulator(i, j) += A(i, k) * B(k, j);
          }
        }
      }

      // Copy the accumulators to the output.
      for (index_t i : C_ijo.i()) {
        for (index_t j : C_ijo.j()) {
          C_ijo(i, j) = accumulator(i, j);
        }
      }
    }
  }
}
```

This runs in 3.4ms on my machine (for 512x512 matrices, the same as the mojo examples).
Recall the best mojo example ran in 7.97ms, over 2x slower.

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
   1a3ae:       48 8b 5a 10             mov    0x10(%rdx),%rbx
   1a3b2:       4c 8b 2a                mov    (%rdx),%r13
   1a3b5:       48 0f af d9             imul   %rcx,%rbx
   1a3b9:       48 01 eb                add    %rbp,%rbx
   1a3bc:       c4 c1 7c 10 44 9d 00    vmovups 0x0(%r13,%rbx,4),%ymm0
   1a3c3:       48 8b 5e 10             mov    0x10(%rsi),%rbx
   1a3c7:       4c 8b 2e                mov    (%rsi),%r13
   1a3ca:       49 0f af d8             imul   %r8,%rbx
   1a3ce:       48 01 cb                add    %rcx,%rbx
   1a3d1:       c4 c2 7d 18 4c 9d 00    vbroadcastss 0x0(%r13,%rbx,4),%ymm1
   1a3d8:       48 8b 5f 10             mov    0x10(%rdi),%rbx
   1a3dc:       4c 8b 2f                mov    (%rdi),%r13
   1a3df:       49 0f af d8             imul   %r8,%rbx
   1a3e3:       48 01 eb                add    %rbp,%rbx
   1a3e6:       c4 c2 7d a8 4c 9d 00    vfmadd213ps 0x0(%r13,%rbx,4),%ymm0,%ymm1
   1a3ed:       c4 c1 7c 11 4c 9d 00    vmovups %ymm1,0x0(%r13,%rbx,4)
   1a3f4:       48 8b 5a 10             mov    0x10(%rdx),%rbx
   1a3f8:       4c 8b 2a                mov    (%rdx),%r13
   1a3fb:       48 0f af d9             imul   %rcx,%rbx
   1a3ff:       48 01 c3                add    %rax,%rbx
   1a402:       c4 c1 7c 10 44 9d 00    vmovups 0x0(%r13,%rbx,4),%ymm0
   1a409:       48 8b 5e 10             mov    0x10(%rsi),%rbx
   1a40d:       4c 8b 2e                mov    (%rsi),%r13
   1a410:       49 0f af d8             imul   %r8,%rbx
   1a414:       48 01 cb                add    %rcx,%rbx
   1a417:       c4 c2 7d 18 4c 9d 00    vbroadcastss 0x0(%r13,%rbx,4),%ymm1
   1a41e:       48 8b 5f 10             mov    0x10(%rdi),%rbx
   1a422:       4c 8b 2f                mov    (%rdi),%r13
   1a425:       49 0f af d8             imul   %r8,%rbx
   1a429:       48 01 c3                add    %rax,%rbx
   1a42c:       c4 c2 7d a8 4c 9d 00    vfmadd213ps 0x0(%r13,%rbx,4),%ymm0,%ymm1
   1a433:       c4 c1 7c 11 4c 9d 00    vmovups %ymm1,0x0(%r13,%rbx,4)
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
There are 4x4 = 16 of these `vfmadd213ps` sequences, as we should expect. There's simply a ton of overhead. It looks like there's a lot of address arithmetic that isn't being simplified and lifted out of these inner loops.

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
2. In C++, I use a 3x4 tile of registers, in mojo I'm using 4x4. Using 4x3 crashes the mojo example, because `tile[]` can't handle tile sizes that don't divide the dimensions. (The C++ version uses some tricks to avoid this, see [array's README.md) for more information](https://github.com/dsharlet/array#slicing-cropping-and-splitting). We actually could fix this in our local `tile` function, but we really need something similar for `vectorize` too.

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
It looks like storing the accumulators in a local temporary eliminated a lot of the overhead due to address computations, but it is still storing and reloading them on every iteration of k.
It's also doing something really weird, rotating all of the registers by one at the end of the inner loop.
If not for these two issues, this inner loop would be pretty good!

It's also allocating and freeing the tile explicitly in every tile of output. Moving the tile outside the `calc_tile` helper fixes this:
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
This runs in 7.5ms, finally (roughly) matching the original matmul.mojo examples!
However, this requires poking holes in the `tile` abstraction, which makes the code not nearly as nice.
We really need a way to make small cheap stack allocations in mojo to avoid this.

## Varying the size of the matrix
One way to possibly work around some of these issues is to change the size of the matrix being computed.
Making `K` larger would amortize some of the overheads noted above. Unfortunately, this causes one of the official examples to crash!

# Thoughts

When I first saw mojo, I liked the idea. I want to be able to write code expressively, but still get good performance, without relying on too much automatic compiler magic.
Being able to be explicit about tiling, splitting loops, unrolling and vectorizing, etc. looked like a big step forward that previously only existed in niche languages like [https://halide-lang.org](Halide) or in messy ways like SIMD intrinsics.

After getting hands on with it for a few days, here are my thoughts:
- The `vectorize` (and `vectorize_unroll`) abstractions are leaky, you still have to write things like `C.load[nelts](...)` to get a vector load of `nelts`. And if `nelts` isn't one of the precise SIMD widths available in the instruction set, the code will fail to compile.
  Languages like Halide completely hide these details, and you can vectorize by any number of elements, even if it isn't a convenient multiple of the SIMD width. Or, as in my C++ code above, we just present scalar code to the compiler that is readily vectorized, and it handles dispatching to multiple vectors, or a mix of SSE and AVX vectors, or maybe something more exotic on other architectures.
- There needs to be more explicit control over where allocations go. There needs to be an easy way to put allocations on the stack, and the compiler needs to be good at promoting those to registers when appropriate. Maybe this exists already, I can't find it in the docs (or the [roadmap](https://docs.modular.com/mojo/roadmap.html)). I understand priorities may be different right now, but this one really seems fundamental to making mojo a useful language, and might be very difficult to support while remaining faithful to python.
- I don't understand why things like address arithmetic aren't being lifted out of inner loops. Assuming mojo is using LLVM as a backend, mojo should be getting optimizations like this for "free". It would be shocking if Chris Lattner *didn't* use his own wildly successful project here...
- Composing the higher order functions for vectorization, parallelism, etc. seems difficult. What if I want to tile a function, and then parallelize the outer loop over rows of tiles? The official [matmul_tiled_parallelized](https://github.com/modularml/mojo/blob/7e667e951008ade31621dbd37217a562ae82472f/examples/matmul.mojo#L152) example didn't find a better way either. I realized after I rewrote my code that I could make a new `tile_parallel`:
```
# Perform 2D tiling on the iteration space defined by end_x and end_y, and parallelize the rows of tiles.
fn tile_parallel[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
  # Note: this assumes that ends are multiples of the tiles.
  @parameter
  fn row(yo: Int):
    let y = tile_y * yo
    for x in range(0, end_x, tile_x):
      tiled_fn[tile_x, tile_y](x, y)

  parallelize[row](end_y // tile_y)
```
But this still seems like a suboptimal approach. We really ought to be able to compose the higher order functions.

Most of these issues seem readily fixable, and it seems like after they are fixed, the inner loop code quality should be comparable to the C++ code.

## Expressiveness/productivity of mojo vs. alternatives

The last issue of composability seems more fundamental.
I find that when I'm trying to experiment with the strategy for optimization, I need to rewrite significant parts of my code in ways that aren't intuitive to read and understand.
And when I do this, I introduce bugs that take me a while to find, and this seems like a more inherent part of the language that can't easily be fixed.
I find it much easier to modify the strategy used for code written in C++ using the array library's helpers like `split`.
Languages like Halide go even further to enable exploring the schedule space without needing to modify the program (and introduce bugs).

Perhaps this is just an artifact of me being new to the language, but mojo really does not seem very expressive.
When I compare my mojo code (or the official matmul.mojo examples), I feel it is quite a bit easier to read, understand, and modify my C++ example above.
