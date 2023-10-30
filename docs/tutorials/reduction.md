# Tutorial: Reduction

Reduction is an algorithmic operation that doesn't trivially map to GPUs but is
ubiquitous enough to be of general interest while providing an excuse to
introduce some key considerations of designing GPU algorithms and optimizing
them.

_(This article is a rejuvenation and extension of the invaluable work of Mark
Harris. While the author usually approaches the topic starting from a less
naive approach, it is still valuable to rehash some of the original material to
see how much the underlying hardware has changed. This article will go beyond
to demonstrate progress in the state of the art since.)_

## The algorithm

Reduction has many names depending on the domain: in functional programming
it's referred to as
[fold](https://en.wikipedia.org/wiki/Fold_(higher-order_function)),
in C++ it's been signidfied by `std::accumulate` and as of C++17 `std::reduce`.
A reduction takes a range of inputs and "reduces" said range with a binary
operation to a singular/scalar output. Canonically a reduction requires a so
called "zero" element, which bootstraps the algorithm and serves as one of the
initial operands to the binary operation. ("Zero" element more generally is
called [identity or neutral](https://en.wikipedia.org/wiki/Identity_element)
element in group theory, meaning it is an operand which doesn not change the
result.) Typical use cases would be calculating a sum or wanting to normalize
a dataset and looking for the maximum value of a dataset. We'll be exploring
the latter in this tutorial.

FOLD IMAGE

There are multiple variations of reduction to allow parallel processing. One
such is what `std::reduce` does, is to require the user-provided binary
operator to operate on any combination of identity, and input range elements,
even exclusively one or the other. This allows inserting any number of
identities to allow parallel processing and then allows combining the partial
results of parallel execution.

PARALLEL FOLD IMAGE

## Reduction on GPUs

Implementing reductions on GPUs will require a basic understanding of the
{doc}`/reference/programming_model`. The article explores aspects of low-level
optimization best discussed through the {ref}`inherent_tread_model`, and as
such will refrain from using Cooperative Groups.

Synchronizing parallel threads of execution across a GPU will be crucial for
correctness; we can't start combining partial results before they manifest.
Synchronizing _all_ the threads running on a GPU at any given time while
possible, is a costly and/or intricate operation. If not absolutely necessary,
we'll map our parallel algorithm as such that Multi Processors / Blocks can
make independent progress and need not sync often.

### 1. Naive shared reduction

The naive algorithm takes a tree-like shape, where the computational domain is
purposefully distributed among Blocks. In all Blocks all threads participate in
loading data from persistent (from the kernel's perspective) Global Memory into
Shared Memory, performing the tree-like reduction for a single thread to write
that partial result to Global in a location unique to the Block, allowing Block
to make independent progress. Combining these partial results will happen in
subsequent launches of the very same kernel until a scalar result is reached.

NAIVE REDUCTION IMAGE

This approach will require temporary storage based on the number of Blocks
launched, as each will output a scalar partial result. Depending on whether it
is valid to destroy the input, a second temporary storage may be needed, large
enough to store the results of the second kernel launch, or one may simply
reuse the storage of the larger than necessary original input; these
implementations differ ever so slightly and the article will assume the input
may be destroyed.

```c++
std::size_t factor = block_size; // block_size from hipGetDeviceProperties()
auto new_size = [factor](const std::size_t actual)
{
 // Every pass reduces input length by 'factor'. If actual size is not divisible by factor,
 // an extra output element is produced using some number of zero_elem inputs.
 return actual / factor + (actual % factor == 0 ? 0 : 1);
};
```

We'll be feeding `zero_elem` instances to threads that don't have unique inputs
of their own. The backing of double-buffering is allocated is such:

```c++
// Initialize host-side storage
std::vector<unsigned> input(input_count);
std::iota(input.begin(), input.end(), 0);

// Initialize device-side storage
unsigned *front,
         *back;
hipMalloc((void**)&front, sizeof(unsigned) * input_count);
hipMalloc((void**)&back,  sizeof(unsigned) * new_size(input_count));

hipMemcpy(front, input.data(), input.size() * sizeof(unsigned), hipMemcpyHostToDevice);
```

Data is initialized on host and dispatched to the device. Then device-side
reduction may commence. We omit swapping the double-buffer on the last
iteration so the result is in the back-buffer no matter the input size.

```c++
for(uint32_t curr = input_count; curr > 1;)
{
 hipLaunchKernelGGL(
  kernel,
  dim3(new_size(curr)),
  dim3(block_size),
  factor * sizeof(unsigned),
  hipStreamDefault,
  front,
  back,
  kernel_op,
  zero_elem,
  curr);

 curr = new_size(curr);
 if(curr > 1)
  std::swap(front, back);
}
```

This structure will persist throughout all the variations of reduction with
slight modifications to `factor` and shared memory allocation, but primarily
the kernel itself:

```c++
template<typename T, typename F>
__global__ void kernel(
 T* front,
 T* back,
 F op,
 T zero_elem,
 uint32_t front_size)
{
 extern __shared__ T shared[];

 // Overindex-safe read of input
 auto read_global_safe = [&](const uint32_t i)
 {
  return i < front_size ? front[i] : zero_elem;
 };

 const uint32_t tid = threadIdx.x,
                bid = blockIdx.x,
                gid = bid * blockDim.x + tid;

 // Read input from front buffer to shared
 shared[tid] = read_global_safe(gid);
 __syncthreads();

 // Shared reduction
 for(uint32_t i = 1; i < blockDim.x; i *= 2)
 {
  if(tid % (2 * i) == 0)
   shared[tid] = op(shared[tid], shared[tid + i]);
  __syncthreads();
 }

 // Write result from shared to back buffer
 if(tid == 0)
  back[bid] = shared[0];
}
```

While the `tid % (2 * i) == 0` indexing scheme yields correct results, it will
also result in high thread divergence. Thread divergence is when threads in a
warp/wavefront diverge, meaning they'd have to execute different instructions
on a given clock cycle. This manifests easiest using `if/else` branches like
here, but among others could also manifest as thread id dependent `for` loop
lengths. Even though there are less and less active threads participating in
the reduction, warps remain active for longer than necessary.

### 2. Reducing thread divergence

One may reduce divergence by keeping dataflow between memory addresses
identical but reassigning the thread ids.
