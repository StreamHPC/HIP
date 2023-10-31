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
for (uint32_t curr = input_count; curr > 1;)
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
	if (curr > 1)
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
	for (uint32_t i = 1; i < blockDim.x; i *= 2)
	{
		if (tid % (2 * i) == 0)
			shared[tid] = op(shared[tid], shared[tid + i]);
		__syncthreads();
	}

	// Write result from shared to back buffer
	if (tid == 0)
		back[bid] = shared[0];
}
```

While the `tid % (2 * i) == 0` indexing scheme yields correct results, it will
also result in high thread divergence. Thread divergence is when threads in a
warp/wavefront diverge, meaning they'd have to execute different instructions
on a given clock cycle. This manifests easiest using `if/else` branches like
here, but among others could also manifest as thread id dependent `for` loop
lengths. Even though there are less and less active threads participating in
the reduction, warps remain active (at least one lane in a warp hits the `if`
branch) for longer than necessary.

### 2. Reducing thread divergence

One may reduce divergence by keeping dataflow between memory addresses
identical but reassigning the thread ids.

REDUCED THREAD DIVERGENCE IMAGE

```{note}
For those less proficient in reading Git diffs, the coming code segments show
changes between versions of a file. Lines highlighted in red are removed or
changed while lines highlighted green are being introduced.
```

```diff
// Shared reduction
for (uint32_t i = 1; i < blockDim.x; i *= 2)
{
-	if (tid % (2 * i) == 0)
-		shared[tid] = op(shared[tid], shared[tid + i]);
+	if (uint32_t j = 2 * i * tid; j < blockDim.x)
+		shared[j] = op(shared[j], shared[j + i]);
	__syncthreads();
}
```

This way inactive threads start accumulating uniformly toward the higher thread
id index range and may uniformly skip to `__syncthreads()`. This however
introduces a new issue: bank conflicts.

### 3. Resolving bank conflicts

Shared memory on both AMD and NVIDIA is implemented in hardware by storage
which is organized into banks of various sizes. On AMD hardware the name of
this hardware element is LDS, Local Data Share. A truthful mental model of
shared memory is to think of it as a striped 2-dimensional range of memory.

SHARED MEMORY BANKS IMAGE

Shared memory bank count, width and depth depend on the architecture at hand.
A bank conflict occurs when different threads in a warp/wavefront access the
same bank in the same operation. In this case, the "hardware prevents the
attempted concurrent accesses to the same bank by turning them into serial
accesses".

- ["AMD Instinct MI200" Instruction Set Architecture, Chapter 11.1](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf)
- ["RDNA 2" Instruction Set Architecture, Chapter 10.1](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)

A notable exception is when the shared read uniformly evaluates to the same
address across the entire warp/wavefront turning it into a broadcast. A
better change naive implementation is to have not only the activity of
threads form continous ranges but their memory accesses too.

```diff
// Shared reduction
-for (uint32_t i = 1; i < blockDim.x; i *= 2)
-{
-	if (tid % (2 * i) == 0)
+for (uint32_t i = blockDim.x / 2; i != 0; i /= 2)
+{
+	if (tid < i)
		shared[tid] = op(shared[tid], shared[tid + i]);
	__syncthreads();
}
```

BANK CONFLICT FREE THREAD DIVERGENCE IMAGE

```{tip}
It is easiest to avoid bank conflicts if one can read shared memory in a
coalesced manner, meaning reads/writes of each lane in a warp evaluate to
consequtive locations. Additional requirements must be met detailed more
thoroughly in the linked ISA documents, but having simple read/write patterns
help reason about bank conflicts.
```

### 4. Utilize upper half of the block

The previous implementation was free of low-level GPGPU-specific anti-patterns,
however it does still exhibit a few common shortcomings. The loop performing
the reduction in shared memory starts from `i = blockDim.x / 2` and the first
predicate `if (tid < i)` immediately disables half of our block which only
helped load the data into shared. We change the kernel:

```diff
const uint32_t tid = threadIdx.x,
               bid = blockIdx.x,
-              gid = bid * blockDim.x + tid;
+              gid = bid * (blockDim.x * 2) + tid;

// Read input from front buffer to shared
-shared[tid] = read_global_safe(gid);
+shared[tid] = op(read_global_safe(gid), read_global_safe(gid + blockDim.x));
__syncthreads();
```

and the host code as such:

```diff
-	std::size_t factor = block_size;
+	std::size_t factor = block_size * 2;
```

By eliminating half of the threads and giving meaningful work to all the
threads by unconditionally performing a binary `op`, we don't waste half of our
threads.

While global memory is read in a coalesced fashion which the memory controller
prefers, we're still some ways from optimal performance, hinting at being
limited by instruction throughput.

### 5. Omit superfluous synchronization

Warps/Wavefronts are known to execute in a strictly* lockstep fashion,
therefore once shared reduction has reached a point when it's only a single
warp participating meaningfully, we can cut short the loop and let the rest of
the warps terminate, moreover without the need for syncing the entire block, we
can also unroll the loop.

```diff
-template<typename T, typename F>
+template<uint32_t WarpSize, typename T, typename F>
__global__ void kernel(
	...
)
{
	...
// Shared reduction
-for (uint32_t i = blockDim.x / 2; i != 0; i /= 2)
+for (uint32_t i = blockDim.x / 2; i > WarpSize; i /= 2)
{
	if (tid < i)
		shared[tid] = op(shared[tid], shared[tid + i]);
	__syncthreads();
}
+// Warp reduction
+	tmp::static_for<WarpSize, tmp::not_equal<0>, tmp::divide<2>>([&]<int I>()
+	{
+		if (tid < I)
+			shared[tid] = op(shared[tid], shared[tid + I]);
+	});
```

```{note}
Lanes of a warp on NVIDIA hardware may execute somewhat independently, so long
as the programmer assists the compiler using dedicated built-in functions. (A
feature called Independent Thread Scheduling.) The HIP headers do not expose
the matching warp primitive overloads. Portable applications can still tap into
this feature with carefully `#ifdef`ed code, but the benefit of doing so in
this particular case, when the active/inactive threads in a block inherently
manifest [partitioned](https://en.cppreference.com/w/cpp/algorithm/partition)
is next to zero. Benefits may be reaped if partitioning is costly, in which
case the thread scheduler will reorder the threads of a block and schedule the
active threads together into the same warp. It's a hardware feature assisting
highly and unpredictably divergent workflow, such as ray-tracing but is
unnecessary gymnastics in well defined algorithms.
```
