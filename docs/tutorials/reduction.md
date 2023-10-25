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
a dataset and looking for the maximum value of a dataset.

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
implementations differ ever so slightly and the article will follow when the
input may be destroyed.
