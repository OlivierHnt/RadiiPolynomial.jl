```@setup sequences
using RadiiPolynomial
```

# Sequences

A [`Sequence`](@ref) is a structure representing a sequence in a prescribed [`VectorSpace`](@ref). More precisely, a [`Sequence`](@ref) is comprised of the two fields `space::VectorSpace` and `coefficients::AbstractVector` with matching dimension and length.

```@repl sequences
a = Sequence(Taylor(1), [1, 2])
```

The two fields `space` and `coefficients` are accessible via the respective functions of the same name.

```@repl sequences
space(a)
coefficients(a)
```

For convenience, the methods `zeros`, `ones`, `fill` and `fill!` are available:

```@repl sequences
s = Taylor(1)
zeros(s)
ones(s)
fill(2, s)
fill!(zeros(s), 2)
```

The coefficients of a [`Sequence`](@ref) are indexed according to the indices of the space (as given by `indices`).

```@repl sequences
a[0:1] # indices(space(a))
```

When the space of a [`Sequence`](@ref) is a [`CartesianSpace`](@ref), its coefficients are given as the concatenation of the coefficients associated with each space. The function [`component`](@ref) extracts a [`Sequence`](@ref) composing the cartesian space; its coefficients are a view into the parent sequence, so the two stay glued. The function [`unpack`](@ref) unglues the sequence into the `Vector` of its components, which lets generic code written for vectors operate on the components seamlessly.

```@repl sequences
b = Sequence(ScalarSpace() × Taylor(1)^2, [1, 2, 3, 4, 5])
b[1:5] # indices(space(b))
component(b, 1) # extract the sequence associated with the space ScalarSpace()
component(b, 2) # extract the sequence associated with the space Taylor(1)^2
component(component(b, 2), 1)
component(component(b, 2), 2)
unpack(b)
```

Similarly, the function [`eachcomponent`](@ref) returns a `Generator` whose iterates yield each [`Sequence`](@ref) composing the cartesian space.

## Arithmetic

The addition and subtraction operations are implemented as the `+` and `-` functions respectively.

```@repl sequences
c = Sequence(Taylor(1), [0, 1])
d = Sequence(Taylor(2), [1, 2, 1])
c + d
c - d
```

The discrete convolution between sequences whose spaces are a [`SequenceSpace`](@ref) is implemented as the `*`, `mul!` and `^` functions. Their *bar* counterparts `mul_bar` (unicode alias `*\bar<tab>`) and `pow_bar` (unicode alias `^\bar<tab>`) give the result projected in the smallest compatible space between the operands; in general, `mul_bar` is not associative.

```@repl sequences
c * d
c ^ 3
mul_bar(c, d) # project(c * d, Taylor(1))
pow_bar(c, 3) # project(c ^ 3, Taylor(1))
```

To improve performance, the FFT algorithm may be used to compute discrete convolutions via the [Convolution Theorem](https://en.wikipedia.org/wiki/Convolution_theorem). However, the performance gain is tempered with the loss of accuracy which may stop the decay of the coefficients. To circumvent machine precision limitations, the `banach_rounding!` method enclose rigorously each term of the convolution beyond a prescribed order.[^1]

[^1]: J.-P. Lessard, [Computing discrete convolutions with verified accuracy via Banach algebras and the FFT](https://doi.org/10.21136/AM.2018.0082-18), *Applications of Mathematics*, **63** (2018), 219-235.

```@repl sequences
x = Sequence(Taylor(3), interval.([inv(10_000.0 ^ i) for i ∈ 0:3]))
RadiiPolynomial.set_conv_algorithm(:fft)
x³ = x ^ 3
RadiiPolynomial.set_conv_algorithm(:sum) # default algorithm
x³ = x ^ 3 # only rounding error
```

## Grids and interpolation

The functions `to_grid` and `to_seq` convert between a sequence and its values on a sampling grid, using rigorous FFTs when the coefficients are intervals. `grid_size(space)` gives the number of nodes per factor, as a tuple with one entry per factor: for `Taylor` these are the roots of unity, for `Fourier` the equispaced points of the period, and for `Chebyshev` the Chebyshev–Lobatto nodes ``x_k = \cos(\pi (k-1)/(m-1))``, ordered from ``x = 1`` down to ``x = -1``.

```@repl sequences
m = grid_size(Chebyshev(2)) # one entry per factor
g = [cospi((k-1)/2) for k ∈ 1:m[1]] # values of f(x) = x at the nodes
to_seq(g, Chebyshev(2)) # x = 2 (0.5 T₁(x)); interior modes carry an implicit factor 2
to_grid(ans, m) # back to the nodes
```

The second argument of `to_grid` is a tuple of grid sizes, one per discretized axis (an `Integer` is accepted as shorthand for a single axis, and `Chebyshev` axes take either a ``2^k+1`` Lobatto half grid or a power-of-two full grid). Giving fewer sizes than the space has factors discretizes only the leading factors, which is convenient for families of sequences depending on a parameter (e.g. in continuation): `to_grid(a, m)` then returns a grid of `Sequence`s on the remaining factors, and `to_seq(x_grid, s)` interpolates such a grid back into a `Sequence` on `s ⊗ inner_space`. The analogous methods exist for `LinearOperator`s, where `s` must match the leading factors of the codomain. Symmetric inner spaces are supported, and round-trip with their symmetry intact.

```@repl sequences
a = Sequence(Chebyshev(2) ⊗ Fourier(1, 1.0), collect(1.0:9)) # a family of Fourier sequences
x_grid = to_grid(a, grid_size(Chebyshev(2))) # one Fourier sequence per Chebyshev–Lobatto node
to_seq(x_grid, Chebyshev(2)) # interpolate back
```

## API

```@meta
CollapsedDocStrings = true
```

```@autodocs
Modules = [RadiiPolynomial]
Private = false
Pages   = ["sequence_spaces/sequences/sequence.jl",
    "sequence_spaces/sequences/convolution.jl",
    "sequence_spaces/sequences/fft.jl"]
```
