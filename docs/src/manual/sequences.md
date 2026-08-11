```@setup sequences
using RadiiPolynomial
```

# Sequences

A [`Sequence`](@ref) is a structure representing a sequence in a prescribed [`VectorSpace`](@ref).
More precisely, a [`Sequence`](@ref) is comprised of the two fields `space::VectorSpace` and `coefficients::AbstractVector` where the `dimension` of the former matches the `length` of the latter.

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

When the space of a [`Sequence`](@ref) is a [`CartesianSpace`](@ref), its coefficients are given as the concatenation of the coefficients associated with each space.
The function [`component`](@ref) extracts a [`Sequence`](@ref) composing the cartesian space; its coefficients are a view into the parent sequence, so the two stay glued.
The function [`unpack`](@ref) unglues the sequence into the `Vector` of its components, which lets generic code written for vectors operate on the components seamlessly.

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

The discrete convolution between sequences whose spaces are a [`SequenceSpace`](@ref) is implemented as the `*`, `mul!` and `^` functions.
Their *bar* counterparts `mul_bar` (unicode alias `*\bar<tab>`) and `pow_bar` (unicode alias `^\bar<tab>`) give the result projected in the smallest compatible space between the operands; in general, `mul_bar` is not associative.

```@repl sequences
c * d
c ^ 3
mul_bar(c, d) # project(c * d, Taylor(1))
pow_bar(c, 3) # project(c ^ 3, Taylor(1))
```

To improve performance, the FFT algorithm may be used to compute discrete convolutions via the [Convolution Theorem](https://en.wikipedia.org/wiki/Convolution_theorem).
However, the performance gain is tempered with the loss of accuracy which may stop the decay of the coefficients.
To circumvent machine precision limitations, the coefficients beyond a prescribed order are enclosed rigorously by a Banach-algebra estimate.[^1]

[^1]: J.-P. Lessard, [Computing discrete convolutions with verified accuracy via Banach algebras and the FFT](https://doi.org/10.21136/AM.2018.0082-18), *Applications of Mathematics*, **63** (2018), 219-235.

```@repl sequences
x = Sequence(Taylor(3), interval.([inv(10_000.0 ^ i) for i ∈ 0:3]))
set_conv_algorithm(:fft)
x³ = x ^ 3
set_conv_algorithm(:loop) # default algorithm
x³ = x ^ 3 # only rounding error
```

## Grids and interpolation

The functions [`to_grid`](@ref) and [`to_coef`](@ref) convert between coefficient and grid space, using rigorous FFTs when the coefficients are intervals.
`grid_size(space)` gives the number of nodes, as a tuple with one entry per space.
The nodes are:
- for [`Taylor`](@ref): the roots of unity ``e^{\mathrm{i} 2\pi j / (m-1)}`` for ``j = 0, ..., m-1``.
- for [`Fourier`](@ref) (with frequency ``\omega``): the equispaced points of the period ``2\pi/\omega j / (m-1)`` for ``j = 0, ..., m-1``.
- for [`Chebyshev`](@ref): the Chebyshev--Lobatto nodes ``\cos(\pi j /(m-1))`` for ``j = 0, ..., m-1`` (ordered from ``1`` down to ``-1``).
It is the smallest grid that determines the space, so `to_coef(f, space)` returns the interpolant of `f` at those nodes, and `to_coef(to_grid(a), space(a))` recovers `a`.

```@repl sequences
m = grid_size(Chebyshev(2)) # one entry per factor
g = [cospi((k-1)/(m[1]-1)) for k ∈ 1:m[1]] # values of f(x) = x at the nodes
to_coef(g, Chebyshev(2)) # x = 2 (0.5 T₁(x)) due to the normalization
to_grid(ans, m) # back to the nodes
```

Oversampling a grid keeps the interpolant unchanged.
Giving fewer sizes than a [`TensorSpace`](@ref) has factors discretizes only the leading factors `lead_space`, so that `to_grid(a, m)` then returns a grid of [`Sequence`](@ref)s on the remaining factors, and `to_coef(x_grid, s)` interpolates such a grid back into a `Sequence` on `lead_space ⊗ inner_space`.

```@repl sequences
a = Sequence(Chebyshev(2) ⊗ Fourier(1, 1.0), collect(1:9)) # a family of Fourier sequences
x_grid = to_grid(a, grid_size(Chebyshev(2))) # one Fourier sequence per Chebyshev–Lobatto node
to_coef(x_grid, Chebyshev(2)) # interpolate back
```

## API

```@meta
CollapsedDocStrings = true
```

```@autodocs
Modules = [RadiiPolynomial]
Private = false
Pages   = ["sequence_spaces/sequences/sequence.jl",
    "sequence_spaces/sequences/infinite_sequence.jl",
    "sequence_spaces/sequences/convolution.jl",
    "sequence_spaces/sequences/fft.jl"]
```

```@docs
*(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace})
^(::Sequence{<:SequenceSpace}, ::Integer)
```
