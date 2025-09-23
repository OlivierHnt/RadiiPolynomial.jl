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

When the space of a [`Sequence`](@ref) is a [`CartesianSpace`](@ref), its coefficients are given as the concatenation of the coefficients associated with each space. The function `component` extracts a [`Sequence`](@ref) composing the cartesian space.

```@repl sequences
b = Sequence(ParameterSpace() × Taylor(1)^2, [1, 2, 3, 4, 5])
b[1:5] # indices(space(b))
component(b, 1) # extract the sequence associated with the space ParameterSpace()
component(b, 2) # extract the sequence associated with the space Taylor(1)^2
component(component(b, 2), 1)
component(component(b, 2), 2)
```

Similarly, the function `eachcomponent` returns a `Generator` whose iterates yield each [`Sequence`](@ref) composing the cartesian space.

## Arithmetic

The addition and subtraction operations are implemented as the `+` and `-` functions respectively.

```@repl sequences
c = Sequence(Taylor(1), [0, 1])
d = Sequence(Taylor(2), [1, 2, 1])
c + d
c - d
```

The discrete convolution between sequences whose spaces are a [`SequenceSpace`](@ref) is implemented as the [`*(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace})`](@ref), [`mul!(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Number, ::Number)`](@ref) and [`^(::Sequence{<:SequenceSpace}, ::Int)`](@ref) functions. Their *bar* counterparts `mul_bar` (unicode alias `*\bar<tab>`) and `pow_bar` (unicode alias `^\bar<tab>`) give the result projected in the smallest compatible space between the operands; in general, `mul_bar` is not associative.

```@repl sequences
c * d
c ^ 3
mul_bar(c, d) # project(c * d, Taylor(1))
pow_bar(c, 3) # project(c ^ 3, Taylor(1))
```

To improve performance, the FFT algorithm may be used to compute discrete convolutions via the [Convolution Theorem](https://en.wikipedia.org/wiki/Convolution_theorem). However, the performance gain is tempered with the loss of accuracy which may stop the decay of the coefficients.

```@repl sequences
x = Sequence(Taylor(3), interval.([inv(10_000.0 ^ i) for i ∈ 0:3]))
x³ = x ^ 3
x³_fft = rifft!(zero(x³), fft(x, fft_size(space(x³))) .^ 3)
```

To circumvent machine precision limitations, the `banach_rounding!` method enclose rigorously each term of the convolution beyond a prescribed order.[^1]

[^1]: J.-P. Lessard, [Computing discrete convolutions with verified accuracy via Banach algebras and the FFT](https://doi.org/10.21136/AM.2018.0082-18), *Applications of Mathematics*, **63** (2018), 219-235.

The rounding strategy for [`*(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace})`](@ref), [`mul!(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Number, ::Number)`](@ref), [`^(::Sequence{<:SequenceSpace}, ::Int)`](@ref), `mul_bar` and `pow_bar` is integrated in the functions [`banach_rounding_mul`](@ref), [`banach_rounding_mul!`](@ref), [`banach_rounding_pow`](@ref), `banach_rounding_mul_bar` and `banach_rounding_pow_bar` respectively.

```@repl sequences
X = ℓ¹(GeometricWeight(interval(10_000.0)))
banach_rounding!(x³_fft, norm(x, X) ^ 3, X, 5)
```

## API

```@meta
CollapsedDocStrings = true
```

```@autodocs
Modules = [RadiiPolynomial]
Private = false
Pages   = ["sequence_spaces/sequence.jl",
    "sequence_spaces/arithmetic/convolution.jl",
    "sequence_spaces/arithmetic/fft.jl"]
```
