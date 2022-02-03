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

```@docs
Sequence
```

## Arithmetic

The addition and subtraction operations are implemented as the `+` and `-` functions respectively. Their *bar* counterparts `+̄` (`+\bar<TAB>`) and `-̄` (`-\bar<TAB>`) give the result projected in the smallest compatible space between the operands.

```@repl sequences
c = Sequence(Taylor(1), [0, 1])
d = Sequence(Taylor(2), [1, 2, 1])
c + d
c - d
c +̄ d # project(c + d, Taylor(1))
c -̄ d # project(c - d, Taylor(1))
```

The discrete convolution between sequences whose spaces are a [`SequenceSpace`](@ref) is implemented as the `*` and `^` functions. Their *bar* counterparts `*̄` (`*\bar<TAB>`) and `^̄` (`^\bar<TAB>`) give the result projected in the smallest compatible space between the operands; in general, `*̄` is not associative.

```@repl sequences
c * d
c ^ 3
c *̄ d # project(c * d, Taylor(1))
c ^̄ 3 # project(c ^ 3, Taylor(1))
```

To improve performance, the FFT algorithm may be used to compute discrete convolutions via the [Convolution Theorem](https://en.wikipedia.org/wiki/Convolution_theorem). However, the performance gain is tempered with the loss of accuracy which may stop the decay of the coefficients.

```@repl sequences
x = Sequence(Taylor(3), Interval.([inv(10_000.0 ^ i) for i ∈ 0:3]))
x³ = x ^ 3
x³_fft = rifft!(similar(x³), fft(x, fft_size(space(x), 3)) .^ 3)
```

To circumvent machine precision limitations, the `banach_rounding!` method enclose rigorously each term of the convolution beyond a prescribed order.[^1]

[^1]: J.-P. Lessard, [Computing discrete convolutions with verified accuracy via Banach algebras and the FFT](https://doi.org/10.21136/AM.2018.0082-18), *Applications of Mathematics*, **63** (2018), 219-235.

The rounding strategy for `*`, `^`, `*̄` and `^̄` is integrated in the functions `banach_rounding_mul`, `banach_rounding_pow`, `banach_rounding_mul_bar` and `banach_rounding_pow_bar` respectively.

```@repl sequences
X = ℓ¹(GeometricWeight(Interval(10_000.0)))
banach_rounding!(x³_fft, norm(x, X) ^ 3, X, 5)
```
