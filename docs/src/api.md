# API

## Radii polynomial

```@docs
rpa_finite_dimension
rpa_finite_dimension_newton
roots_radii_polynomial
```

## Spaces

```@docs
SequenceSpace
UnivariateSpace
Taylor
Fourier
Chebyshev
TensorSpace
```

## Sequences

```@docs
Sequence
project(::Sequence, ::SequenceSpace)
selectdim(::Sequence{TensorSpace{T}}, ::Int, ::Int) where {N,T<:NTuple{N,UnivariateSpace}}
shift
banach_algebra_rounding!
```

## Functionals

```@docs
Functional
project(::Functional, ::SequenceSpace)
selectdim(::Functional{TensorSpace{T}}, ::Int, ::Int) where {N,T<:NTuple{N,UnivariateSpace}}
```

## Operators

```@docs
Operator
project(::Operator, ::SequenceSpace, ::SequenceSpace)
```
