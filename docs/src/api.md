# API

## Radii polynomial

```@docs
Y
Z₁
Z₂
roots_radii_polynomial
FixedPointProblemFiniteDimension
ZeroFindingProblemFiniteDimension
TailProblem
ZeroFindingProblemCategory1
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
