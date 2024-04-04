```@setup special_operators
using RadiiPolynomial
```

# Special operators

In this section, we present several operations common to dynamical systems.

## Projection

When working with [`SequenceSpace`](@ref), one frequently needs to adjust the order of truncation of the chosen basis. This operation is implemented as the [`project`](@ref) and [`project!`](@ref) functions. In fact, these functions provide a general mechanism to retrieve a finite part of the infinite dimensional operators introduced later in this section.

Each [`project`](@ref) or [`project!`](@ref) call verifies a compatibility criterion between spaces. For [`Sequence`](@ref) and [`LinearOperator`](@ref), two [`VectorSpace`](@ref) are *compatible* if:
- all comprised [`SequenceSpace`](@ref) only differ from their order. For instance, `Taylor(n)` and `Taylor(m)` are compatible for any positive `n::Int` and `m::Int`. However, `Taylor(n)` and `TensorSpace(Taylor(m), Fourier(k, 1.0))` are not compatible for any positive `n::Int`, `m::Int` and `k::Int`.
- all comprised [`CartesianSpace`](@ref) have the same number of cartesian products. For instance, `CartesianPower(a, 2)` and `CartesianProduct(a, a)` are compatible for any `a::VectorSpace`. However, `CartesianProduct(a, b)` and `CartesianProduct(CartesianPower(a, 1), b)` are not compatible for any `a::VectorSpace` and `b::VectorSpace`.

```@repl special_operators
A = LinearOperator(Taylor(1) ⊗ Chebyshev(1), Taylor(1) ⊗ Chebyshev(1), [1 0 0 0 ; 0 1 0 0 ; 0 0 1 0 ; 0 0 0 1]) # project(I, Taylor(1) ⊗ Chebyshev(1), Taylor(1) ⊗ Chebyshev(1))
project(A, Taylor(1) ⊗ Chebyshev(2), Taylor(2) ⊗ Chebyshev(1))
```

Moreover, the following identifications are permitted:

```@repl special_operators
a = Sequence(Taylor(1), [1, 1]) # 1 + x
A = project(a, ParameterSpace(), Taylor(2))
project(A, space(a))
```

## Multiplication

Let ``V`` be a [`SequenceSpace`](@ref) with discrete convolution ``*`` and ``a \in V``. The multiplication operator [`Multiplication`](@ref) represents the mapping ``\mathcal{M}_a : V \to V`` defined by

```math
\mathcal{M}_a (b) \bydef a * b, \qquad \text{for all } b \in V.
```

The action of [`Multiplication`](@ref) is performed by the right product `*` of a [`Multiplication`](@ref) with a `Sequence{<:SequenceSpace}`; alternatively, [`Multiplication`](@ref) defines a method on a `Sequence{<:SequenceSpace}` representing `*`.

```@repl special_operators
a = Sequence(Taylor(1), [1, 1]); # 1 + x
b = Sequence(Taylor(2), [0, 0, 1]); # x^2
a * b
ℳ = Multiplication(a)
ℳ * b # ℳ(b)
```

A finite dimensional truncation of [`Multiplication`](@ref) may be obtained via [`project`](@ref project(::Multiplication, ::SequenceSpace, ::SequenceSpace)) or [`project!`](@ref project!(::LinearOperator{<:SequenceSpace,<:SequenceSpace}, ::Multiplication)).

```@repl special_operators
project(ℳ, Taylor(2), image(*, Taylor(1), Taylor(2)))
```

## Derivation and integration

Both [`Derivative`](@ref) and [`Integral`](@ref) have a field `order::Union{Int,Tuple{Vararg{Int}}}` to specify how many times the operator is composed with itself. No derivation or integration is performed whenever a value of `0` is given.

```@repl special_operators
a = Sequence(Taylor(2), [1, 1, 1]); # 1 + x + x^2
differentiate(a)
𝒟 = Derivative(1)
𝒟 * a # 𝒟(a)
```

A finite dimensional truncation of [`Derivative`](@ref) and [`Integral`](@ref) may be obtained via [`project`](@ref) or [`project!`](@ref):

```@repl special_operators
project(Derivative(1), Taylor(2), image(Derivative(1), Taylor(2)), Float64)
project(Integral(1), Taylor(2), image(Integral(1), Taylor(2)), Float64)
```

## Evaluation

The evaluation operator [`Evaluation`](@ref) has a field `value::Union{Number,Nothing,Tuple{Vararg{Union{Number,Nothing}}}}` representing the evaluation point. No scaling is performed whenever a value of `nothing` is given.

```@repl special_operators
a = Sequence(Taylor(2), [1, 1, 1]); # 1 + x + x^2
evaluate(a, 0.1)
ℰ = Evaluation(0.1)
ℰ * a # ℰ(a)
b = Sequence(Taylor(1) ⊗ Fourier(1, 1.0), [0.5, 0.5, 0.0, 0.0, 0.5, 0.5]); # (1 + x) cos(y)
evaluate(b, (0.1, nothing)) # Evaluation(0.1, nothing) * b
```

Moreover, [`Evaluation`](@ref) is defined on [`CartesianSpace`](@ref) by acting component-wise.

```@repl special_operators
c = Sequence(Taylor(1)^2, [1, 1, 2, 2]); # 1 + x, 2 + 2x
evaluate(c, 0.1) # Evaluation(0.1) * c
```

A finite dimensional truncation of [`Evaluation`](@ref) may be obtained via [`project`](@ref project(::Evaluation, ::VectorSpace, ::VectorSpace)) or [`project!`](@ref project!(::LinearOperator, ::Evaluation)):

```@repl special_operators
project(Evaluation(0.1), Taylor(2), image(Evaluation(0.1), Taylor(2)), Float64)
```

Furthermore, in the context of [`Evaluation`](@ref), the concept of compatibility between two [`VectorSpace`](@ref) is more permissive to allow manipulating [`Evaluation`](@ref) more like a functional:

```@repl special_operators
project(Evaluation(0.1), Taylor(2), ParameterSpace(), Float64)
```

## Scale

The scale operator [`Scale`](@ref) has a field `value::Union{Number,Tuple{Vararg{Number}}}` representing the scaling factor. No scaling is performed whenever a value of `1` is given.

!!! note
    Currently, only [`Taylor`](@ref) and [`Fourier`](@ref) spaces allow values different than `1`.

```@repl special_operators
a = Sequence(Taylor(2), [1, 1, 1]) # 1 + x + x^2
scale(a, 2)
𝒮 = Scale(2)
𝒮 * a # 𝒮(a)
```

A finite dimensional truncation of [`Scale`](@ref) may be obtained via [`project`](@ref project(::Scale, ::VectorSpace, ::VectorSpace)) or [`project!`](@ref project!(::LinearOperator, ::Scale)):

```@repl special_operators
project(Scale(2), Taylor(2), image(Scale(2), Taylor(2)), Float64)
```

## Shift

The shift operator [`Shift`](@ref) has a field `value::Union{Number,Tuple{Vararg{Number}}}` representing the shift. No shift is performed whenever a value of `0` is given.

!!! note
    Currently, only [`Fourier`](@ref) space allows values different than `0`.

```@repl special_operators
a = Sequence(Fourier(1, 1.0), [0.5, 0.0, 0.5]) # cos(x)
shift(a, π)
𝒮 = Shift(π)
𝒮 * a # 𝒮(a)
```

A finite dimensional truncation of [`Shift`](@ref) may be obtained via [`project`](@ref project(::Shift, ::VectorSpace, ::VectorSpace)) or [`project!`](@ref project!(::LinearOperator, ::Shift)):

```@repl special_operators
project(Shift(π), Fourier(1, 1.0), image(Shift(π), Fourier(1, 1.0)), Complex{Float64})
```

## API

```@meta
CollapsedDocStrings = true
```

```@autodocs
Modules = [RadiiPolynomial]
Private = false
Pages   = ["sequence_spaces/special_operators/special_operator.jl",
    "sequence_spaces/special_operators/projection.jl",
    "sequence_spaces/special_operators/multiplication.jl",
    "sequence_spaces/special_operators/calculus.jl",
    "sequence_spaces/special_operators/evaluation.jl",
    "sequence_spaces/special_operators/scale.jl",
    "sequence_spaces/special_operators/shift.jl"]
```
