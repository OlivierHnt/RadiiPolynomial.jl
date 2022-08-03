```@setup vector_spaces
using RadiiPolynomial
```

# [Vector spaces](@id vector_spaces)

RadiiPolynomial defines a variety of vector spaces to represent the Banach space on which one applies the Radii Polynomial Theorem.

All spaces mentioned below are a subtype of the abstract type [`VectorSpace`](@ref).

```
VectorSpace
├─ CartesianSpace
│  ├─ CartesianPower
│  └─ CartesianProduct
├─ ParameterSpace
└─ SequenceSpace
   ├─ BaseSpace
   │  ├─ Chebyshev
   │  ├─ Fourier
   │  └─ Taylor
   └─ TensorSpace
```

```@docs
VectorSpace
```

## Parameter space

A [`ParameterSpace`](@ref) represents the commutative field of a parameter. This is the standard space to use for an unfolding parameter.

```@repl vector_spaces
𝒫 = ParameterSpace()
dimension(𝒫)
indices(𝒫)
```

```@docs
ParameterSpace
```

## Sequence space

[`SequenceSpace`](@ref) is the abstract type for all sequence spaces.

```
SequenceSpace
├─ BaseSpace
│  ├─ Chebyshev
│  ├─ Fourier
│  └─ Taylor
└─ TensorSpace
```

```@docs
SequenceSpace
```

### BaseSpace

[`BaseSpace`](@ref) is the abstract type for all sequence spaces that are not a [`TensorSpace`](@ref) but can be interlaced to form one.

```
BaseSpace
├─ Chebyshev
├─ Fourier
└─ Taylor
```

```@docs
BaseSpace
```

#### Taylor

For a given order ``n``, a [`Taylor`](@ref) sequence space is the span of ``\{\phi_0, \dots, \phi_n\}`` where ``\phi_k(t) := t^k`` for ``k = 0, \dots, n`` and ``t \in [-\nu, \nu]`` for some appropriate ``\nu > 0``.

```@repl vector_spaces
𝒯 = Taylor(1)
order(𝒯)
dimension(𝒯)
indices(𝒯)
```

```@docs
Taylor
```

#### Fourier

For a given order ``n`` and frequency ``\omega``, a [`Fourier`](@ref) sequence space is the span of ``\{\phi_{-n}, \dots, \phi_n\}`` where ``\phi_k(t) := e^{i \omega k t}`` for ``k = -n, \dots, n`` and ``t \in \mathbb{R}/2\pi\omega^{-1}\mathbb{Z}``.

```@repl vector_spaces
ℱ = Fourier(1, 1.0)
order(ℱ)
frequency(ℱ)
dimension(ℱ)
indices(ℱ)
```

```@docs
Fourier
```

#### Chebyshev

For a given order ``n``, a [`Chebyshev`](@ref) sequence space is the span of ``\{\phi_0, \phi_1, \dots, \phi_n\}`` where ``\phi_0(t) := 1``, ``\phi_1(t) := t`` and ``\phi_k(t) := 2 t \phi_{k-1}(t) - \phi_{k-2}(t)`` for ``k = 2, \dots, n`` and ``t \in [-1, 1]``.

It is important to note that the coefficients ``\{a_0, a_1, \dots, a_n\}`` associated to a [`Chebyshev`](@ref) space are normalized such that ``\{a_0, 2a_1, \dots, 2a_n\}`` are the actual Chebyshev coefficients.

```@repl vector_spaces
𝒞 = Chebyshev(1)
order(𝒞)
dimension(𝒞)
indices(𝒞)
```

```@docs
Chebyshev
```

### Tensor space

A [`TensorSpace`](@ref) is the tensor product of some [`BaseSpace`](@ref). The standard constructor for [`TensorSpace`](@ref) is the `⊗` (`\otimes<tab>`) operator.

```@repl vector_spaces
𝒯_otimes_ℱ_otimes_𝒞 = Taylor(1) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(1) # TensorSpace((Taylor(1), Fourier(1, 1.0), Chebyshev(1)))
nspaces(𝒯_otimes_ℱ_otimes_𝒞)
order(𝒯_otimes_ℱ_otimes_𝒞)
frequency(𝒯_otimes_ℱ_otimes_𝒞, 2)
dimension(𝒯_otimes_ℱ_otimes_𝒞)
dimensions(𝒯_otimes_ℱ_otimes_𝒞)
indices(𝒯_otimes_ℱ_otimes_𝒞)
```

```@docs
TensorSpace
TensorIndices
⊗
```

## Cartesian space

[`CartesianSpace`](@ref) is the abstract type for all cartesian spaces.

```
CartesianSpace
├─ CartesianPower
└─ CartesianProduct
```

```@docs
CartesianSpace
```

### Cartesian power

A [`CartesianPower`](@ref) is the cartesian product of an identical [`VectorSpace`](@ref). The standard constructor for [`CartesianPower`](@ref) is the `^` operator.

```@repl vector_spaces
𝒯² = Taylor(1) ^ 2 # CartesianPower(Taylor(1), 2)
nspaces(𝒯²)
dimension(𝒯²)
indices(𝒯²)
```

```@docs
CartesianPower
^(::VectorSpace, ::Int)
```

### Cartesian product

A [`CartesianProduct`](@ref) is the cartesian product of some [`VectorSpace`](@ref). The standard constructor for [`CartesianProduct`](@ref) is the `×` (`\times<tab>`) operator.

```@repl vector_spaces
𝒫_times_𝒯 = ParameterSpace() × Taylor(1) # CartesianProduct((ParameterSpace(), Taylor(1)))
nspaces(𝒫_times_𝒯)
dimension(𝒫_times_𝒯)
indices(𝒫_times_𝒯)
```

```@docs
CartesianProduct
×
```
