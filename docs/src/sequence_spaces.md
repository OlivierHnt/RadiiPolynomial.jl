# Spaces

## Parameter space

 Given a field ``\mathbb{K}``, a [`ParameterSpace`](@ref) corresponds to ``\mathbb{K}``. This is the main space for unfolding parameters.

## Sequence space

The spaces under considerations are the subspaces of ``\ell^1_{\mathbb{S}_1 \times \dots \times \mathbb{S}_d,\nu}`` such that their elements are compactly supported sequences. The abstract type [`SequenceSpace`](@ref) has the two subtypes [`UnivariateSpace`](@ref) and [`TensorSpace`](@ref) where the latter corresponds to the tensor products ``\otimes`` of some spaces in [`UnivariateSpace`](@ref).

Each [`UnivariateSpace`](@ref) comes with a field `order` beyond which the coefficients of all sequences in this space are zero.

### Taylor

A [`Taylor`](@ref) sequence space is a truncated Taylor space which essentially amounts to a univariate polynomial of a prescribed order ``n``. The ordered basis under consideration is ``\{\phi_0, \dots, \phi_n\}`` where ``\phi_k(t) \doteqdot t^k`` for ``k = 0, \dots, n``. This is the main sequence space when dealing with analytic functions.

### Fourier

A [`Fourier`](@ref) sequence space is a truncated Fourier space of a prescribed order ``n`` and frequency ``\omega``. The ordered basis under consideration is ``\{\phi_{-n}, \dots, \phi_n\}`` where ``\phi_k(t) \doteqdot e^{i \omega k t}`` for ``k = -n, \dots, n``. This is the main sequence space when dealing with periodic functions.

### Chebyshev

A [`Chebyshev`](@ref) sequence space is a truncated Chebyshev space of a prescribed order ``n``. The ordered basis under consideration is ``\{\phi_0, \dots, \phi_n\}`` where ``\phi_0 \doteqdot 1`` and ``\phi_k(\cos(\theta)) \doteqdot 2\cos(k\theta)`` for ``k = 1, \dots, n``.

### Tensor space

A [`TensorSpace`](@ref) is the tensor product of some [`UnivariateSpace`](@ref). The ordered basis under consideration is generated from the ordered basis of each [`UnivariateSpace`](@ref).






# Sequences

A [`Sequence`](@ref) is a structure representing an element of a [`SequenceSpace`](@ref), that is it corresponds to a compactly supported sequence. The coefficients of a [`Sequence`](@ref) are organized according to the space indexing.

## Arithmetic

The arithmetic operations `+,-,*,^` are implemented along with the convenient *bar operations* `+̄,-̄,*̄,^̄` (`+\bar<TAB>, -\bar<TAB>, *\bar<TAB>, ^\bar<TAB>`).

!!! note
    Divisions between sequences and other elementary functions (e.g. `exp`, `log`, `cos`, `sin`)
    are purposely not supported since, in general, they yield sequences which are not compactly supported.

```Julia
using RadiiPolynomial
a = Sequence(Fourier(1, 1.0) ⊗ Taylor(2), [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) # a(x,y) = cos(x) + y^2
a^2 # a(x,y)^2 = cos^2(x) + 2cos(x)y^2 + y^4
```

The internal routines of the multiplication use the results of [Computing Discrete Convolutions with Verified Accuracy Via Banach Algebras and the FFT](https://link.springer.com/article/10.21136/AM.2018.0082-18) where the author shows how to rigorously bound convolutions to temper machine precision limitations.

```Julia
using RadiiPolynomial, IntervalArithmetic
space = Taylor(2^8-1)
a = Sequence(space, [@interval(x) for x ∈ rand(dimension(space))])
maximum(radius.(a))
maximum(radius.(a^3))
maximum(radius.(a^5))
```

The above shows a considerable loss of accuracy. Although, in practice series exhibit a decay which prevents such a large error to occur. Indeed, if we add a small decay to our sequence:

```Julia
using RadiiPolynomial, IntervalArithmetic
space = Taylor(2^8-1)
a = Sequence(space, [@interval(rand()/4.0^i) for i ∈ allindices(space)])
maximum(radius.(a^5))
```

!!! note
    The intervals precision can be adjusted via `setprecision` (cf. [IntervalArithmetic.jl](https://github.com/JuliaIntervals/IntervalArithmetic.jl)).




# Operators

Given a linear operator `A` defined from a sequence space to an other, an [`Operator`](@ref) is a structure representing the restriction of `A` to an *effective domain* and *effective codomain* of `A`. The *effective domain* and *effective codomain* of an operator are defined as sequence spaces with the property that the operator and its restriction to the *effective domain* and *effective codomain* coincide.

Consequently, we can conveniently extend an [`Operator`](@ref) when needed.

## Action

The action of an [`Operator`](@ref) is performed by the left product of a [`Sequence`](@ref) with an [`Operator`](@ref).

```Julia
using RadiiPolynomial
A = Operator(Taylor(2), Taylor(3), [1.0 2.0 3.0 ; 4.0 5.0 6.0 ; 7.0 8.0 9.0 ; 10.0 11.0 12.0]);
A*Sequence(Taylor(1), [1.0, 2.0])
A*Sequence(Taylor(2), [1.0, 2.0, 3.0])
A*Sequence(Taylor(3), [1.0, 2.0, 3.0, 4.0])
```

!!! note
    [`Operator`](@ref) is callable: `(A::Operator)(b::Sequence) = A*b`.

## Arithmetic

The arithmetic operations `+,-,*` are implemented along with the convenient *bar operations* `+̄,-̄` (`+\bar<TAB>, -\bar<TAB>`).

```Julia
using RadiiPolynomial
A = Operator(Taylor(1) ⊗ Chebyshev(1), Taylor(2) ⊗ Chebyshev(1), rand(1:3, dimension(Taylor(2) ⊗ Chebyshev(1)), dimension(Taylor(1) ⊗ Chebyshev(1))));
A.coefficients
B = Operator(Taylor(2) ⊗ Chebyshev(0), Taylor(1) ⊗ Chebyshev(2), rand(1:3, dimension(Taylor(1) ⊗ Chebyshev(2)), dimension(Taylor(2) ⊗ Chebyshev(0))));
B.coefficients
C = A + B;
C.domain
C.codomain
C.coefficients
C̄ = A +̄ B;
C̄.domain
C̄.codomain
C̄.coefficients
```
