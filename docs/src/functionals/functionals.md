# Functionals

Given a linear functional `A` defined on a sequence space, a [`Functional`](@ref) is a structure representing the restriction of `A` to an *effective domain* of `A`. The *effective domain* of a functional is defined as a sequence space with the property that the functional and its restriction to the *effective domain* coincide.

Consequently, we can conveniently extend a [`Functional`](@ref) when needed.

```@docs
Functional
```

## Action

The action of a [`Functional`](@ref) is performed by the left product of a [`Sequence`](@ref) with a [`Functional`](@ref).

```@repl
using RadiiPolynomial
A = Functional(Taylor(2), [1.0, 2.0, 3.0]);
A*Sequence(Taylor(1), [1.0, 2.0])
A*Sequence(Taylor(2), [1.0, 2.0, 3.0])
A*Sequence(Taylor(3), [1.0, 2.0, 3.0, 4.0])
```

!!! note
    [`Functional`](@ref) is callable: `(A::Functional)(b::Sequence) = A*b`.

## Arithmetic

The arithmetic operations `+,-` are implemented along with the convenient *bar operations* `+̄,-̄` (`+\bar<TAB>, -\bar<TAB>`).

```@repl
using RadiiPolynomial
A = Functional(Taylor(1) ⊗ Chebyshev(1), rand(1:3, length(Taylor(1) ⊗ Chebyshev(1))));
A.coefficients
B = Functional(Taylor(0) ⊗ Chebyshev(2), rand(1:3, length(Taylor(0) ⊗ Chebyshev(2))));
B.coefficients
C = A + B;
C.domain
C.coefficients
C̄ = A +̄ B;
C̄.domain
C̄.coefficients
```
