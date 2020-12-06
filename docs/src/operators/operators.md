# Operators

Given a linear operator `A` defined from a sequence space to an other, an [`Operator`](@ref) is a structure representing the restriction of `A` to an *effective domain* and *effective range* of `A`. The *effective domain* and *effective range* of an operator are defined as sequence spaces with the property that the operator and its restriction to the *effective domain* and *effective range* coincide.

Consequently, we can conveniently extend an [`Operator`](@ref) when needed.

```@docs
Operator
```

## Action

The action of an [`Operator`](@ref) is performed by the left product of a [`Sequence`](@ref) with an [`Operator`](@ref).

```@repl
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

```@repl
using RadiiPolynomial
A = Operator(Taylor(1) ⊗ Chebyshev(1), Taylor(2) ⊗ Chebyshev(1), rand(1:3, length(Taylor(2) ⊗ Chebyshev(1)), length(Taylor(1) ⊗ Chebyshev(1))));
A.coefficients
B = Operator(Taylor(2) ⊗ Chebyshev(0), Taylor(1) ⊗ Chebyshev(2), rand(1:3, length(Taylor(1) ⊗ Chebyshev(2)), length(Taylor(2) ⊗ Chebyshev(0))));
B.coefficients
C = A + B;
C.domain
C.range
C.coefficients
C̄ = A +̄ B;
C̄.domain
C̄.range
C̄.coefficients
```
