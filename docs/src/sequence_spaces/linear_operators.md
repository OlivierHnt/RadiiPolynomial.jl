```@setup linear_operators
using RadiiPolynomial
```

# Linear operators

A [`LinearOperator`](@ref) is a structure representing a linear operator from a [`VectorSpace`](@ref) to an other. More precisely, a [`LinearOperator`](@ref) is comprised of the three fields `domain::VectorSpace`, `codomain::VectorSpace` and `coefficients::AbsractMatrix` with matching dimensions and size.

```@repl linear_operators
A = LinearOperator(Taylor(1), Taylor(1), [1 2 ; 3 4])
```

The three fields `domain`, `codomain` and `coefficients` are accessible via the respective functions of the same name.

```@repl linear_operators
domain(A)
codomain(A)
coefficients(A)
```

The coefficients of a [`LinearOperator`](@ref) are indexed according to the indices of the domain and codomain (as given by `indices`).

```@repl linear_operators
A[0:1,0:1] # indices(domain(A)), indices(codomain(A))
```

When the domain and/or the codomain of a [`LinearOperator`](@ref) is a [`CartesianSpace`](@ref), its coefficients can be thought of as a block matrix . The function `component` extracts a [`LinearOperator`](@ref) composing the cartesian space.

```@repl linear_operators
B = LinearOperator(ParameterSpace() × Taylor(1)^2, ParameterSpace() × Taylor(1)^2, reshape(1:25, 5, 5))
B[1:5,1:5] # indices(domain(B)), indices(codomain(B))
component(B, 1, 1) # extract the linear operator associated with the domain ParameterSpace() and codomain ParameterSpace()
component(B, 2, 2) # extract the linear operator associated with the domain Taylor(1)^2 and codomain Taylor(1)^2
component(component(B, 2, 2), 1, 1)
component(component(B, 2, 2), 2, 2)
```

Similarly, the function `eachcomponent` returns a `Generator` whose iterates yield each [`LinearOperator`](@ref) composing the cartesian space.

```@docs
LinearOperator
```

## Arithmetic

The addition and subtraction operations are implemented as the `+` and `-` functions respectively. Their *bar* counterparts `+̄` (`+\bar<TAB>`) and `-̄` (`-\bar<TAB>`) give the result projected in the smallest compatible domain and codomain between the operands.

```@repl linear_operators
C = LinearOperator(Taylor(1), Taylor(1), [1 2 ; 3 4])
D = LinearOperator(Taylor(1), Taylor(2), [1 2 ; 3 4 ; 5 6])
C + D
C - D
C +̄ D # project(C + D, Taylor(1), Taylor(1))
C -̄ D # project(C - D, Taylor(1), Taylor(1))
C + I
C - I
```

The product between [`LinearOperator`](@ref) is implemented as the `*` and `^` functions. The division between [`LinearOperator`](@ref) is implemented as the `\` method.

```@repl linear_operators
C * D
C ^ 3
C \ C
```

The action of a [`LinearOperator`](@ref) is performed by the right product `*` of a [`LinearOperator`](@ref) with a [`Sequence`](@ref); alternatively, [`LinearOperator`](@ref) defines a method on a [`Sequence`](@ref) representing `*`. Naturally, the resulting sequence is an element of the codomain of the [`LinearOperator`](@ref).

Conversely, the operator `\` between a [`LinearOperator`](@ref) and a [`Sequence`](@ref) corresponds to the action of the inverse of the [`LinearOperator`](@ref); the output sequence is an element of the domain of the [`LinearOperator`](@ref).

```@repl linear_operators
x = Sequence(Taylor(2), [1, 1, 1])
C * x # C(x)
D \ x
```
