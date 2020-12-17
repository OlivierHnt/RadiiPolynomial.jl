# Spaces

The spaces under considerations are the subspaces of ``\ell^1_{\mathbb{S}_1 \times \dots \times \mathbb{S}_d,\nu}`` such that their elements are compactly supported sequences. The abstract type [`SequenceSpace`](@ref) has the two subtypes [`UnivariateSpace`](@ref) and [`TensorSpace`](@ref) where the latter corresponds to the tensor products `⊗` of some spaces in [`UnivariateSpace`](@ref).

Each [`UnivariateSpace`](@ref) comes with a field `order` beyond which the coefficients of all sequences in this space are zero. Naturally, one can take the union `∪` and the intersection `∩` of some [`UnivariateSpace`](@ref).

## Taylor

A [`Taylor`](@ref) sequence space is a truncated Taylor space which essentially amounts to a univariate polynomial of a prescribed order ``n``. The ordered basis under consideration is ``\{\phi_0, \dots, \phi_n\}`` where ``\phi_k(t) \doteqdot t^k`` for ``k = 0, \dots, n``. This is the main sequence space when dealing with analytic functions.

## Fourier

A [`Fourier`](@ref) sequence space is a truncated Fourier space of a prescribed order ``n`` and frequency ``\omega``. The ordered basis under consideration is ``\{\phi_{-n}, \dots, \phi_n\}`` where ``\phi_k(t) \doteqdot e^{i \omega k t}`` for ``k = -n, \dots, n``. This is the main sequence space when dealing with periodic functions.

## Chebyshev

A [`Chebyshev`](@ref) sequence space is a truncated Chebyshev space of a prescribed order ``n``. The ordered basis under consideration is ``\{\phi_0, \dots, \phi_n\}`` where ``\phi_0 \doteqdot 1`` and ``\phi_k(\cos(\theta)) \doteqdot 2\cos(k\theta)`` for ``k = 1, \dots, n``.

## Tensor space

A [`TensorSpace`](@ref) is the tensor product of some [`UnivariateSpace`](@ref). The ordered basis under consideration is generated from the ordered basis of each [`UnivariateSpace`](@ref).

```@repl
using RadiiPolynomial
taylor = Taylor(2)
fourier = Fourier(1, 1.)
taylor ⊗ fourier
eachindex(taylor ⊗ fourier)
```

## Union, intersection

!!! note
    In the future, we might want to add symmetries to the sequence spaces; to anticipate this feature, there is a *bar union* `∪̄` (`∪\bar<TAB>`) operation which currently coincides with `∩`.
