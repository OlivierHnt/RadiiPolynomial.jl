# Spaces

The abstract type `SequenceSpace` has the two subtypes `UnivariateSpace` and `TensorSpace` where the latter corresponds to the tensor products of some spaces in `UnivariateSpace`. Every `UnivariateSpace` spaces have a field `order :: Int` characterizing the largest order such that `a[i] = 0` for all `|i| > a.order` and sequences `a :: Sequence` in the given space.

## Taylor

The `Taylor` sequence space should be used when the functions under consideration are known to be analytic.

## Fourier

The `Fourier` sequence space should be used when the functions under consideration are known to be periodic functions.

## Chebyshev

The `Chebyshev`sequence space represents a Chebyshev series of the first kind.

## Tensor space
