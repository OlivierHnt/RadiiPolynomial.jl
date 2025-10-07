# safe equality

_safe_iszero(x) = iszero(x)
_safe_iszero(x::Union{Interval,Complex{<:Interval}}) = isthinzero(x)

_safe_isone(x) = isone(x)
_safe_isone(x::Union{Interval,Complex{<:Interval}}) = isthinone(x)

_safe_isequal(x, y) = x == y
_safe_isequal(x::Union{Interval,Complex{<:Interval}}, y::Union{Interval,Complex{<:Interval}}) =
    isequal_interval(x, y)

#

_setguarantee(a::Interval, t::Bool) = IntervalArithmetic._unsafe_interval(bareinterval(a), decoration(a), t)

# allocation free reshaping (cf. Issue #36313)

_no_alloc_reshape(a::Sequence{<:BaseSpace}) = coefficients(a)
_no_alloc_reshape(a::Sequence{<:TensorSpace}) = _no_alloc_reshape(coefficients(a), dimensions(space(a)))

_no_alloc_reshape(a, dims) = invoke(Base._reshape, Tuple{AbstractArray,typeof(dims)}, a, dims)
