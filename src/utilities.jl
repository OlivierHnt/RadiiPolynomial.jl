# safe equality

_safe_isequal(x, y) = x == y
_safe_isequal(x::Union{Interval,Complex{<:Interval}}, y::Union{Interval,Complex{<:Interval}}) =
    isequal_interval(x, y)

# allocation free reshaping (cf. Issue #36313)

_no_alloc_reshape(a, dims) = invoke(Base._reshape, Tuple{AbstractArray,typeof(dims)}, a, dims)
