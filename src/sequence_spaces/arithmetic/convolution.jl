function banach_rounding_order(bound::T, X::Ell1{GeometricWeight{T}}) where {T<:AbstractFloat}
    (rate(X.weight) ≤ 1) | isinf(bound) && return typemax(Int)
    ϵ = eps(T)
    bound ≤ ϵ && return 0
    order = log(bound/ϵ)/log(rate(X.weight))
    isinf(order) && return typemax(Int)
    return ceil(Int, order)
end
function banach_rounding_order(bound::T,  X::Ell1{AlgebraicWeight{T}}) where {T<:AbstractFloat}
    iszero(rate(X.weight)) | isinf(bound) && return typemax(Int)
    ϵ = eps(T)
    bound ≤ ϵ && return 0
    order = exp(log(bound/ϵ)/rate(X.weight))-1
    isinf(order) && return typemax(Int)
    return ceil(Int, order)
end

for T ∈ (:GeometricWeight, :AlgebraicWeight)
    @eval begin
        function banach_rounding_order(bound::Real, X::Ell1{<:$T})
            bound_, rate_ = promote(float(bound), float(rate(X.weight)))
            return banach_rounding_order(bound_, Ell1($T(rate_)))
        end

        banach_rounding_order(bound::Interval, X::Ell1{<:$T{<:Interval}}) =
            banach_rounding_order(sup(bound), Ell1($T(sup(rate(X.weight)))))

        banach_rounding_order(bound::Real, X::Ell1{<:$T{<:Interval}}) =
            banach_rounding_order(bound, Ell1($T(sup(rate(X.weight)))))

        banach_rounding_order(bound::Interval, X::Ell1{<:$T}) =
            banach_rounding_order(sup(bound), Ell1($T(rate(X.weight))))
    end
end

banach_rounding_order(bound::Real, X::Ell1{<:Tuple}) =
    map(wᵢ -> banach_rounding_order(bound, Ell1(wᵢ)), X.weight)

banach_rounding_order(bound::Interval, X::Ell1{<:Tuple}) = banach_rounding_order(sup(bound), X)

#

_interval_box(::Type{<:Interval}, x) = Interval(-x, x)
_interval_box(::Type{<:Complex{<:Interval}}, x) = (y = Interval(-x, x); Complex(y, y))

#

banach_rounding!(a, bound, X) = banach_rounding!(a, bound, banach_rounding_order(bound, X))

function banach_rounding!(a::Sequence{TensorSpace{T},<:AbstractVector{S}}, bound::Real, X::Ell1, rounding_order::NTuple{N,Int}) where {N,T<:NTuple{N,BaseSpace},S<:Union{Interval,Complex{<:Interval}}}
    bound ≥ 0 || return throw(ArgumentError("the bound must be positive"))
    space_a = space(a)
    M = typemax(Int)
    @inbounds for α ∈ indices(space_a)
        if mapreduce((i, ord) -> ifelse(ord == M, 0//1, ifelse(ord == 0, 1//1, abs(i) // ord)), +, α, rounding_order) ≥ 1
            μᵅ = bound / _getindex(X.weight, space_a, α)
            a[α] = _interval_box(S, sup(μᵅ))
        end
    end
    return a
end

# Taylor

function banach_rounding!(a::Sequence{Taylor,<:AbstractVector{T}}, bound::Real, X::Ell1{<:GeometricWeight}, rounding_order::Int) where {T<:Union{Interval,Complex{<:Interval}}}
    (bound ≥ 0) & (rounding_order ≥ 0) || return throw(ArgumentError("the bound and the rounding order must be positive"))
    if rounding_order ≤ order(a)
        ν⁻¹ = inv(rate(X.weight))
        μⁱ = bound / _getindex(X.weight, space(a), rounding_order)
        @inbounds for i ∈ rounding_order:order(a)
            a[i] = _interval_box(T, sup(μⁱ))
            μⁱ *= ν⁻¹
        end
    end
    return a
end

function banach_rounding!(a::Sequence{Taylor,<:AbstractVector{T}}, bound::Real, X::Ell1{<:AlgebraicWeight}, rounding_order::Int) where {T<:Union{Interval,Complex{<:Interval}}}
    (bound ≥ 0) & (rounding_order ≥ 0) || return throw(ArgumentError("the bound and the rounding order must be positive"))
    space_a = space(a)
    @inbounds for i ∈ rounding_order:order(a)
        μⁱ = bound / _getindex(X.weight, space_a, i)
        a[i] = _interval_box(T, sup(μⁱ))
    end
    return a
end

# Fourier

function banach_rounding!(a::Sequence{<:Fourier,<:AbstractVector{T}}, bound::Real, X::Ell1{<:GeometricWeight}, rounding_order::Int) where {T<:Union{Interval,Complex{<:Interval}}}
    (bound ≥ 0) & (rounding_order ≥ 0) || return throw(ArgumentError("the bound and the rounding order must be positive"))
    if rounding_order ≤ order(a)
        ν⁻¹ = inv(rate(X.weight))
        μⁱ = bound / _getindex(X.weight, space(a), rounding_order)
        @inbounds for i ∈ rounding_order:order(a)
            x = _interval_box(T, sup(μⁱ))
            a[i] = x
            a[-i] = x
            μⁱ *= ν⁻¹
        end
    end
    return a
end

function banach_rounding!(a::Sequence{<:Fourier,<:AbstractVector{T}}, bound::Real, X::Ell1{<:AlgebraicWeight}, rounding_order::Int) where {T<:Union{Interval,Complex{<:Interval}}}
    (bound ≥ 0) & (rounding_order ≥ 0) || return throw(ArgumentError("the bound and the rounding order must be positive"))
    space_a = space(a)
    @inbounds for i ∈ rounding_order:order(a)
        μⁱ = bound / _getindex(X.weight, space_a, i)
        x = _interval_box(T, sup(μⁱ))
        a[i] = x
        a[-i] = x
    end
    return a
end

# Chebyshev

function banach_rounding!(a::Sequence{Chebyshev,<:AbstractVector{T}}, bound::Real, X::Ell1{<:GeometricWeight}, rounding_order::Int) where {T<:Union{Interval,Complex{<:Interval}}}
    (bound ≥ 0) & (rounding_order ≥ 0) || return throw(ArgumentError("the bound and the rounding order must be positive"))
    if rounding_order ≤ order(a)
        ν⁻¹ = inv(rate(X.weight))
        μⁱ = bound / _getindex(X.weight, space(a), rounding_order)
        @inbounds for i ∈ rounding_order:order(a)
            a[i] = _interval_box(T, sup(μⁱ))
            μⁱ *= ν⁻¹
        end
    end
    return a
end

function banach_rounding!(a::Sequence{Chebyshev,<:AbstractVector{T}}, bound::Real, X::Ell1{<:AlgebraicWeight}, rounding_order::Int) where {T<:Union{Interval,Complex{<:Interval}}}
    (bound ≥ 0) & (rounding_order ≥ 0) || return throw(ArgumentError("the bound and the rounding order must be positive"))
    space_a = space(a)
    @inbounds for i ∈ rounding_order:order(a)
        μⁱ = bound / _getindex(X.weight, space_a, i)
        a[i] = _interval_box(T, sup(μⁱ))
    end
    return a
end

#

"""
    *(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})

Compute the discrete convolution (associated with `space(a)` and `space(b)`) of
`a` and `b`.

See also: [`mul!(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Number, ::Number)`](@ref),
[`^(::Sequence{<:SequenceSpace}, ::Int)`](@ref), [`banach_rounding_mul`](@ref)
and [`banach_rounding_mul!`](@ref) and [`banach_rounding_pow`](@ref).
"""
function Base.:*(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    new_space = image(*, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _mul!(c, a, b, true, false)
    return c
end

function mul_bar(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    new_space = image(mul_bar, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _mul!(c, a, b, true, false)
    return c
end

"""
    mul!(c::Sequence{<:SequenceSpace}, a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}, α::Number, β::Number)

Compute `project(a * b, space(c)) * α + c * β` in-place. The result is stored in
`c` by overwriting it.

Note: `c` must not be aliased with either `a` or `b`.

See also: [`*(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace})`](@ref),
[`^(::Sequence{<:SequenceSpace}, ::Int)`](@ref), [`banach_rounding_mul`](@ref),
[`banach_rounding_mul!`](@ref) and [`banach_rounding_pow`](@ref).
"""
function mul!(c::Sequence{<:SequenceSpace}, a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}, α::Number, β::Number)
    space_c = space(c)
    new_space = image(*, space(a), space(b))
    _iscompatible(space_c, new_space) || return throw(ArgumentError("spaces must be compatible: c has space $space_c, a*b has space $new_space"))
    _mul!(c, a, b, α, β)
    return c
end
function _mul!(c::Sequence{<:SequenceSpace}, a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}, α::Number, β::Number)
    if iszero(β)
        coefficients(c) .= zero(eltype(c))
    elseif !isone(β)
        coefficients(c) .*= β
    end
    _add_mul!(c, a, b, α)
    return c
end

"""
    banach_rounding_mul(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}, X::Ell1)

Compute the discrete convolution (associated with `space(a)` and `space(b)`) of
`a` and `b`. A cut-off order is estimated such that the coefficients of the
output beyond this order are rigorously enclosed.

See also: [`banach_rounding_mul!`](@ref), [`banach_rounding_pow`](@ref),
[`*(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace})`](@ref),
[`mul!(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Number, ::Number)`](@ref)
and [`^(::Sequence{<:SequenceSpace}, ::Int)`](@ref).
"""
function banach_rounding_mul(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}, X::Ell1)
    new_space = image(*, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    bound_ab = norm(a, X) * norm(b, X)
    _add_mul!(c, a, b, true, bound_ab, X)
    return c
end

function banach_rounding_mul_bar(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}, X::Ell1)
    new_space = image(mul_bar, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    bound_ab = norm(a, X) * norm(b, X)
    _add_mul!(c, a, b, true, bound_ab, X)
    return c
end

"""
    banach_rounding_mul!(c::Sequence{<:SequenceSpace}, a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}, X::Ell1)

Compute `project(banach_rounding_mul(a, b, X), space(c))` in-place. The result
is stored in `c` by overwriting it.

Note: `c` must not be aliased with either `a` or `b`.

See also: [`banach_rounding_mul`](@ref), [`banach_rounding_pow`](@ref),
[`*(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace})`](@ref),
[`mul!(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Number, ::Number)`](@ref)
and [`^(::Sequence{<:SequenceSpace}, ::Int)`](@ref).
"""
function banach_rounding_mul!(c::Sequence{<:SequenceSpace}, a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}, X::Ell1)
    space_c = space(c)
    new_space = image(*, space(a), space(b))
    _iscompatible(space_c, new_space) || return throw(ArgumentError("spaces must be compatible: c has space $space_c, a*b has space $new_space"))
    coefficients(c) .= zero(eltype(c))
    bound_ab = norm(a, X) * norm(b, X)
    _add_mul!(c, a, b, true, bound_ab, X)
    return c
end

function _add_mul!(c::Sequence{<:BaseSpace}, a, b, α)
    space_a = space(a)
    space_b = space(b)
    _0 = zero(promote_type(eltype(a), eltype(b)))
    @inbounds for i ∈ indices(space(c))
        cᵢ = _0
        @inbounds @simd for j ∈ _convolution_indices(space_a, space_b, i)
            cᵢ += a[_extract_valid_index(space_a, i, j)] * b[_extract_valid_index(space_b, j, 0)]
        end
        c[i] += cᵢ * α
    end
    return c
end

function _add_mul!(c::Sequence{TensorSpace{T}}, a, b, α) where {N,T<:NTuple{N,BaseSpace}}
    space_a = space(a)
    space_b = space(b)
    _0 = zero(promote_type(eltype(a), eltype(b)))
    _0_ = ntuple(_ -> 0, Val(N))
    @inbounds for i ∈ indices(space(c))
        cᵢ = _0
        @inbounds for j ∈ _convolution_indices(space_a, space_b, i)
            cᵢ += a[_extract_valid_index(space_a, i, j)] * b[_extract_valid_index(space_b, j, _0_)]
        end
        c[i] += cᵢ * α
    end
    return c
end

function _add_mul!(c::Sequence{<:BaseSpace}, a, b, α, bound_ab, X)
    rounding_order = banach_rounding_order(bound_ab, X)
    space_c = space(c)
    space_a = space(a)
    space_b = space(b)
    _0 = zero(promote_type(eltype(a), eltype(b)))
    CoefType = eltype(c)
    @inbounds for i ∈ indices(space_c)
        if abs(i) ≥ rounding_order
            μⁱ = bound_ab / _getindex(X.weight, space_c, i)
            c[i] += _interval_box(CoefType, sup(α * μⁱ))
        else
            cᵢ = _0
            @inbounds @simd for j ∈ _convolution_indices(space_a, space_b, i)
                cᵢ += a[_extract_valid_index(space_a, i, j)] * b[_extract_valid_index(space_b, j, 0)]
            end
            c[i] += cᵢ * α
        end
    end
    return c
end

function _add_mul!(c::Sequence{TensorSpace{T}}, a, b, α, bound_ab, X) where {N,T<:NTuple{N,BaseSpace}}
    rounding_order = banach_rounding_order(bound_ab, X)
    space_c = space(c)
    space_a = space(a)
    space_b = space(b)
    M = typemax(Int)
    _0 = zero(promote_type(eltype(a), eltype(b)))
    _0_ = ntuple(_ -> 0, Val(N))
    CoefType = eltype(c)
    @inbounds for i ∈ indices(space_c)
        if mapreduce((i′, ord) -> ifelse(ord == M, 0//1, ifelse(ord == 0, 1//1, abs(i′) // ord)), +, i, rounding_order) ≥ 1
            μⁱ = bound_ab / _getindex(X.weight, space_c, i)
            c[i] += _interval_box(CoefType, sup(α * μⁱ))
        else
            cᵢ = _0
            @inbounds for j ∈ _convolution_indices(space_a, space_b, i)
                cᵢ += a[_extract_valid_index(space_a, i, j)] * b[_extract_valid_index(space_b, j, _0_)]
            end
            c[i] += cᵢ * α
        end
    end
    return c
end

_convolution_indices(s₁::TensorSpace, s₂::TensorSpace, i) =
    TensorIndices(map(_convolution_indices, spaces(s₁), spaces(s₂), i))

_convolution_indices(s₁::Taylor, s₂::Taylor, i) = max(i-order(s₁), 0):min(i, order(s₂))

_convolution_indices(s₁::Fourier, s₂::Fourier, i) = max(i-order(s₁), -order(s₂)):min(i+order(s₁), order(s₂))

_convolution_indices(s₁::Chebyshev, s₂::Chebyshev, i) = max(i-order(s₁), -order(s₂)):min(i+order(s₁), order(s₂))

#

"""
    ^(a::Sequence{<:SequenceSpace}, n::Int)

Compute the discrete convolution (associated with `space(a)`) of `a` with itself
`n` times.

See also: [`*(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace})`](@ref),
[`mul!(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Number, ::Number)`](@ref),
[`banach_rounding_mul`](@ref), [`banach_rounding_mul!`](@ref) and [`banach_rounding_pow`](@ref).
"""
function Base.:^(a::Sequence{<:SequenceSpace}, n::Int)
    n < 0 && return throw(DomainError(n, "^ is only defined for positive integers"))
    n == 0 && return one(a)
    n == 1 && return copy(a)
    n == 2 && return _sqr(a)
    # power by squaring
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        a = _sqr(a)
    end
    c = a
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            a = _sqr(a)
        end
        c = c * a
    end
    return c
end

function pow_bar(a::Sequence{<:SequenceSpace}, n::Int)
    n < 0 && return throw(DomainError(n, "pow_bar is only defined for positive integers"))
    n == 0 && return one(a)
    n == 1 && return copy(a)
    n == 2 && return _sqr_bar(a)
    # power by squaring
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        a = _sqr_bar(a)
    end
    c = a
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            a = _sqr_bar(a)
        end
        c = mul_bar(c, a)
    end
    return c
end

function _sqr(a::Sequence{<:SequenceSpace})
    new_space = image(^, space(a), 2)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    _add_sqr!(c, a)
    return c
end

function _sqr_bar(a::Sequence{<:SequenceSpace})
    new_space = image(pow_bar, space(a), 2)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    _add_sqr!(c, a)
    return c
end

"""
    banach_rounding_pow(a::Sequence{<:SequenceSpace}, n::Int, X::Ell1)

Compute the discrete convolution (associated with `space(a)`) of `a` with itself
`n` times. A cut-off order is estimated such that the coefficients of the output
beyond this order are rigorously enclosed.

See also: [`banach_rounding_mul`](@ref), [`banach_rounding_mul!`](@ref),
[`*(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace})`](@ref),
[`mul!(::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Sequence{<:SequenceSpace}, ::Number, ::Number)`](@ref)
and [`^(::Sequence{<:SequenceSpace}, ::Int)`](@ref).
"""
function banach_rounding_pow(a::Sequence{<:SequenceSpace}, n::Int, X::Ell1)
    n < 0 && return throw(DomainError(n, "banach_rounding_pow is only defined for positive integers"))
    n == 0 && return one(a)
    n == 1 && return copy(a)
    n == 2 && return _banach_rounding_sqr(a, X)
    # power by squaring
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        a = _banach_rounding_sqr(a, X)
    end
    c = a
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            a = _banach_rounding_sqr(a, X)
        end
        c = banach_rounding_mul(c, a, X)
    end
    return c
end

function banach_rounding_pow_bar(a::Sequence{<:SequenceSpace}, n::Int, X::Ell1)
    n < 0 && return throw(DomainError(n, "banach_rounding_pow_bar is only defined for positive integers"))
    n == 0 && return one(a)
    n == 1 && return copy(a)
    n == 2 && return _banach_rounding_sqr_bar(a, X)
    # power by squaring
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        a = _banach_rounding_sqr_bar(a, X)
    end
    c = a
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            a = _banach_rounding_sqr_bar(a, X)
        end
        c = banach_rounding_mul_bar(c, a, X)
    end
    return c
end

function _banach_rounding_sqr(a::Sequence{<:SequenceSpace}, X::Ell1)
    new_space = image(^, space(a), 2)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    norm_a = norm(a, X)
    bound_a² = norm_a * norm_a
    _add_sqr!(c, a, bound_a², X)
    return c
end

function _banach_rounding_sqr_bar(a::Sequence{<:SequenceSpace}, X::Ell1)
    new_space = image(pow_bar, space(a), 2)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    norm_a = norm(a, X)
    bound_a² = norm_a * norm_a
    _add_sqr!(c, a, bound_a², X)
    return c
end

_add_sqr!(c::Sequence, a) = _add_mul!(c, a, a, true)
_add_sqr!(c::Sequence, a, bound_a², X) = _add_mul!(c, a, a, true, bound_a², X)

# Taylor

function _add_sqr!(c::Sequence{Taylor}, a)
    order_a = order(space(a))
    _0 = zero(eltype(a))
    @inbounds a₀ = a[0]
    @inbounds c[0] += a₀ * a₀
    @inbounds for i ∈ 1:order(space(c))
        cᵢ = _0
        i_odd = i%2
        i_end = (i-2+i_odd)÷2
        @inbounds for j ∈ max(i-order_a, 0):i_end
            cᵢ += a[i-j] * a[j]
        end
        if iszero(i_odd)
            a_i½ = a[i÷2]
            c[i] += 2cᵢ + a_i½ * a_i½
        else
            c[i] += 2cᵢ
        end
    end
    return c
end

function _add_sqr!(c::Sequence{Taylor}, a, bound_a², X)
    rounding_order = banach_rounding_order(bound_a², X)
    if rounding_order > 0
        order_a = order(space(a))
        _0 = zero(eltype(a))
        @inbounds a₀ = a[0]
        @inbounds c[0] += a₀ * a₀
        @inbounds for i ∈ 1:min(order(space(c)), rounding_order-1)
            cᵢ = _0
            i_odd = i%2
            i_end = (i-2+i_odd)÷2
            @inbounds for j ∈ max(i-order_a, 0):i_end
                cᵢ += a[i-j] * a[j]
            end
            if iszero(i_odd)
                a_i½ = a[i÷2]
                c[i] += 2cᵢ + a_i½ * a_i½
            else
                c[i] += 2cᵢ
            end
        end
    end
    banach_rounding!(c, bound_a², X, rounding_order)
    return c
end

# Fourier

function _add_sqr!(c::Sequence{<:Fourier}, a)
    order_a = order(space(a))
    _0 = zero(eltype(a))
    c₀ = _0
    @inbounds for j ∈ 1:order_a
        c₀ += a[j] * a[-j]
    end
    @inbounds a₀ = a[0]
    @inbounds c[0] += 2c₀ + a₀ * a₀
    @inbounds for i ∈ 1:order(space(c))
        cᵢ = c₋ᵢ = _0
        i½, i_odd = divrem(i, 2)
        @inbounds for j ∈ i½+1:order_a
            cᵢ += a[i-j] * a[j]
            c₋ᵢ += a[j-i] * a[-j]
        end
        if iszero(i_odd)
            a_i½ = a[i½]
            a_neg_i½ = a[-i½]
            c[i] += 2cᵢ + a_i½ * a_i½
            c[-i] += 2c₋ᵢ + a_neg_i½ * a_neg_i½
        else
            c[i] += 2cᵢ
            c[-i] += 2c₋ᵢ
        end
    end
    return c
end

function _add_sqr!(c::Sequence{<:Fourier}, a, bound_a², X)
    rounding_order = banach_rounding_order(bound_a², X)
    if rounding_order > 0
        order_a = order(space(a))
        _0 = zero(eltype(a))
        c₀ = _0
        @inbounds for j ∈ 1:order_a
            c₀ += a[j] * a[-j]
        end
        @inbounds a₀ = a[0]
        @inbounds c[0] += 2c₀ + a₀ * a₀
        @inbounds for i ∈ 1:min(order(space(c)), rounding_order-1)
            cᵢ = c₋ᵢ = _0
            i½, i_odd = divrem(i, 2)
            @inbounds for j ∈ i½+1:order_a
                cᵢ += a[i-j] * a[j]
                c₋ᵢ += a[j-i] * a[-j]
            end
            if iszero(i_odd)
                a_i½ = a[i½]
                a_neg_i½ = a[-i½]
                c[i] += 2cᵢ + a_i½ * a_i½
                c[-i] += 2c₋ᵢ + a_neg_i½ * a_neg_i½
            else
                c[i] += 2cᵢ
                c[-i] += 2c₋ᵢ
            end
        end
    end
    banach_rounding!(c, bound_a², X, rounding_order)
    return c
end

# Chebyshev

function _add_sqr!(c::Sequence{Chebyshev}, a)
    order_a = order(space(a))
    _0 = zero(eltype(a))
    c₀ = _0
    @inbounds for j ∈ 1:order_a
        aⱼ = a[j]
        c₀ += aⱼ * aⱼ
    end
    @inbounds a₀ = a[0]
    @inbounds c[0] += 2c₀ + a₀ * a₀
    @inbounds for i ∈ 1:order(space(c))
        cᵢ = _0
        i½, i_odd = divrem(i, 2)
        @inbounds for j ∈ i½+1:order_a
            cᵢ += a[abs(i-j)] * a[j]
        end
        if iszero(i_odd)
            a_i½ = a[i½]
            c[i] += 2cᵢ + a_i½ * a_i½
        else
            c[i] += 2cᵢ
        end
    end
    return c
end

function _add_sqr!(c::Sequence{Chebyshev}, a, bound_a², X)
    rounding_order = banach_rounding_order(bound_a², X)
    if rounding_order > 0
        order_a = order(space(a))
        _0 = zero(eltype(a))
        c₀ = _0
        @inbounds for j ∈ 1:order_a
            aⱼ = a[j]
            c₀ += aⱼ * aⱼ
        end
        @inbounds a₀ = a[0]
        @inbounds c[0] += 2c₀ + a₀ * a₀
        @inbounds for i ∈ 1:min(order(space(c)), rounding_order-1)
            cᵢ = _0
            i½, i_odd = divrem(i, 2)
            @inbounds for j ∈ i½+1:order_a
                cᵢ += a[abs(i-j)] * a[j]
            end
            if iszero(i_odd)
                a_i½ = a[i½]
                c[i] += 2cᵢ + a_i½ * a_i½
            else
                c[i] += 2cᵢ
            end
        end
    end
    banach_rounding!(c, bound_a², X, rounding_order)
    return c
end
