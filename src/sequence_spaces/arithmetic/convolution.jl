function banach_rounding_order(bound::T, X::ℓ¹{GeometricWeight{T}}) where {T<:AbstractFloat}
    isone(rate(X.weight)) | isinf(bound) && return typemax(Int)
    ϵ = eps(T)
    bound ≤ ϵ && return 0
    order = log(bound/ϵ)/log(rate(X.weight))
    isinf(order) && return typemax(Int)
    return ceil(Int, order)
end
function banach_rounding_order(bound::T,  X::ℓ¹{AlgebraicWeight{T}}) where {T<:AbstractFloat}
    iszero(rate(X.weight)) | isinf(bound) && return typemax(Int)
    ϵ = eps(T)
    bound ≤ ϵ && return 0
    order = exp(log(bound/ϵ)/rate(X.weight))-1
    isinf(order) && return typemax(Int)
    return ceil(Int, order)
end

for T ∈ (:GeometricWeight, :AlgebraicWeight)
    @eval begin
        function banach_rounding_order(bound::Real, X::ℓ¹{<:$T})
            bound_, rate_ = promote(float(bound), float(rate(X.weight)))
            return banach_rounding_order(bound_, ℓ¹($T(rate_)))
        end

        banach_rounding_order(bound::Interval, X::ℓ¹{<:$T{<:Interval}}) =
            banach_rounding_order(sup(bound), ℓ¹($T(sup(rate(X.weight)))))

        banach_rounding_order(bound::Real, X::ℓ¹{<:$T{<:Interval}}) =
            banach_rounding_order(bound, ℓ¹($T(sup(rate(X.weight)))))

        banach_rounding_order(bound::Interval, X::ℓ¹{<:$T}) =
            banach_rounding_order(sup(bound), ℓ¹($T(rate(X.weight))))
    end
end

banach_rounding_order(bound::Real, X::ℓ¹{<:Tuple}) =
    map(wᵢ -> banach_rounding_order(bound, ℓ¹(wᵢ)), X.weight)

banach_rounding_order(bound::Interval, X::ℓ¹{<:Tuple}) = banach_rounding_order(sup(bound), X)

#

banach_rounding!(a, bound, X) = banach_rounding!(a, bound, banach_rounding_order(bound, X))

function banach_rounding!(a::Sequence{TensorSpace{T},<:AbstractVector{<:Interval}}, bound::Real, X::ℓ¹, rounding_order::NTuple{N,Int}) where {N,T<:NTuple{N,BaseSpace}}
    bound ≥ 0 || return throw(DomainError(bound, "bound must be positive"))
    M = typemax(Int)
    w = map(ord -> ifelse(ord == 0, 1, ord), rounding_order)
    space_a = space(a)
    @inbounds for α ∈ indices(space_a)
        if mapreduce((wᵢ, αᵢ) -> wᵢ == M ? 0//1 : abs(αᵢ) // wᵢ, +, w, α) ≥ 1
            μᵅ = bound / _getindex(X.weight, space_a, α)
            sup_μᵅ = sup(μᵅ)
            a[α] = Interval(-sup_μᵅ, sup_μᵅ)
        end
    end
    return a
end

function banach_rounding!(a::Sequence{TensorSpace{T},<:AbstractVector{<:Complex{<:Interval}}}, bound::Real, X::ℓ¹, rounding_order::NTuple{N,Int}) where {N,T<:NTuple{N,BaseSpace}}
    bound ≥ 0 || return throw(DomainError(bound, "bound must be positive"))
    M = typemax(Int)
    w = map(ord -> ifelse(ord == 0, 1, ord), rounding_order)
    space_a = space(a)
    @inbounds for α ∈ indices(space_a)
        if mapreduce((wᵢ, αᵢ) -> wᵢ == M ? 0//1 : abs(αᵢ) // wᵢ, +, w, α) ≥ 1
            μᵅ = bound / _getindex(X.weight, space_a, α)
            sup_μᵅ = sup(μᵅ)
            x = Interval(-sup_μᵅ, sup_μᵅ)
            a[α] = complex(x, x)
        end
    end
    return a
end

# Taylor

function banach_rounding!(a::Sequence{Taylor,<:AbstractVector{<:Interval}}, bound::Real, X::ℓ¹{<:GeometricWeight}, rounding_order::Int)
    bound ≥ 0 || return throw(DomainError(bound, "bound must be positive"))
    if rounding_order ≤ order(a)
        ν⁻¹ = inv(rate(X.weight))
        μⁱ = bound / _getindex(X.weight, space(a), rounding_order)
        @inbounds for i ∈ rounding_order:order(a)
            sup_μⁱ = sup(μⁱ)
            a[i] = Interval(-sup_μⁱ, sup_μⁱ)
            μⁱ *= ν⁻¹
        end
    end
    return a
end
function banach_rounding!(a::Sequence{Taylor,<:AbstractVector{<:Complex{<:Interval}}}, bound::Real, X::ℓ¹{<:GeometricWeight}, rounding_order::Int)
    bound ≥ 0 || return throw(DomainError(bound, "bound must be positive"))
    if rounding_order ≤ order(a)
        ν⁻¹ = inv(rate(X.weight))
        μⁱ = bound / _getindex(X.weight, space(a), rounding_order)
        @inbounds for i ∈ rounding_order:order(a)
            sup_μⁱ = sup(μⁱ)
            x = Interval(-sup_μⁱ, sup_μⁱ)
            a[i] = complex(x, x)
            μⁱ *= ν⁻¹
        end
    end
    return a
end

function banach_rounding!(a::Sequence{Taylor,<:AbstractVector{<:Interval}}, bound::Real, X::ℓ¹{<:AlgebraicWeight}, rounding_order::Int)
    bound ≥ 0 || return throw(DomainError(bound, "bound must be positive"))
    space_a = space(a)
    @inbounds for i ∈ rounding_order:order(a)
        sup_μⁱ = sup(bound / _getindex(X.weight, space_a, i))
        a[i] = Interval(-sup_μⁱ, sup_μⁱ)
    end
    return a
end
function banach_rounding!(a::Sequence{Taylor,<:AbstractVector{<:Complex{<:Interval}}}, bound::Real, X::ℓ¹{<:AlgebraicWeight}, rounding_order::Int)
    bound ≥ 0 || return throw(DomainError(bound, "bound must be positive"))
    space_a = space(a)
    @inbounds for i ∈ rounding_order:order(a)
        sup_μⁱ = sup(bound / _getindex(X.weight, space_a, i))
        x = Interval(-sup_μⁱ, sup_μⁱ)
        a[i] = complex(x, x)
    end
    return a
end

# Fourier

function banach_rounding!(a::Sequence{<:Fourier,<:AbstractVector{<:Interval}}, bound::Real, X::ℓ¹{<:GeometricWeight}, rounding_order::Int)
    bound ≥ 0 || return throw(DomainError(bound, "bound must be positive"))
    if rounding_order ≤ order(a)
        ν⁻¹ = inv(rate(X.weight))
        μⁱ = bound / _getindex(X.weight, space(a), rounding_order)
        @inbounds for i ∈ rounding_order:order(a)
            sup_μⁱ = sup(μⁱ)
            x = Interval(-sup_μⁱ, sup_μⁱ)
            a[i] = x
            a[-i] = x
            μⁱ *= ν⁻¹
        end
    end
    return a
end
function banach_rounding!(a::Sequence{<:Fourier,<:AbstractVector{<:Complex{<:Interval}}}, bound::Real, X::ℓ¹{<:GeometricWeight}, rounding_order::Int)
    bound ≥ 0 || return throw(DomainError(bound, "bound must be positive"))
    if rounding_order ≤ order(a)
        ν⁻¹ = inv(rate(X.weight))
        μⁱ = bound / _getindex(X.weight, space(a), rounding_order)
        @inbounds for i ∈ rounding_order:order(a)
            sup_μⁱ = sup(μⁱ)
            x = Interval(-sup_μⁱ, sup_μⁱ)
            complex_x = complex(x, x)
            a[i] = complex_x
            a[-i] = complex_x
            μⁱ *= ν⁻¹
        end
    end
    return a
end

function banach_rounding!(a::Sequence{<:Fourier,<:AbstractVector{<:Interval}}, bound::Real, X::ℓ¹{<:AlgebraicWeight}, rounding_order::Int)
    bound ≥ 0 || return throw(DomainError(bound, "bound must be positive"))
    space_a = space(a)
    @inbounds for i ∈ rounding_order:order(a)
        sup_μⁱ = sup(bound / _getindex(X.weight, space_a, i))
        x = Interval(-sup_μⁱ, sup_μⁱ)
        a[i] = x
        a[-i] = x
    end
    return a
end
function banach_rounding!(a::Sequence{<:Fourier,<:AbstractVector{<:Complex{<:Interval}}}, bound::Real, X::ℓ¹{<:AlgebraicWeight}, rounding_order::Int)
    bound ≥ 0 || return throw(DomainError(bound, "bound must be positive"))
    space_a = space(a)
    @inbounds for i ∈ rounding_order:order(a)
        sup_μⁱ = sup(bound / _getindex(X.weight, space_a, i))
        x = Interval(-sup_μⁱ, sup_μⁱ)
        complex_x = complex(x, x)
        a[i] = complex_x
        a[-i] = complex_x
    end
    return a
end

# Chebyshev

function banach_rounding!(a::Sequence{Chebyshev,<:AbstractVector{<:Interval}}, bound::Real, X::ℓ¹{<:GeometricWeight}, rounding_order::Int)
    bound ≥ 0 || return throw(DomainError(bound, "bound must be positive"))
    if rounding_order ≤ order(a)
        ν⁻¹ = inv(rate(X.weight))
        μⁱ = bound / _getindex(X.weight, space(a), rounding_order)
        @inbounds for i ∈ rounding_order:order(a)
            sup_μⁱ = sup(μⁱ)
            a[i] = Interval(-sup_μⁱ, sup_μⁱ)
            μⁱ *= ν⁻¹
        end
    end
    return a
end
function banach_rounding!(a::Sequence{Chebyshev,<:AbstractVector{<:Complex{<:Interval}}}, bound::Real, X::ℓ¹{<:GeometricWeight}, rounding_order::Int)
    bound ≥ 0 || return throw(DomainError(bound, "bound must be positive"))
    if rounding_order ≤ order(a)
        ν⁻¹ = inv(rate(X.weight))
        μⁱ = bound / _getindex(X.weight, space(a), rounding_order)
        @inbounds for i ∈ rounding_order:order(a)
            sup_μⁱ = sup(μⁱ)
            x = Interval(-sup_μⁱ, sup_μⁱ)
            a[i] = complex(x, x)
            μⁱ *= ν⁻¹
        end
    end
    return a
end

function banach_rounding!(a::Sequence{Chebyshev,<:AbstractVector{<:Interval}}, bound::Real, X::ℓ¹{<:AlgebraicWeight}, rounding_order::Int)
    bound ≥ 0 || return throw(DomainError(bound, "bound must be positive"))
    space_a = space(a)
    @inbounds for i ∈ rounding_order:order(a)
        sup_μⁱ = sup(bound / _getindex(X.weight, space_a, i))
        a[i] = Interval(-sup_μⁱ, sup_μⁱ)
    end
    return a
end
function banach_rounding!(a::Sequence{Chebyshev,<:AbstractVector{<:Complex{<:Interval}}}, bound::Real, X::ℓ¹{<:AlgebraicWeight}, rounding_order::Int)
    bound ≥ 0 || return throw(DomainError(bound, "bound must be positive"))
    space_a = space(a)
    @inbounds for i ∈ rounding_order:order(a)
        sup_μⁱ = sup(bound / _getindex(X.weight, space_a, i))
        x = Interval(-sup_μⁱ, sup_μⁱ)
        a[i] = complex(x, x)
    end
    return a
end

#

_max_order(::BaseSpace) = typemax(Int)
_max_order(::BaseSpace, rounding_order::Int) = rounding_order-1
_max_order(::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} = ntuple(i -> typemax(Int), Val(N))
_max_order(::TensorSpace{<:NTuple{N,BaseSpace}}, rounding_order::NTuple{N,Int}) where {N} =
    ntuple(i -> rounding_order[i]-1, Val(N))

function Base.:*(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    new_space = image(*, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _mul!(c, a, b, true, false)
    return c
end
function *̄(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    new_space = image(*̄, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _mul!(c, a, b, true, false)
    return c
end
function LinearAlgebra.mul!(c::Sequence{<:SequenceSpace}, a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}, α::Number, β::Number)
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
    _add_mul!(c, a, b, α, _max_order(space(c)))
    return c
end

function banach_rounding_mul(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}, X::ℓ¹)
    bound_ab = norm(a, X) * norm(b, X)
    rounding_order = banach_rounding_order(bound_ab, X)
    new_space = image(*, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    banach_rounding!(c, bound_ab, X, rounding_order)
    _add_mul!(c, a, b, true, _max_order(new_space, rounding_order))
    return c
end
function banach_rounding_mul_bar(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}, X::ℓ¹)
    bound_ab = norm(a, X) * norm(b, X)
    rounding_order = banach_rounding_order(bound_ab, X)
    new_space = image(*̄, space(a), space(b))
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    banach_rounding!(c, bound_ab, X, rounding_order)
    _add_mul!(c, a, b, true, _max_order(new_space, rounding_order))
    return c
end
function banach_rounding_mul!(c::Sequence{<:SequenceSpace}, a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}, X::ℓ¹)
    space_c = space(c)
    new_space = image(*, space(a), space(b))
    _iscompatible(space_c, new_space) || return throw(ArgumentError("spaces must be compatible: c has space $space_c, a*b has space $new_space"))
    bound_ab = norm(a, X) * norm(b, X)
    rounding_order = banach_rounding_order(bound_ab, X)
    coefficients(c) .= zero(eltype(c))
    banach_rounding!(c, bound_ab, X, rounding_order)
    _add_mul!(c, a, b, true, _max_order(space_c, rounding_order))
    return c
end

function _add_mul!(c::Sequence{<:BaseSpace}, a, b, α, max_order)
    C = coefficients(c)
    A = coefficients(a)
    B = coefficients(b)
    _add_mul!(C, A, B, α, space(c), space(a), space(b), max_order)
    return c
end

function _add_mul!(c::Sequence{<:TensorSpace}, a, b, α, max_order)
    space_c = space(c)
    space_a = space(a)
    space_b = space(b)
    C = _no_alloc_reshape(coefficients(c), dimensions(space_c))
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    B = _no_alloc_reshape(coefficients(b), dimensions(space_b))
    _add_mul!(C, A, B, α, space_c, space_a, space_b, max_order, 0//1)
    return c
end

function _add_mul!(C, A, B, α, space_c::BaseSpace, space_a, space_b, max_order)
    ord = min(order(space_c), max_order)
    for i ∈ _convolution_indices(space_c, ord)
        _convolution!(C, A, B, α, space_c, space_a, space_b, i)
    end
    return C
end

function _add_mul!(C, A, B, α, space_c::TensorSpace{<:NTuple{N,BaseSpace}}, space_a, space_b, max_order, n) where {N}
    remaining_space_c = Base.front(space_c)
    remaining_space_a = Base.front(space_a)
    remaining_space_b = Base.front(space_b)
    remaining_max_order = Base.front(max_order)
    @inbounds current_space_c = space_c[N]
    @inbounds current_space_a = space_a[N]
    @inbounds current_space_b = space_b[N]
    @inbounds current_max_order = max_order[N]
    current_order_c = order(current_space_c)
    if current_max_order == typemax(Int)
        for i ∈ _convolution_indices(current_space_c, current_order_c)
            _convolution!(C, A, B, α, current_space_c, current_space_a, current_space_b, remaining_space_c, remaining_space_a, remaining_space_b, remaining_max_order, n, i)
        end
    else
        ord = min(current_order_c, floor(Int, current_max_order * (1 - n)))
        w = ifelse(current_max_order == 0, 1, current_max_order)
        for i ∈ _convolution_indices(current_space_c, ord)
            n_ = n + abs(i) // w
            _convolution!(C, A, B, α, current_space_c, current_space_a, current_space_b, remaining_space_c, remaining_space_a, remaining_space_b, remaining_max_order, n_, i)
        end
    end
    return C
end

function _add_mul!(C, A, B, α, space_c::TensorSpace{<:Tuple{BaseSpace}}, space_a, space_b, max_order, n)
    @inbounds current_space_c = space_c[1]
    @inbounds current_space_a = space_a[1]
    @inbounds current_space_b = space_b[1]
    @inbounds current_max_order = max_order[1]
    current_order_c = order(current_space_c)
    ord = current_max_order == typemax(Int) ? current_order_c : min(current_order_c, floor(Int, current_max_order * (1 - n)))
    for i ∈ _convolution_indices(current_space_c, ord)
        _convolution!(C, A, B, α, current_space_c, current_space_a, current_space_b, i)
    end
    return C
end

# Taylor

_convolution_indices(::Taylor, order) = 0:order

function _convolution!(C, A, B, α, space_c::Taylor, space_a::Taylor, space_b::Taylor, i)
    order_a = order(space_a)
    order_b = order(space_b)
    v = zero(promote_type(eltype(A), eltype(B)))
    @inbounds for j ∈ max(i-order_a, 0):min(i, order_b)
        v += A[i-j+1] * B[j+1]
    end
    C[i+1] += v * α
    return C
end

function _convolution!(C::AbstractArray{T,N}, A, B, α, current_space_c::Taylor, current_space_a::Taylor, current_space_b::Taylor, remaining_space_c, remaining_space_a, remaining_space_b, remaining_max_order, n, i) where {T,N}
    current_order_a = order(current_space_a)
    current_order_b = order(current_space_b)
    @inbounds Cᵢ = selectdim(C, N, i+1)
    @inbounds for j ∈ max(i-current_order_a, 0):min(i, current_order_b)
        _add_mul!(Cᵢ, selectdim(A, N, i-j+1), selectdim(B, N, j+1), α, remaining_space_c, remaining_space_a, remaining_space_b, remaining_max_order, n)
    end
    return C
end

# Fourier

_convolution_indices(::Fourier, order) = -order:order

function _convolution!(C, A, B, α, space_c::Fourier, space_a::Fourier, space_b::Fourier, i)
    order_c = order(space_c)
    order_a = order(space_a)
    order_b = order(space_b)
    v = zero(promote_type(eltype(A), eltype(B)))
    @inbounds for j ∈ max(i-order_a, -order_b):min(i+order_a, order_b)
        v += A[i-j+order_a+1] * B[j+order_b+1]
    end
    C[i+order_c+1] += v * α
    return C
end

function _convolution!(C::AbstractArray{T,N}, A, B, α, current_space_c::Fourier, current_space_a::Fourier, current_space_b::Fourier, remaining_space_c, remaining_space_a, remaining_space_b, remaining_max_order, n, i) where {T,N}
    current_order_c = order(current_space_c)
    current_order_a = order(current_space_a)
    current_order_b = order(current_space_b)
    @inbounds Cᵢ = selectdim(C, N, i+current_order_c+1)
    @inbounds for j ∈ max(i-current_order_a, -current_order_b):min(i+current_order_a, current_order_b)
        _add_mul!(Cᵢ, selectdim(A, N, i-j+current_order_a+1), selectdim(B, N, j+current_order_b+1), α, remaining_space_c, remaining_space_a, remaining_space_b, remaining_max_order, n)
    end
    return C
end

# Chebyshev

_convolution_indices(::Chebyshev, order) = 0:order

function _convolution!(C, A, B, α, space_c::Chebyshev, space_a::Chebyshev, space_b::Chebyshev, i)
    order_a = order(space_a)
    order_b = order(space_b)
    v = zero(promote_type(eltype(A), eltype(B)))
    @inbounds for j ∈ max(i-order_a, -order_b):min(i+order_a, order_b)
        v += A[abs(i-j)+1] * B[abs(j)+1]
    end
    C[i+1] += v * α
    return C
end

function _convolution!(C::AbstractArray{T,N}, A, B, α, current_space_c::Chebyshev, current_space_a::Chebyshev, current_space_b::Chebyshev, remaining_space_c, remaining_space_a, remaining_space_b, remaining_max_order, n, i) where {T,N}
    current_order_a = order(current_space_a)
    current_order_b = order(current_space_b)
    @inbounds Cᵢ = selectdim(C, N, i+1)
    @inbounds for j ∈ max(i-current_order_a, -current_order_b):min(i+current_order_a, current_order_b)
        _add_mul!(Cᵢ, selectdim(A, N, abs(i-j)+1), selectdim(B, N, abs(j)+1), α, remaining_space_c, remaining_space_a, remaining_space_b, remaining_max_order, n)
    end
    return C
end

#

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
function ^̄(a::Sequence{<:SequenceSpace}, n::Int)
    n < 0 && return throw(DomainError(n, "^̄ is only defined for positive integers"))
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
        c = c *̄ a
    end
    return c
end
function _sqr(a::Sequence{<:SequenceSpace})
    new_space = image(^, space(a), 2)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    _add_sqr!(c, a, _max_order(new_space))
    return c
end
function _sqr_bar(a::Sequence{<:SequenceSpace})
    new_space = image(^̄, space(a), 2)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    _add_sqr!(c, a, _max_order(new_space))
    return c
end

function banach_rounding_pow(a::Sequence{<:SequenceSpace}, n::Int, X::ℓ¹)
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
function banach_rounding_pow_bar(a::Sequence{<:SequenceSpace}, n::Int, X::ℓ¹)
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
function _banach_rounding_sqr(a::Sequence{<:SequenceSpace}, X::ℓ¹)
    norm_a = norm(a, X)
    bound_a² = norm_a * norm_a
    rounding_order = banach_rounding_order(bound_a², X)
    new_space = image(^, space(a), 2)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    banach_rounding!(c, bound_a², X, rounding_order)
    _add_sqr!(c, a, _max_order(new_space, rounding_order))
    return c
end
function _banach_rounding_sqr_bar(a::Sequence{<:SequenceSpace}, X::ℓ¹)
    norm_a = norm(a, X)
    bound_a² = norm_a * norm_a
    rounding_order = banach_rounding_order(bound_a², X)
    new_space = image(^̄, space(a), 2)
    CoefType = eltype(a)
    c = Sequence(new_space, zeros(CoefType, dimension(new_space)))
    banach_rounding!(c, bound_a², X, rounding_order)
    _add_sqr!(c, a, _max_order(new_space, rounding_order))
    return c
end

function _add_sqr!(c::Sequence{<:BaseSpace}, a, max_order)
    C = coefficients(c)
    A = coefficients(a)
    _add_sqr!(C, A, space(c), space(a), max_order)
    return c
end

function _add_sqr!(c::Sequence{<:TensorSpace}, a, max_order)
    space_c = space(c)
    space_a = space(a)
    C = _no_alloc_reshape(coefficients(c), dimensions(space_c))
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    _add_mul!(C, A, A, true, space_c, space_a, space_a, max_order, 0//1)
    return c
end

# Taylor

function _add_sqr!(C, A, space_c::Taylor, space_a::Taylor, max_order)
    order_a = order(space_a)
    @inbounds A₀ = A[1]
    @inbounds C[1] = A₀ * A₀
    ord = min(order(space_c), max_order)
    @inbounds for i ∈ 1:ord
        i_odd = i%2
        i_end = (i-2+i_odd)÷2
        @inbounds for j ∈ max(i-order_a, 0):i_end
            C[i+1] += A[i-j+1] * A[j+1]
        end
        C[i+1] *= 2
        if iszero(i_odd)
            A_i½ = A[i÷2+1]
            C[i+1] += A_i½ * A_i½
        end
    end
    return C
end

# Fourier

function _add_sqr!(C, A, space_c::Fourier, space_a::Fourier, max_order)
    order_c = order(space_c)
    order_a = order(space_a)
    @inbounds for j ∈ 1:order_a
        C[order_c+1] += A[j+order_a+1] * A[-j+order_a+1]
    end
    @inbounds A₀ = A[order_a+1]
    @inbounds C[order_c+1] = 2C[order_c+1] + A₀ * A₀
    ord = min(order_c, max_order)
    @inbounds for i ∈ 1:ord
        i½, i_odd = divrem(i, 2)
        @inbounds for j ∈ i½+1:order_a
            C[i+order_c+1] += A[i-j+order_a+1] * A[j+order_a+1]
            C[-i+order_c+1] += A[j-i+order_a+1] * A[-j+order_a+1]
        end
        C[i+order_c+1] *= 2
        C[-i+order_c+1] *= 2
        if iszero(i_odd)
            A_i½ = A[i½+order_a+1]
            A_neg_i½ = A[-i½+order_a+1]
            C[i+order_c+1] += A_i½ * A_i½
            C[-i+order_c+1] += A_neg_i½ * A_neg_i½
        end
    end
    return C
end

# Chebyshev

function _add_sqr!(C, A, space_c::Chebyshev, space_a::Chebyshev, max_order)
    order_a = order(space_a)
    @inbounds for j ∈ 1:order_a
        Aⱼ = A[j+1]
        C[1] += Aⱼ * Aⱼ
    end
    @inbounds A₀ = A[1]
    @inbounds C[1] = 2C[1] + A₀ * A₀
    ord = min(order(space_c), max_order)
    @inbounds for i ∈ 1:ord
        i½, i_odd = divrem(i, 2)
        @inbounds for j ∈ i½+1:order_a
            C[i+1] += A[abs(i-j)+1] * A[j+1]
        end
        C[i+1] *= 2
        if iszero(i_odd)
            A_i½ = A[i½+1]
            C[i+1] += A_i½ * A_i½
        end
    end
    return C
end
