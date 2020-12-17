## arithmetic operations +, - between sequences

Base.:+(a::Sequence) = Sequence(a.space, +(a.coefficients))
Base.:-(a::Sequence) = Sequence(a.space, -(a.coefficients))

function Base.:+(a::Sequence, b::Sequence)
    space = a.space ∪ b.space
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(space, Vector{CoefType}(undef, length(space)))
    if a.space == b.space
        @. c.coefficients = a.coefficients + b.coefficients
        return c
    elseif a.space ⊆ b.space
        @. c.coefficients = b.coefficients
        @inbounds for α ∈ eachindex(a.space)
            c[α] += a[α]
        end
        return c
    elseif b.space ⊆ a.space
        @. c.coefficients = a.coefficients
        @inbounds for α ∈ eachindex(b.space)
            c[α] += b[α]
        end
        return c
    else
        c.coefficients .= zero(CoefType)
        @inbounds for α ∈ eachindex(a.space)
            c[α] = a[α]
        end
        @inbounds for α ∈ eachindex(b.space)
            c[α] += b[α]
        end
        return c
    end
end

function Base.:-(a::Sequence, b::Sequence)
    space = a.space ∪ b.space
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(space, Vector{CoefType}(undef, length(space)))
    if a.space == b.space
        @. c.coefficients = a.coefficients - b.coefficients
        return c
    elseif a.space ⊆ b.space
        @. c.coefficients = -b.coefficients
        @inbounds for α ∈ eachindex(a.space)
            c[α] += a[α]
        end
        return c
    elseif b.space ⊆ a.space
        @. c.coefficients = a.coefficients
        @inbounds for α ∈ eachindex(b.space)
            c[α] -= b[α]
        end
        return c
    else
        c.coefficients .= zero(CoefType)
        @inbounds for α ∈ eachindex(a.space)
            c[α] = a[α]
        end
        @inbounds for α ∈ eachindex(b.space)
            c[α] -= b[α]
        end
        return c
    end
end

## tools for convolution

function _geometric_decay(A::AbstractVector{T}) where {T}
    abs_A = abs.(A)
    if real(T) <: Interval
        filter!(!iszero, abs_A)
        replace!(Aᵢ -> Aᵢ.lo == 0 ? @interval(Aᵢ.hi, Aᵢ.hi) : Aᵢ, abs_A)
    else
        filter!(Aᵢ -> Aᵢ > 0, abs_A)
    end
    isempty(abs_A) && return exp(zero(real(T)))
    log_abs_A = log.(abs_A)
    n = length(log_abs_A)
    n == 1 && return exp(zero(real(T)))
    log_abs_Ā = sum(log_abs_A)/n
    ī = (n+1)/2
    β = sum(t -> (t[1]-ī)*(t[2]-log_abs_Ā), enumerate(log_abs_A))/sum(i -> (i - ī)^2, eachindex(log_abs_A))
    return exp(-β)
end

function _geometric_decay(a::Sequence{<:UnivariateSpace})
    decay = _geometric_decay(a.coefficients)
    return max(one(decay), decay)
end

function _geometric_decay(a::Sequence{<:Fourier})
    @inbounds a₋ = view(a, -order(a):0)
    @inbounds a₊ = view(a, 0:order(a))
    decay = min(inv(_geometric_decay(a₋)), _geometric_decay(a₊))
    return max(one(decay), decay)
end

_geometric_decay(a::Sequence{TensorSpace{T}}) where {N,T<:NTuple{N,UnivariateSpace}} =
    min(_geometric_decay(selectdim(a, N, 0)), _geometric_decay(selectdim(a, N-1, 0)))

function _banach_rounding_order(decay::Real, bound::Real)
    isone(decay) && return typemax(Int)
    isinf(bound) && return typemax(Int)
    ϵ = eps(decay)
    bound ≤ ϵ && return 0
    return ceil(Int, log(bound/ϵ)/log(decay))
end

_banach_rounding_order(decay::Interval, bound::Interval) =
    _banach_rounding_order(sup(decay), sup(bound))

function _banach_rounding!(a::Sequence{Taylor}, decay::Real, bound::Real, max_order::Int)
    typemax(Int) == max_order && return a
    ν⁻¹ = inv(decay)
    μⁱ = bound * ν⁻¹ ^ max_order
    @inbounds for i ∈ max_order+1:a.space.order
        μⁱ *= ν⁻¹
        a[i] = @interval(-μⁱ, μⁱ)
    end
    return a
end

function _banach_rounding!(a::Sequence{<:Fourier}, decay::Real, bound::Real, max_order::Int)
    typemax(Int) == max_order && return a
    ν⁻¹ = inv(decay)
    μⁱ = bound * ν⁻¹ ^ max_order
    @inbounds for i ∈ max_order+1:a.space.order
        μⁱ *= ν⁻¹
        interval_μⁱ = @interval(-μⁱ, μⁱ)
        a[i] = interval_μⁱ
        a[-i] = interval_μⁱ
    end
    return a
end

function _banach_rounding!(a::Sequence{Chebyshev}, decay::Real, bound::Real, max_order::Int)
    typemax(Int) == max_order && return a
    ν⁻¹ = inv(decay)
    μⁱ = bound * ν⁻¹ ^ max_order
    @inbounds for i ∈ max_order+1:a.space.order
        μⁱ *= ν⁻¹
        a[i] = @interval(-μⁱ, μⁱ)
    end
    return a
end

function _banach_rounding!(a::Sequence{<:TensorSpace}, decay::Real, bound::Real, max_order::Int)
    typemax(Int) == max_order && return a
    ν⁻¹ = inv(decay)
    @inbounds for α ∈ eachindex(a.space)
        sα = sum(abs, α)
        if sα > max_order
            μᵅ = bound * ν⁻¹ ^ sα
            a[α] = @interval(-μᵅ, μᵅ)
        end
    end
    return a
end

# could be used to resize the multiplication_range space and remove zeros
_effective_multiplication_range(s::Taylor, order::Int) = Taylor(min(s.order, order))
_effective_multiplication_range(s::Fourier, order::Int) = Fourier(min(s.order, order), s.frequency)
_effective_multiplication_range(s::Chebyshev, order::Int) = Chebyshev(min(s.order, order))
_effective_multiplication_range(s::TensorSpace, order::Int) = TensorSpace(map(sᵢ -> _effective_multiplication_range(sᵢ, order), s.spaces))

## arithmetic operation * between sequences

function Base.:*(a::Sequence, b::Sequence)
    decay = min(_geometric_decay(a), _geometric_decay(b))
    bound_ab = norm(a, decay) * norm(b, decay)
    banach_rounding_order = _banach_rounding_order(decay, bound_ab)
    return _mul(a, b, decay, bound_ab, banach_rounding_order)
end

function Base.:*(a::Sequence, b::Sequence, c::Sequence...)
    decay = mapreduce(_geometric_decay, min, c; init = min(_geometric_decay(a), _geometric_decay(b)))
    bound_ab = norm(a, decay) * norm(b, decay)
    ab = _mul(a, b, decay, bound_ab, _banach_rounding_order(decay, bound_ab))
    return _mul(ab, c, decay, bound_ab)
end

function _mul(a::Sequence, b::Tuple, decay::Real, bound_a::Real)
    @inbounds b₁ = b[1]
    bound_ab₁ = bound_a * norm(b₁, decay)
    ab₁ = _mul(a, b₁, decay, bound_ab₁, _banach_rounding_order(decay, bound_ab₁))
    return _mul(ab₁, Base.tail(b), decay, bound_ab₁)
end

function _mul(a::Sequence, b::Tuple{Sequence}, decay::Real, bound_a::Real)
    @inbounds b₁ = b[1]
    bound_ab₁ = bound_a * norm(b₁, decay)
    return _mul(a, b₁, decay, bound_ab₁, _banach_rounding_order(decay, bound_ab₁))
end

function _mul(a::Sequence, b::Sequence, decay::Real, bound::Real, banach_rounding_order::Int)
    CoefType = promote_type(eltype(a), eltype(b))
    space = multiplication_range(a.space, b.space)
    c = Sequence(space, zeros(CoefType, length(space)))
    _add_mul!(c, a, b, banach_rounding_order)
    real(CoefType) <: Interval && return _banach_rounding!(c, decay, bound, banach_rounding_order)
    return c
end

@inline function _add_mul!(c::Sequence{Taylor}, a::Sequence{Taylor}, b::Sequence{Taylor}, max_order::Int)
    order_a = order(a)
    order_b = order(b)
    for i ∈ 0:min(order(c), max_order)
        @inbounds for j ∈ max(i-order_a, 0):min(i, order_b)
            c[i] += a[i-j] * b[j]
        end
    end
    return c
end

@inline function _add_mul!(c::Sequence{<:Fourier}, a::Sequence{<:Fourier}, b::Sequence{<:Fourier}, max_order::Int)
    order_a = order(a)
    order_b = order(b)
    ord = min(order(c), max_order)
    for i ∈ -ord:ord
        @inbounds for j ∈ max(i-order_a, -order_b):min(i+order_a, order_b)
            c[i] += a[i-j] * b[j]
        end
    end
    return c
end

@inline function _add_mul!(c::Sequence{Chebyshev}, a::Sequence{Chebyshev}, b::Sequence{Chebyshev}, max_order::Int)
    order_a = order(a)
    order_b = order(b)
    for i ∈ 0:min(order(c), max_order)
        @inbounds for j ∈ max(i-order_a, -order_b):min(i+order_a, order_b)
            c[i] += a[abs(i-j)] * b[abs(j)]
        end
    end
    return c
end

_add_mul!(c::Sequence{TensorSpace{T}}, a, b, max_order::Int) where {N,T<:NTuple{N,UnivariateSpace}} =
    @inbounds _add_mul!(c, c.space[N], a, b, max_order)

_add_mul!(c::Sequence{<:UnivariateSpace}, current_space::UnivariateSpace, a, b, max_order::Int) =
    _add_mul!(c, a, b, max_order)

@inline function _add_mul!(c::Sequence{TensorSpace{T}}, current_space::Taylor, a, b, max_order::Int) where {N,T<:NTuple{N,UnivariateSpace}}
    @inbounds next_space = c.space[N-1]
    @inbounds order_a = order(a.space[N])
    @inbounds order_b = order(b.space[N])
    @inbounds for i ∈ 0:min(order(current_space), max_order)
        max_order_i = max_order-i
        cᵢ = selectdim(c, N, i)
        @inbounds for j ∈ max(i-order_a, 0):min(i, order_b)
            _add_mul!(cᵢ, next_space, selectdim(a, N, i-j), selectdim(b, N, j), max_order_i)
        end
    end
    return c
end

@inline function _add_mul!(c::Sequence{TensorSpace{T}}, current_space::Fourier, a, b, max_order::Int) where {N,T<:NTuple{N,UnivariateSpace}}
    @inbounds next_space = c.space[N-1]
    @inbounds order_a = order(a.space[N])
    @inbounds order_b = order(b.space[N])
    ord = min(order(current_space), max_order)
    @inbounds for i ∈ -ord:ord
        max_order_i = max_order-abs(i)
        cᵢ = selectdim(c, N, i)
        @inbounds for j ∈ max(i-order_a, -order_b):min(i+order_a, order_b)
            _add_mul!(cᵢ, next_space, selectdim(a, N, i-j), selectdim(b, N, j), max_order_i)
        end
    end
    return c
end

@inline function _add_mul!(c::Sequence{TensorSpace{T}}, current_space::Chebyshev, a, b, max_order::Int) where {N,T<:NTuple{N,UnivariateSpace}}
    @inbounds next_space = c.space[N-1]
    @inbounds order_a = order(a.space[N])
    @inbounds order_b = order(b.space[N])
    @inbounds for i ∈ 0:min(order(current_space), max_order)
        max_order_i = max_order-i
        cᵢ = selectdim(c, N, i)
        @inbounds for j ∈ max(i-order_a, -order_b):min(i+order_a, order_b)
            _add_mul!(cᵢ, next_space, selectdim(a, N, abs(i-j)), selectdim(b, N, abs(j)), max_order_i)
        end
    end
    return c
end

## arithmetic operation ^

function Base.:^(a::Sequence, n::Int)
    n < 0 && return throw(DomainError(n, "^ is only defined for positive integers."))
    n == 0 && return one(a)
    n == 1 && return copy(a)
    decay = _geometric_decay(a)
    norm_a = norm(a, decay)
    if n == 2
        bound_a² = norm_a*norm_a
        return _sqr(a, decay, bound_a², _banach_rounding_order(decay, bound_a²))
    else
        # power by squaring
        bound_a = norm_a
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) > 0
            bound_a = bound_a^2
            a = _sqr(a, decay, bound_a, _banach_rounding_order(decay, bound_a))
        end
        bound_c = bound_a
        c = a
        while n > 0
            t = trailing_zeros(n) + 1
            n >>= t
            while (t -= 1) ≥ 0
                bound_a = bound_a^2
                a = _sqr(a, decay, bound_a, _banach_rounding_order(decay, bound_a))
            end
            bound_c *= bound_a
            c = _mul(c, a, decay, bound_c, _banach_rounding_order(decay, bound_c))
        end
        return c
    end
end

function _sqr(a::Sequence, decay::Real, bound::Real, banach_rounding_order::Int)
    CoefType = eltype(a)
    space = multiplication_range(a.space, a.space)
    c = Sequence(space, zeros(CoefType, length(space)))
    _add_sqr!(c, a, banach_rounding_order)
    real(CoefType) <: Interval && return _banach_rounding!(c, decay, bound, banach_rounding_order)
    return c
end

@inline function _add_sqr!(c::Sequence{Taylor}, a::Sequence{Taylor}, max_order::Int)
    order_a = order(a)
    @inbounds c[0] += a[0]^2
    @inbounds for i ∈ 1:min(order(c), max_order)
        i_odd = i%2
        i_end = (i-2+i_odd)÷2
        @inbounds for j ∈ max(i-order_a, 0):i_end
            c[i] += a[i-j] * a[j]
        end
        c[i] *= 2
        i_odd == 1 && continue
        c[i] += a[i÷2]^2
    end
    return c
end

@inline function _add_sqr!(c::Sequence{<:Fourier}, a::Sequence{<:Fourier}, max_order::Int)
    order_a = order(a)
    @inbounds for j ∈ 1:order_a
        c[0] += a[j] * a[-j]
    end
    @inbounds c[0] = 2c[0] + a[0]^2
    @inbounds for i ∈ 1:min(order(c), max_order)
        i½, i_odd = divrem(i, 2)
        @inbounds for j ∈ i½+1:order_a
            c[i] += a[i-j] * a[j]
            c[-i] += a[j-i] * a[-j]
        end
        c[i] *= 2
        c[-i] *= 2
        i_odd == 1 && continue
        c[i] += a[i½]^2
        c[-i] += a[-i½]^2
    end
    return c
end

@inline function _add_sqr!(c::Sequence{Chebyshev}, a::Sequence{Chebyshev}, max_order::Int)
    order_a = order(a)
    @inbounds for j ∈ 1:order_a
        c[0] += a[j]^2
    end
    @inbounds c[0] = 2c[0] + a[0]^2
    @inbounds for i ∈ 1:min(order(c), max_order)
        i½, i_odd = divrem(i, 2)
        @inbounds for j ∈ i½+1:order_a
            c[i] += a[abs(i-j)] * a[j]
        end
        c[i] *= 2
        i_odd == 1 && continue
        c[i] += a[i½]^2
    end
    return c
end

_add_sqr!(c::Sequence{TensorSpace{T}}, a, max_order::Int) where {N,T<:NTuple{N,UnivariateSpace}} =
    @inbounds _add_mul!(c, c.space[N], a, a, max_order) # _add_sqr!(c, c.space[N], a, max_order)

## arithmetic operations +, -, *, /, \ with field elements

function Base.:+(a::Sequence, b)
    CoefType = promote_type(eltype(a), typeof(b))
    c = Sequence(a.space, Vector{CoefType}(undef, length(a.space)))
    @. c.coefficients = a.coefficients
    @inbounds c[_constant_index(a.space)] += b
    return c
end

Base.:+(b, a::Sequence) = +(a, b)

function Base.:-(a::Sequence, b)
    CoefType = promote_type(eltype(a), typeof(b))
    c = Sequence(a.space, Vector{CoefType}(undef, length(a.space)))
    @. c.coefficients = a.coefficients
    @inbounds c[_constant_index(a.space)] -= b
    return c
end

function Base.:-(b, a::Sequence)
    CoefType = promote_type(eltype(a), typeof(b))
    c = Sequence(a.space, Vector{CoefType}(undef, length(a.space)))
    @. c.coefficients = -a.coefficients
    @inbounds c[_constant_index(a.space)] += b
    return c
end

Base.:*(a::Sequence, b) = Sequence(a.space, *(a.coefficients, b))
Base.:*(b, a::Sequence) = Sequence(a.space, *(b, a.coefficients))

Base.:/(a::Sequence, b) = Sequence(a.space, /(a.coefficients, b))
Base.:\(b, a::Sequence) = Sequence(a.space, \(b, a.coefficients))

## arithmetic operations +̄, -̄ between sequences

function +̄(a::Sequence, b::Sequence)
    space = a.space ∪̄ b.space
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(space, Vector{CoefType}(undef, length(space)))
    if a.space == b.space
        @. c.coefficients = a.coefficients + b.coefficients
        return c
    elseif a.space ⊆ b.space
        @. c.coefficients = a.coefficients
        @inbounds for α ∈ eachindex(a.space)
            c[α] += b[α]
        end
        return c
    elseif b.space ⊆ a.space
        @. c.coefficients = b.coefficients
        @inbounds for α ∈ eachindex(b.space)
            c[α] += a[α]
        end
        return c
    else
        c.coefficients .= zero(CoefType)
        @inbounds for α ∈ eachindex(a.space ∪̄ space)
            c[α] = a[α]
        end
        @inbounds for α ∈ eachindex(b.space ∪̄ space)
            c[α] += b[α]
        end
        return c
    end
end

function -̄(a::Sequence{T₁,S₁}, b::Sequence{T₂,S₂}) where {T₁<:SequenceSpace,S₁,T₂<:SequenceSpace,S₂}
    space = a.space ∪̄ b.space
    CoefType = promote_type(eltype(a), eltype(b))
    c = Sequence(space, Vector{CoefType}(undef, length(space)))
    if a.space == b.space
        @. c.coefficients = a.coefficients - b.coefficients
        return c
    elseif a.space ⊆ b.space
        @. c.coefficients = a.coefficients
        @inbounds for α ∈ eachindex(a.space)
            c[α] -= b[α]
        end
        return c
    elseif b.space ⊆ a.space
        @. c.coefficients = -b.coefficients
        @inbounds for α ∈ eachindex(b.space)
            c[α] += a[α]
        end
        return c
    else
        c.coefficients .= zero(CoefType)
        @inbounds for α ∈ eachindex(a.space ∪̄ space)
            c[α] = a[α]
        end
        @inbounds for α ∈ eachindex(b.space ∪̄ space)
            c[α] += b[α]
        end
        return c
    end
end

## arithmetic operation *̄ between sequences

*̄(a::Sequence, b::Sequence) = project(a * b, a.space ∪̄ b.space)
*̄(a::Sequence, b::Sequence, c::Sequence...) = project(*(a, b, c...), mapreduce(cᵢ -> cᵢ.space, ∪̄, c; init = a.space ∪̄ b.space))

## arithmetic operation ^̄

^̄(a::Sequence, n::Integer) = project(a ^ n, a.space)
