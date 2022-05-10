struct Evaluation{T<:Union{Nothing,Number,Tuple{Vararg{Union{Nothing,Number}}}}}
    value :: T
end

# fallback arithmetic methods

function Base.:+(A::LinearOperator, ℰ::Evaluation)
    domain_A = domain(A)
    return ladd!(A, project(ℰ, domain_A, codomain(A), _coeftype(ℰ, domain_A, eltype(A))))
end
function Base.:+(ℰ::Evaluation, A::LinearOperator)
    domain_A = domain(A)
    return radd!(project(ℰ, domain_A, codomain(A), _coeftype(ℰ, domain_A, eltype(A))), A)
end
function Base.:-(A::LinearOperator, ℰ::Evaluation)
    domain_A = domain(A)
    return lsub!(A, project(ℰ, domain_A, codomain(A), _coeftype(ℰ, domain_A, eltype(A))))
end
function Base.:-(ℰ::Evaluation, A::LinearOperator)
    domain_A = domain(A)
    return rsub!(project(ℰ, domain_A, codomain(A), _coeftype(ℰ, domain_A, eltype(A))), A)
end

#

(ℰ::Evaluation)(a::Sequence) = *(ℰ, a)
Base.:*(ℰ::Evaluation, a::Sequence) = evaluate(a, ℰ.value)

(a::Sequence)(x) = evaluate(a, x)
(a::Sequence)(x...) = evaluate(a, x)

function evaluate(a::Sequence, x)
    ℰ = Evaluation(x)
    space_a = space(a)
    new_space = image(ℰ, space_a)
    CoefType = _coeftype(ℰ, space_a, eltype(a))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _apply!(c, ℰ, a)
    return c
end

function evaluate!(c::Sequence, a::Sequence, x)
    ℰ = Evaluation(x)
    space(c) == image(ℰ, space(a)) || return throw(DomainError)
    _apply!(c, ℰ, a)
    return c
end

function project(ℰ::Evaluation, domain::VectorSpace, codomain::VectorSpace, ::Type{T}) where {T}
    _iscompatible(domain, codomain) || return throw(DimensionMismatch)
    C = LinearOperator(domain, codomain, zeros(T, dimension(codomain), dimension(domain)))
    _project!(C, ℰ, _memo(domain, codomain, T))
    return C
end

function project!(C::LinearOperator, ℰ::Evaluation)
    domain_C = domain(C)
    codomain_C = codomain(C)
    _iscompatible(domain_C, codomain_C) || return throw(DimensionMismatch)
    CoefType = eltype(C)
    coefficients(C) .= zero(CoefType)
    _project!(C, ℰ, _memo(domain_C, codomain_C, CoefType))
    return C
end

# Sequence spaces

_memo(s₁::TensorSpace, s₂::TensorSpace, ::Type{T}) where {T} =
    map((s₁ᵢ, s₂ᵢ) -> _memo(s₁ᵢ, s₂ᵢ, T), spaces(s₁), spaces(s₂))

image(ℰ::Evaluation{<:NTuple{N,Union{Nothing,Number}}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((xᵢ, sᵢ) -> image(Evaluation(xᵢ), sᵢ), ℰ.value, spaces(s)))

_coeftype(ℰ::Evaluation, s::TensorSpace, ::Type{T}) where {T} =
    @inbounds promote_type(_coeftype(Evaluation(ℰ.value[1]), s[1], T), _coeftype(Evaluation(Base.tail(ℰ.value)), Base.tail(s), T))
_coeftype(ℰ::Evaluation, s::TensorSpace{<:Tuple{BaseSpace}}, ::Type{T}) where {T} =
    @inbounds _coeftype(Evaluation(ℰ.value[1]), s[1], T)

function _apply!(c::Sequence{<:TensorSpace}, ℰ::Evaluation, a)
    space_a = space(a)
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    C = _no_alloc_reshape(coefficients(c), _effective_dimensions(ℰ, space(c)))
    _apply!(C, ℰ, space_a, A)
    return c
end

_effective_dimensions(ℰ::Evaluation{<:Tuple{Nothing,Vararg{Union{Nothing,Number}}}}, space::TensorSpace) =
    (dimension(space[1]), _effective_dimensions(Evaluation(Base.tail(ℰ.value)), Base.tail(space))...)
_effective_dimensions(::Evaluation{<:Tuple{Nothing}}, space::TensorSpace) =
    (dimension(space[1]),)
_effective_dimensions(ℰ::Evaluation, space::TensorSpace) =
    _effective_dimensions(Evaluation(Base.tail(ℰ.value)), Base.tail(space))
_effective_dimensions(::Evaluation{<:Tuple{Number}}, ::TensorSpace) = ()

_apply!(C, ℰ::Evaluation, space::TensorSpace, A) =
    @inbounds _apply!(C, Evaluation(ℰ.value[1]), space[1], _apply(Evaluation(Base.tail(ℰ.value)), Base.tail(space), A))

_apply!(C, ℰ::Evaluation, space::TensorSpace{<:Tuple{BaseSpace}}, A) =
    @inbounds _apply!(C, Evaluation(ℰ.value[1]), space[1], A)

_apply(ℰ::Evaluation, space::TensorSpace{<:NTuple{N₁,BaseSpace}}, A::AbstractArray{T,N₂}) where {N₁,T,N₂} =
    @inbounds _apply(Evaluation(ℰ.value[1]), space[1], Val(N₂-N₁+1), _apply(Evaluation(Base.tail(ℰ.value)), Base.tail(space), A))

_apply(ℰ::Evaluation, space::TensorSpace{<:Tuple{BaseSpace}}, A::AbstractArray{T,N}) where {T,N} =
    @inbounds _apply(Evaluation(ℰ.value[1]), space[1], Val(N), A)

function _project!(C::LinearOperator{<:SequenceSpace,<:SequenceSpace}, ℰ::Evaluation, memo)
    domain_C = domain(C)
    codomain_C = codomain(C)
    CoefType = eltype(C)
    @inbounds for β ∈ indices(domain_C), α ∈ indices(codomain_C)
        C[α,β] = _getindex(ℰ, domain_C, codomain_C, CoefType, α, β, memo)
    end
    return C
end

_getindex(ℰ::Evaluation{<:NTuple{N,Union{Nothing,Number}}}, domain::TensorSpace{<:NTuple{N,BaseSpace}}, codomain::TensorSpace{<:NTuple{N,BaseSpace}}, ::Type{T}, α, β, memo) where {N,T} =
    @inbounds _getindex(Evaluation(ℰ.value[1]), domain[1], codomain[1], T, α[1], β[1], memo[1]) * _getindex(Evaluation(Base.tail(ℰ.value)), Base.tail(domain), Base.tail(codomain), T, Base.tail(α), Base.tail(β), Base.tail(memo))
_getindex(ℰ::Evaluation{<:Tuple{Union{Nothing,Number}}}, domain::TensorSpace{<:Tuple{BaseSpace}}, codomain::TensorSpace{<:Tuple{BaseSpace}}, ::Type{T}, α, β, memo) where {T} =
    @inbounds _getindex(Evaluation(ℰ.value[1]), domain[1], codomain[1], T, α[1], β[1], memo[1])

# Taylor

_memo(::Taylor, ::Taylor, ::Type) = nothing

image(::Evaluation{Nothing}, s::Taylor) = s
image(::Evaluation, ::Taylor) = Taylor(0)

_coeftype(::Evaluation{Nothing}, ::Taylor, ::Type{T}) where {T} = T
_coeftype(::Evaluation{T}, ::Taylor, ::Type{S}) where {T,S} = promote_type(T, S)

function _apply!(c::Sequence{Taylor}, ::Evaluation{Nothing}, a)
    coefficients(c) .= coefficients(a)
    return c
end
function _apply!(c::Sequence{Taylor}, ℰ::Evaluation, a)
    x = ℰ.value
    if iszero(x)
        @inbounds c[0] = a[0]
    else
        ord = order(a)
        @inbounds c[0] = a[ord]
        @inbounds for i ∈ ord-1:-1:0
            c[0] = c[0] * x + a[i]
        end
    end
    return c
end

function _apply!(C::AbstractArray, ::Evaluation{Nothing}, ::Taylor, A)
    C .= A
    return C
end
function _apply!(C::AbstractArray{T}, ℰ::Evaluation, space::Taylor, A) where {T}
    x = ℰ.value
    if iszero(x)
        @inbounds C .= selectdim(A, 1, 1)
    else
        ord = order(space)
        @inbounds C .= selectdim(A, 1, ord+1)
        @inbounds for i ∈ ord-1:-1:0
            C .= C .* x .+ selectdim(A, 1, i+1)
        end
    end
    return C
end

_apply(::Evaluation{Nothing}, ::Taylor, ::Val, A::AbstractArray) = A
function _apply(ℰ::Evaluation, space::Taylor, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    x = ℰ.value
    CoefType = _coeftype(ℰ, space, T)
    if iszero(x)
        @inbounds C = convert(Array{CoefType,N-1}, selectdim(A, D, 1))
    else
        ord = order(space)
        @inbounds C = convert(Array{CoefType,N-1}, selectdim(A, D, ord+1))
        @inbounds for i ∈ ord-1:-1:0
            C .= C .* x .+ selectdim(A, D, i+1)
        end
    end
    return C
end

_getindex(::Evaluation{Nothing}, ::Taylor, ::Taylor, ::Type{T}, i, j, memo) where {T} =
    ifelse(i == j, one(T), zero(T))
function _getindex(ℰ::Evaluation, ::Taylor, ::Taylor, ::Type{T}, i, j, memo) where {T}
    if i == 0
        x = ℰ.value
        if j == 0
            return one(T)
        else
            if iszero(x)
                return zero(T)
            else
                return convert(T, convert(T, x)^j)
            end
        end
    else
        return zero(T)
    end
end

# Fourier

_memo(::Fourier, ::Fourier, ::Type) = nothing

image(::Evaluation{Nothing}, s::Fourier) = s
image(::Evaluation, s::Fourier) = Fourier(0, frequency(s))

_coeftype(::Evaluation{Nothing}, ::Fourier, ::Type{T}) where {T} = T
_coeftype(::Evaluation{T}, s::Fourier, ::Type{S}) where {T,S} =
    promote_type(typeof(cis(frequency(s)*zero(T))), S)

function _apply!(c::Sequence{<:Fourier}, ::Evaluation{Nothing}, a)
    coefficients(c) .= coefficients(a)
    return c
end
_apply!(c::Sequence{<:Fourier}, ℰ::Evaluation, a) = __apply!(c, ℰ, a)
function __apply!(c::Sequence{<:Fourier}, ℰ::Evaluation, a)
    x = ℰ.value
    ord = order(a)
    if iszero(x)
        @inbounds c[0] = a[0]
        @inbounds for j ∈ 1:ord
            c[0] += a[j] + a[-j]
        end
    else
        @inbounds c[0] = a[ord]
        if ord > 0
            @inbounds s₋ = convert(eltype(c), a[-ord])
            eiωx = cis(frequency(a)*x)
            eiωx_conj = conj(eiωx)
            @inbounds for j ∈ ord-1:-1:1
                c[0] = c[0] * eiωx + a[j]
                s₋ = s₋ * eiωx_conj + a[-j]
            end
            @inbounds c[0] = s₋ * eiωx_conj + a[0] + c[0] * eiωx
        end
    end
    return c
end
function __apply!(c::Sequence{<:Fourier,<:AbstractVector{<:Union{Interval,Complex{<:Interval}}}}, ℰ::Evaluation, a)
    x = ℰ.value
    ord = order(a)
    @inbounds c[0] = a[0]
    if iszero(x)
        @inbounds for j ∈ 1:ord
            c[0] += a[j] + a[-j]
        end
    elseif ord > 0
        ωx = frequency(a)*x
        @inbounds for j ∈ 1:ord
            eiωxj = cis(ωx*j)
            c[0] += a[j] * eiωxj + a[-j] * conj(eiωxj)
        end
    end
    return c
end

function _apply!(C::AbstractArray, ::Evaluation{Nothing}, ::Fourier, A)
    C .= A
    return C
end
_apply!(C::AbstractArray, ℰ::Evaluation, space::Fourier, A) = __apply!(C, ℰ, space, A)
function __apply!(C::AbstractArray, ℰ::Evaluation, space::Fourier, A)
    x = ℰ.value
    ord = order(space)
    @inbounds C .= selectdim(A, 1, ord+1)
    if iszero(x)
        @inbounds for j ∈ 1:ord
            C .+= selectdim(A, 1, ord+1+j) .+ selectdim(A, 1, ord+1-j)
        end
    elseif ord > 0
        eiωx = cis(frequency(space)*x)
        eiωxj = one(eiωx)
        @inbounds for j ∈ 1:ord
            eiωxj *= eiωx
            C .+= selectdim(A, 1, ord+1+j) .* eiωxj .+ selectdim(A, 1, ord+1-j) .* conj(eiωxj)
        end
    end
    return C
end
function __apply!(C::AbstractArray{<:Union{Interval,Complex{<:Interval}}}, ℰ::Evaluation, space::Fourier, A)
    x = ℰ.value
    ord = order(space)
    @inbounds C .= selectdim(A, 1, ord+1)
    if iszero(x)
        @inbounds for j ∈ 1:ord
            C .+= selectdim(A, 1, ord+1+j) .+ selectdim(A, 1, ord+1-j)
        end
    elseif ord > 0
        ωx = frequency(space)*x
        @inbounds for j ∈ 1:ord
            eiωxj = cis(ωx*j)
            C .+= selectdim(A, 1, ord+1+j) .* eiωxj .+ selectdim(A, 1, ord+1-j) .* conj(eiωxj)
        end
    end
    return C
end

_apply(::Evaluation{Nothing}, ::Fourier, ::Val, A::AbstractArray) = A
_apply(ℰ::Evaluation, space::Fourier, d::Val, A::AbstractArray) = __apply(ℰ, space, d, A, _coeftype(ℰ, space, eltype(A)))
function __apply(ℰ::Evaluation, space::Fourier, ::Val{D}, A::AbstractArray{T,N}, ::Type{CoefType}) where {D,T,N,CoefType}
    x = ℰ.value
    ord = order(space)
    @inbounds C = convert(Array{CoefType,N-1}, selectdim(A, D, ord+1))
    if iszero(x)
        @inbounds for j ∈ 1:ord
            C .+= selectdim(A, D, ord+1+j) .+ selectdim(A, D, ord+1-j)
        end
    elseif ord > 0
        eiωx = cis(frequency(space)*x)
        eiωxj = one(eiωx)
        @inbounds for j ∈ 1:ord
            eiωxj *= eiωx
            C .+= selectdim(A, D, ord+1+j) .* eiωxj .+ selectdim(A, D, ord+1-j) .* conj(eiωxj)
        end
    end
    return C
end
function __apply(ℰ::Evaluation, space::Fourier, ::Val{D}, A::AbstractArray{T,N}, ::Type{CoefType}) where {D,T,N,CoefType<:Union{Interval,Complex{<:Interval}}}
    x = ℰ.value
    ord = order(space)
    @inbounds C = convert(Array{CoefType,N-1}, selectdim(A, D, ord+1))
    if iszero(x)
        @inbounds for j ∈ 1:ord
            C .+= selectdim(A, D, ord+1+j) .+ selectdim(A, D, ord+1-j)
        end
    elseif ord > 0
        ωx = frequency(space)*x
        @inbounds for j ∈ 1:ord
            eiωxj = cis(ωx*j)
            C .+= selectdim(A, D, ord+1+j) .* eiωxj .+ selectdim(A, D, ord+1-j) .* conj(eiωxj)
        end
    end
    return C
end

_getindex(::Evaluation{Nothing}, domain::Fourier, codomain::Fourier, ::Type{T}, i, j, memo) where {T} =
    ifelse(i == j, one(T), zero(T))
function _getindex(ℰ::Evaluation, domain::Fourier, codomain::Fourier, ::Type{T}, i, j, memo) where {T}
    if i == 0
        x = ℰ.value
        if j == 0 || iszero(x)
            return one(T)
        else
            return convert(T, cis(frequency(domain)*j*x))
        end
    else
        return zero(T)
    end
end

# Chebyshev

_memo(::Chebyshev, ::Chebyshev, ::Type{T}) where {T} = Dict{Int,T}()

image(::Evaluation{Nothing}, s::Chebyshev) = s
image(::Evaluation, ::Chebyshev) = Chebyshev(0)

_coeftype(::Evaluation{Nothing}, ::Chebyshev, ::Type{T}) where {T} = T
_coeftype(::Evaluation{T}, ::Chebyshev, ::Type{S}) where {T,S} = promote_type(T, S)

function _apply!(c::Sequence{Chebyshev}, ::Evaluation{Nothing}, a)
    coefficients(c) .= coefficients(a)
    return c
end
function _apply!(c::Sequence{Chebyshev}, ℰ::Evaluation, a)
    x = ℰ.value
    ord = order(a)
    if iszero(x)
        c[0] = a[0]
        @inbounds for i ∈ 2:2:ord
            c[0] += ifelse(i%4 == 0, 2, -2) * a[i]
        end
    elseif isone(-x)
        c[0] = a[0]
        @inbounds for i ∈ 1:ord
            c[0] += ifelse(isodd(i), -2, 2) * a[i]
        end
    elseif isone(x)
        c[0] = a[0]
        @inbounds for i ∈ 1:ord
            c[0] += 2a[i]
        end
    else
        if ord == 0
            @inbounds c[0] = a[0]
        elseif ord == 1
            @inbounds c[0] = a[0] + 2x * a[1]
        else
            CoefType = eltype(c)
            x2 = 2x
            s = zero(CoefType)
            @inbounds t = convert(CoefType, 2a[ord])
            @inbounds c[0] = zero(CoefType)
            @inbounds for i ∈ ord-1:-1:1
                c[0] = t
                t = x2 * t - s + 2a[i]
                s = c[0]
            end
            @inbounds c[0] = x * t - s + a[0]
        end
    end
    return c
end

function _apply!(C::AbstractArray, ::Evaluation{Nothing}, ::Chebyshev, A)
    C .= A
    return C
end
function _apply!(C::AbstractArray{T}, ℰ::Evaluation, space::Chebyshev, A) where {T}
    x = ℰ.value
    ord = order(space)
    if iszero(x)
        @inbounds C .= selectdim(A, 1, 1)
        @inbounds for i ∈ 2:2:ord
            C .+= ifelse(i%4 == 0, 2, -2) .* selectdim(A, 1, i+1)
        end
    elseif isone(-x)
        @inbounds C .= selectdim(A, 1, 1)
        @inbounds for i ∈ 1:ord
            C .+= ifelse(isodd(i), -2, 2) .* selectdim(A, 1, i+1)
        end
    elseif isone(x)
        @inbounds C .= selectdim(A, 1, 1)
        @inbounds for i ∈ 1:ord
            C .+= 2 .* selectdim(A, 1, i+1)
        end
    else
        if ord == 0
            @inbounds C .= selectdim(A, 1, 1)
        elseif ord == 1
            @inbounds C .= selectdim(A, 1, 1) .+ (2x) .* selectdim(A, 1, 2)
        else
            x2 = 2x
            @inbounds Aᵢ = selectdim(A, 1, ord+1)
            sz = size(Aᵢ)
            s = zeros(T, sz)
            t = Array{T}(undef, sz)
            t .= 2 .* Aᵢ
            @inbounds for i ∈ ord-1:-1:1
                C .= t
                t .= x2 .* t .- s .+ 2 .* selectdim(A, 1, i+1)
                s .= C
            end
            @inbounds C .= x .* t .- s .+ selectdim(A, 1, 1)
        end
    end
    return C
end

_apply(::Evaluation{Nothing}, ::Chebyshev, ::Val, A::AbstractArray) = A
function _apply(ℰ::Evaluation, space::Chebyshev, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    x = ℰ.value
    CoefType = _coeftype(ℰ, space, T)
    ord = order(space)
    if iszero(x)
        C = convert(Array{CoefType,N-1}, selectdim(A, D, 1))
        @inbounds for i ∈ 2:2:ord
            C .+= ifelse(i%4 == 0, 2, -2) .* selectdim(A, D, i+1)
        end
        return C
    elseif isone(-x)
        C = convert(Array{CoefType,N-1}, selectdim(A, D, 1))
        @inbounds for i ∈ 1:ord
            C .+= ifelse(isodd(i), -2, 2) .* selectdim(A, D, i+1)
        end
        return C
    elseif isone(x)
        C = convert(Array{CoefType,N-1}, selectdim(A, D, 1))
        @inbounds for i ∈ 1:ord
            C .+= 2 .* selectdim(A, D, i+1)
        end
        return C
    else
        if ord == 0
            return @inbounds convert(Array{CoefType,N-1}, selectdim(A, D, 1))
        elseif ord == 1
            return @inbounds convert(Array{CoefType,N-1}, selectdim(A, D, 1) .+ (2x) .* selectdim(A, D, 2))
        else
            x2 = 2x
            @inbounds Aᵢ = selectdim(A, D, ord+1)
            sz = size(Aᵢ)
            s = zeros(CoefType, sz)
            t = Array{CoefType,N-1}(undef, sz)
            t .= 2 .* Aᵢ
            C = Array{CoefType,N-1}(undef, sz)
            @inbounds for i ∈ ord-1:-1:1
                C .= t
                t .= x2 .* t .- s .+ 2 .* selectdim(A, D, i+1)
                s .= C
            end
            @inbounds C .= x .* t .- s .+ selectdim(A, D, 1)
            return C
        end
    end
end

_getindex(::Evaluation{Nothing}, ::Chebyshev, ::Chebyshev, ::Type{T}, i, j, memo) where {T} =
    ifelse(i == j, one(T), zero(T))
function _getindex(ℰ::Evaluation, domain::Chebyshev, codomain::Chebyshev, ::Type{T}, i, j, memo) where {T}
    if i == 0
        if j == 0
            return one(T)
        else
            x = ℰ.value
            if iszero(x)
                if isodd(j)
                    return zero(T)
                elseif j%4 == 0
                    return convert(T, 2)
                else
                    return convert(T, -2)
                end
            elseif isone(-x)
                if isodd(j)
                    return convert(T, -2)
                else
                    return convert(T, 2)
                end
            elseif isone(x)
                return convert(T, 2)
            else
                x2 = convert(T, 2x)
                if j == 1
                    return x2
                elseif j == 2
                    return convert(T, x2*x2 - 2)
                else
                    return get!(memo, j) do
                        x2*_getindex(ℰ, domain, codomain, T, i, j-1, memo) - _getindex(ℰ, domain, codomain, T, i, j-2, memo)
                    end
                end
            end
        end
    else
        return zero(T)
    end
end

# Cartesian spaces

_memo(s₁::CartesianSpace, s₂::CartesianSpace, ::Type{T}) where {T} =
    @inbounds _memo(s₁[1], s₂[1], T)

image(ℰ::Evaluation, s::CartesianPower) =
    CartesianPower(image(ℰ, space(s)), nb_cartesian_product(s))

image(ℰ::Evaluation, s::CartesianProduct) =
    CartesianProduct(map(sᵢ -> image(ℰ, sᵢ), spaces(s)))

_coeftype(ℰ::Evaluation, s::CartesianPower, ::Type{T}) where {T} =
    _coeftype(ℰ, space(s), T)

_coeftype(ℰ::Evaluation, s::CartesianProduct, ::Type{T}) where {T} =
    @inbounds promote_type(_coeftype(ℰ, s[1], T), _coeftype(ℰ, Base.tail(s), T))
_coeftype(ℰ::Evaluation, s::CartesianProduct{<:Tuple{VectorSpace}}, ::Type{T}) where {T} =
    @inbounds _coeftype(ℰ, s[1], T)

function _apply!(c::Sequence{<:CartesianPower}, ℰ::Evaluation, a)
    @inbounds for i ∈ 1:nb_cartesian_product(space(c))
        _apply!(component(c, i), ℰ, component(a, i))
    end
    return c
end
function _apply!(c::Sequence{CartesianProduct{T}}, ℰ::Evaluation, a) where {N,T<:NTuple{N,VectorSpace}}
    @inbounds _apply!(component(c, 1), ℰ, component(a, 1))
    @inbounds _apply!(component(c, 2:N), ℰ, component(a, 2:N))
    return c
end
function _apply!(c::Sequence{CartesianProduct{T}}, ℰ::Evaluation, a) where {T<:Tuple{VectorSpace}}
    @inbounds _apply!(component(c, 1), ℰ, component(a, 1))
    return c
end

function _project!(C::LinearOperator{<:CartesianSpace,<:CartesianSpace}, ℰ::Evaluation, memo)
    @inbounds for i ∈ 1:nb_cartesian_product(domain(C))
        _project!(component(C, i, i), ℰ, memo)
    end
    return C
end
