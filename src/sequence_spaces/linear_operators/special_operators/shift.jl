"""
    Shift{T<:Union{Number,Tuple{Vararg{Number}}}} <: AbstractLinearOperator

Generic shift operator.

Field:
- `value :: T`

Constructors:
- `Shift(::Number)`
- `Shift(::Tuple{Vararg{Number}})`
- `Shift(value::Number...)`: equivalent to `Shift(value)`

# Examples

```jldoctest
julia> Shift(1.0)
Shift{Float64}(1.0)

julia> Shift(1.0, 2.0)
Shift{Tuple{Float64, Float64}}((1.0, 2.0))
```
"""
struct Shift{T<:Union{Number,Tuple{Vararg{Number}}}} <: AbstractLinearOperator
    value :: T
    Shift{T}(value::T) where {T<:Union{Number,Tuple{Vararg{Number}}}} = new{T}(value)
    Shift{Tuple{}}(::Tuple{}) = throw(ArgumentError("Shift is only defined for at least one Number"))
end

Shift(value::T) where {T<:Number} = Shift{T}(value)
Shift(value::T) where {T<:Tuple{Vararg{Number}}} = Shift{T}(value)
Shift(value::Number...) = Shift(value)

value(𝒮::Shift) = 𝒮.value

Base.:*(𝒮₁::Shift{<:Number}, 𝒮₂::Shift{<:Number}) = Shift(value(𝒮₁) + value(𝒮₂))
Base.:*(𝒮₁::Shift{<:NTuple{N,Number}}, 𝒮₂::Shift{<:NTuple{N,Number}}) where {N} = Shift(map(+, value(𝒮₁), value(𝒮₂)))

Base.:^(𝒮::Shift{<:Number}, n::Integer) = Shift(value(𝒮) * exact(n))
Base.:^(𝒮::Shift{<:Tuple{Vararg{Number}}}, n::Integer) = Shift(map(τᵢ -> τᵢ * exact(n), value(𝒮)))
Base.:^(𝒮::Shift{<:NTuple{N,Number}}, n::NTuple{N,Integer}) where {N} = Shift(map((τᵢ, nᵢ) -> τᵢ * exact(nᵢ), value(𝒮), n))

# Tensor space

domain(S::Shift{<:NTuple{N,Number}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((τᵢ, sᵢ) -> domain(Shift(τᵢ), sᵢ), value(S), spaces(s)))

codomain(𝒮::Shift{<:NTuple{N,Number}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((τᵢ, sᵢ) -> codomain(Shift(τᵢ), sᵢ), value(𝒮), spaces(s)))

_coeftype(𝒮::Shift{<:NTuple{N,Number}}, s::TensorSpace{<:NTuple{N,BaseSpace}}, ::Type{T}) where {N,T} =
    @inbounds promote_type(_coeftype(Shift(value(𝒮)[1]), s[1], T), _coeftype(Shift(Base.tail(value(𝒮))), Base.tail(s), T))
_coeftype(𝒮::Shift{<:Tuple{Number}}, s::TensorSpace{<:Tuple{BaseSpace}}, ::Type{T}) where {T} =
    @inbounds _coeftype(Shift(value(𝒮)[1]), s[1], T)

getcoefficient(𝒮::Shift{<:NTuple{N,Number}}, (codom, i)::Tuple{TensorSpace{<:NTuple{N,BaseSpace}},NTuple{N,Integer}}, (dom, j)::Tuple{TensorSpace{<:NTuple{N,BaseSpace}},NTuple{N,Integer}}, ::Type{T}) where {N,T} =
    @inbounds getcoefficient(Shift(value(𝒮)[1]), (codom[1], i[1]), (dom[1], j[1]), T) * getcoefficient(Shift(Base.tail(value(𝒮))), (Base.tail(codom), T, Base.tail(i)), (Base.tail(dom), Base.tail(j)), T)
getcoefficient(𝒮::Shift{<:Tuple{Number}}, (codom, i)::Tuple{TensorSpace{<:Tuple{BaseSpace}},Tuple{Integer}}, (dom, j)::Tuple{TensorSpace{<:Tuple{BaseSpace}},Tuple{Integer}}, ::Type{T}) where {T} =
    @inbounds getcoefficient(Shift(value(𝒮)[1]), (codom[1], i[1]), (dom[1], j[1]), T)

# Taylor

domain(::Shift, s::Taylor) = s

codomain(::Shift, s::Taylor) = s

_coeftype(::Shift{T}, ::Taylor, ::Type{S}) where {T,S} = promote_type(T, S)

function getcoefficient(𝒮::Shift, (codom, i)::Tuple{Taylor,Integer}, (dom, j)::Tuple{Taylor,Integer}, ::Type{T}) where {T}
    τ = value(𝒮)
    if iszero(τ)
        return ifelse(i == j, one(T), zero(T))
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

# Fourier

domain(::Shift, s::Fourier) = s

codomain(::Shift, s::Fourier) = s

_coeftype(::Shift{T}, s::Fourier, ::Type{S}) where {T,S} =
    promote_type(typeof(cis(frequency(s)*zero(T))), S)

function getcoefficient(𝒮::Shift, (codom, i)::Tuple{Fourier,Integer}, (dom, j)::Tuple{Fourier,Integer}, ::Type{T}) where {T}
    i == j || return zero(T)
    τ = value(𝒮)
    iszero(τ) && return one(T)
    return convert(T, cis(frequency(dom) * τ * exact(i)))
end

# Chebyshev

domain(::Shift, s::Chebyshev) = s

codomain(::Shift, s::Chebyshev) = s

_coeftype(::Shift{T}, ::Chebyshev, ::Type{S}) where {T,S} = promote_type(T, S)

function getcoefficient(𝒮::Shift, (codom, i)::Tuple{Chebyshev,Integer}, (dom, j)::Tuple{Chebyshev,Integer}, ::Type{T}) where {T}
    τ = value(𝒮)
    if iszero(τ)
        return ifelse(i == j, one(T), zero(T))
    else # TODO: lift restriction
        return throw(DomainError)
    end
end



# action

Base.:*(𝒮::Shift, a::AbstractSequence) = shift(a, value(𝒮))

function shift(a::Sequence, τ::Union{Number,Tuple{Vararg{Number}}})
    𝒮 = Shift(τ)
    space_a = space(a)
    new_space = codomain(𝒮, space_a)
    CoefType = _coeftype(𝒮, space_a, eltype(a))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _apply!(c, 𝒮, a)
    return c
end

function shift!(c::Sequence, a::Sequence, τ::Union{Number,Tuple{Vararg{Number}}})
    𝒮 = Shift(τ)
    space_c = space(c)
    new_space = codomain(𝒮, space(a))
    space_c == new_space || return throw(ArgumentError("spaces must be equal: c has space $space_c, $𝒮(a) has space $new_space"))
    _apply!(c, 𝒮, a)
    return c
end

# Tensor space

function _apply!(c::Sequence{<:TensorSpace}, 𝒮::Shift, a)
    space_a = space(a)
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    C = _no_alloc_reshape(coefficients(c), dimensions(space(c)))
    _apply!(C, 𝒮, space_a, A)
    return c
end

_apply!(C, 𝒮::Shift, space::TensorSpace{<:NTuple{N₁,BaseSpace}}, A::AbstractArray{T,N₂}) where {N₁,T,N₂} =
    @inbounds _apply!(C, Shift(value(𝒮)[1]), space[1], Val(N₂-N₁+1), _apply!(C, Shift(Base.tail(value(𝒮))), Base.tail(space), A))

_apply!(C, 𝒮::Shift, space::TensorSpace{<:Tuple{BaseSpace}}, A::AbstractArray) =
    @inbounds _apply!(C, Shift(value(𝒮)[1]), space[1], A)

# Taylor

function _apply!(c::Sequence{Taylor}, 𝒮::Shift, a)
    τ = value(𝒮)
    if iszero(τ)
        coefficients(c) .= coefficients(a)
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return c
end

function _apply!(C, 𝒮::Shift, ::Taylor, ::Val{D}, A) where {D}
    τ = value(𝒮)
    iszero(τ) || return throw(DomainError) # TODO: lift restriction
    return C
end

function _apply!(C::AbstractArray{T,N}, 𝒮::Shift, ::Taylor, A) where {T,N}
    τ = value(𝒮)
    if iszero(τ)
        C .= A
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return C
end

# Fourier

function _apply!(c::Sequence{<:Fourier}, 𝒮::Shift, a)
    τ = value(𝒮)
    if iszero(τ)
        coefficients(c) .= coefficients(a)
    else
        @inbounds c[0] = a[0]
        eiωτ = cis(frequency(a)*τ)
        eiωτj = one(eiωτ)
        @inbounds for j ∈ 1:order(a)
            eiωτj *= eiωτ
            c[j] = eiωτj * a[j]
            c[-j] = a[-j] / eiωτj
        end
    end
    return c
end

function _apply!(C, 𝒮::Shift, space::Fourier, ::Val{D}, A) where {D}
    τ = value(𝒮)
    if !iszero(τ)
        ord = order(space)
        eiωτ = cis(frequency(space)*τ)
        eiωτj = one(eiωτ)
        @inbounds for j ∈ 1:ord
            eiωτj *= eiωτ
            selectdim(C, D, ord+1+j) .*= eiωτj
            selectdim(C, D, ord+1-j) ./= eiωτj
        end
    end
    return C
end

function _apply!(C::AbstractArray{T,N}, 𝒮::Shift, space::Fourier, A) where {T,N}
    τ = value(𝒮)
    if iszero(τ)
        C .= A
    else
        ord = order(space)
        @inbounds selectdim(C, N, ord+1) .= selectdim(A, N, ord+1)
        eiωτ = cis(frequency(space)*τ)
        eiωτj = one(eiωτ)
        @inbounds for j ∈ 1:ord
            eiωτj *= eiωτ
            selectdim(C, N, ord+1+j) .= eiωτj .* selectdim(A, N, ord+1+j)
            selectdim(C, N, ord+1-j) .= selectdim(A, N, ord+1-j) ./ eiωτj
        end
    end
    return C
end

# Chebyshev

function _apply!(c::Sequence{Chebyshev}, 𝒮::Shift, a)
    τ = value(𝒮)
    if iszero(τ)
        coefficients(c) .= coefficients(a)
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return c
end

function _apply!(C, 𝒮::Shift, ::Chebyshev, ::Val{D}, A) where {D}
    τ = value(𝒮)
    iszero(τ) || return throw(DomainError) # TODO: lift restriction
    return C
end

function _apply!(C::AbstractArray{T,N}, 𝒮::Shift, ::Chebyshev, A) where {T,N}
    τ = value(𝒮)
    if iszero(τ)
        C .= A
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return C
end
