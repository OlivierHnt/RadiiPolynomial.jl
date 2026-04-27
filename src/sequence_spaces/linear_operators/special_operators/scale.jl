"""
    Scale{T<:Union{Number,Tuple{Vararg{Number}}}} <: AbstractLinearOperator

Generic scale operator.

Field:
- `value :: T`

Constructors:
- `Scale(::Number)`
- `Scale(::Tuple{Vararg{Number}})`
- `Scale(value::Number...)`: equivalent to `Scale(value)`

# Examples

```jldoctest
julia> Scale(1.0)
Scale{Float64}(1.0)

julia> Scale(1.0, 2.0)
Scale{Tuple{Float64, Float64}}((1.0, 2.0))
```
"""
struct Scale{T<:Union{Number,Tuple{Vararg{Number}}}} <: AbstractLinearOperator
    value :: T
    Scale{T}(value::T) where {T<:Union{Number,Tuple{Vararg{Number}}}} = new{T}(value)
    Scale{Tuple{}}(::Tuple{}) = throw(ArgumentError("Scale is only defined for at least one Number"))
end

Scale(value::T) where {T<:Number} = Scale{T}(value)
Scale(value::T) where {T<:Tuple{Vararg{Number}}} = Scale{T}(value)
Scale(value::Number...) = Scale(value)

value(𝒮::Scale) = 𝒮.value

Base.:*(𝒮₁::Scale{<:Number}, 𝒮₂::Scale{<:Number}) = Scale(value(𝒮₁) * value(𝒮₂))
Base.:*(𝒮₁::Scale{<:NTuple{N,Number}}, 𝒮₂::Scale{<:NTuple{N,Number}}) where {N} = Scale(map(*, value(𝒮₁), value(𝒮₂)))

Base.:^(𝒮::Scale{<:Number}, n::Integer) = Scale(value(𝒮) ^ exact(n))
Base.:^(𝒮::Scale{<:Tuple{Vararg{Number}}}, n::Integer) = Scale(map(γᵢ -> γᵢ ^ exact(n), value(𝒮)))
Base.:^(𝒮::Scale{<:NTuple{N,Number}}, n::NTuple{N,Integer}) where {N} = Scale(map((γᵢ, nᵢ) -> γᵢ ^ exact(nᵢ), value(𝒮), n))

# Tensor space

domain(S::Scale{<:NTuple{N,Number}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((γᵢ, sᵢ) -> domain(Scale(γᵢ), sᵢ), value(S), spaces(s)))

codomain(𝒮::Scale{<:NTuple{N,Number}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((γᵢ, sᵢ) -> codomain(Scale(γᵢ), sᵢ), value(𝒮), spaces(s)))

_coeftype(𝒮::Scale{<:NTuple{N,Number}}, s::TensorSpace{<:NTuple{N,BaseSpace}}, ::Type{T}) where {N,T} =
    @inbounds promote_type(_coeftype(Scale(value(𝒮)[1]), s[1], T), _coeftype(Scale(Base.tail(value(𝒮))), Base.tail(s), T))
_coeftype(𝒮::Scale{<:Tuple{Number}}, s::TensorSpace{<:Tuple{BaseSpace}}, ::Type{T}) where {T} =
    @inbounds _coeftype(Scale(value(𝒮)[1]), s[1], T)

getcoefficient(𝒮::Scale{<:NTuple{N,Number}}, (codom, i)::Tuple{TensorSpace{<:NTuple{N,BaseSpace}},NTuple{N,Integer}}, (dom, j)::Tuple{TensorSpace{<:NTuple{N,BaseSpace}},NTuple{N,Integer}}, ::Type{T}) where {N,T} =
    @inbounds getcoefficient(Scale(value(𝒮)[1]), (codom[1], i[1]), (dom[1], j[1]), T) * getcoefficient(Scale(Base.tail(value(𝒮))), (Base.tail(codom), Base.tail(i)), (Base.tail(dom), Base.tail(j)), T)
getcoefficient(𝒮::Scale{<:Tuple{Number}}, (codom, i)::Tuple{TensorSpace{<:Tuple{BaseSpace}},Tuple{Integer}}, (dom, j)::Tuple{TensorSpace{<:Tuple{BaseSpace}},Tuple{Integer}}, ::Type{T}) where {T} =
    @inbounds getcoefficient(Scale(value(𝒮)[1]), (codom[1], i[1]), (dom[1], j[1]), T)

# Taylor

domain(::Scale{<:Number}, s::Taylor) = s

codomain(::Scale{<:Number}, s::Taylor) = s

_coeftype(::Scale{T}, ::Taylor, ::Type{S}) where {T<:Number,S} = promote_type(T, S)

function getcoefficient(𝒮::Scale{<:Number}, (codom, i)::Tuple{Taylor,Integer}, (dom, j)::Tuple{Taylor,Integer}, ::Type{T}) where {T}
    i == j || return zero(T)
    γ = value(𝒮)
    isone(γ) && return one(T)
    return convert(T, γ ^ exact(i))
end

# Fourier

function domain(𝒮::Scale{<:Number}, s::Fourier)
    γ = value(𝒮)
    isinteger(γ) || return throw(DomainError(𝒮, "Scale value must be an integer for Fourier spaces"))
    return Fourier(order(s) ÷ abs(γ), frequency(s))
end

function codomain(𝒮::Scale{<:Number}, s::Fourier)
    γ = value(𝒮)
    isinteger(γ) || return throw(DomainError(𝒮, "Scale value must be an integer for Fourier spaces"))
    return Fourier(order(s) * abs(γ), frequency(s))
end

_coeftype(::Scale{<:Number}, ::Fourier, ::Type{T}) where {T} = T

getcoefficient(𝒮::Scale{<:Number}, (codom, i)::Tuple{Fourier,Integer}, (dom, j)::Tuple{Fourier,Integer}, ::Type{T}) where {T} = ifelse(i == j * value(𝒮), one(T), zero(T))

# Chebyshev

domain(::Scale{<:Number}, s::Chebyshev) = s

codomain(::Scale{<:Number}, s::Chebyshev) = s

_coeftype(::Scale{T}, ::Chebyshev, ::Type{S}) where {T<:Number,S} = promote_type(T, S)

function getcoefficient(𝒮::Scale{<:Number}, (codom, i)::Tuple{Chebyshev,Integer}, (dom, j)::Tuple{Chebyshev,Integer}, ::Type{T}) where {T}
    γ = value(𝒮)
    if isone(γ)
        return ifelse(i == j, one(T), zero(T))
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

# Symmetric space

function domain(𝒮::Scale, s::SymmetricSpace{<:Fourier})
    γ = value(𝒮)
    all(isinteger, indices(s) ./ γ) || return throw(DomainError(𝒮, "Division of indices by Scale value must yield a lattice for symmetric Fourier spaces"))
    V = Fourier(order(s) ÷ abs(γ), frequency(s))
    G = unsafe_group!(Set(_groupelem_invscale(𝒮, g, desymmetrize(s))
                  for g ∈ elements(symmetry(s))))
    return SymmetricSpace(V, G)
end

function _groupelem_invscale(𝒮::Scale, g::GroupElement, ::Fourier)
    new_va = CoefAction(g.coef_action.amplitude,
                      g.coef_action.phase * value(𝒮))
    return GroupElement(g.index_action, new_va)
end

function codomain(𝒮::Scale, s::SymmetricSpace{<:Fourier})
    γ = value(𝒮)
    all(isinteger, indices(s) .* γ) || return throw(DomainError(𝒮, "Multiplication of indices by Scale value must yield a lattice for symmetric Fourier spaces"))
    V = Fourier(Int(order(s) * abs(γ)), frequency(s))
    G = unsafe_group!(Set(_groupelem_scale(𝒮, g, desymmetrize(s)) for g ∈ elements(symmetry(s))))
    return SymmetricSpace(V, G)
end

function _groupelem_scale(𝒮::Scale, g::GroupElement, ::Fourier)
    new_va = CoefAction(g.coef_action.amplitude,
                      g.coef_action.phase / value(𝒮))
    return GroupElement(g.index_action, new_va)
end

_coeftype(𝒮::Scale, s::SymmetricSpace, ::Type{T}) where {T} =
    _coeftype(𝒮, desymmetrize(s), T)



# action

Base.:*(𝒮::Scale, a::AbstractSequence) = scale(a, value(𝒮))

function scale(a::Sequence, γ::Union{Number,Tuple{Vararg{Number}}})
    𝒮 = Scale(γ)
    space_a = space(a)
    new_space = codomain(𝒮, space_a)
    CoefType = _coeftype(𝒮, space_a, eltype(a))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _apply!(c, 𝒮, a)
    return c
end

function scale!(c::Sequence, a::Sequence, γ::Union{Number,Tuple{Vararg{Number}}})
    𝒮 = Scale(γ)
    space_c = space(c)
    new_space = codomain(𝒮, space(a))
    space_c == new_space || return throw(ArgumentError("spaces must be equal: c has space $space_c, $𝒮(a) has space $new_space"))
    _apply!(c, 𝒮, a)
    return c
end

# Tensor space

function _apply!(c::Sequence{<:TensorSpace}, 𝒮::Scale, a)
    space_a = space(a)
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    C = _no_alloc_reshape(coefficients(c), dimensions(space(c)))
    _apply!(C, 𝒮, space_a, A)
    return c
end

_apply!(C, 𝒮::Scale, space::TensorSpace{<:NTuple{N₁,BaseSpace}}, A::AbstractArray{T,N₂}) where {N₁,T,N₂} =
    @inbounds _apply!(C, Scale(value(𝒮)[1]), space[1], Val(N₂-N₁+1), _apply!(C, Scale(Base.tail(value(𝒮))), Base.tail(space), A))
_apply!(C, 𝒮::Scale, space::TensorSpace{<:Tuple{BaseSpace}}, A::AbstractArray) =
    @inbounds _apply!(C, Scale(value(𝒮)[1]), space[1], A)

# Taylor

function _apply!(c::Sequence{Taylor}, 𝒮::Scale, a)
    γ = value(𝒮)
    if isone(γ)
        coefficients(c) .= coefficients(a)
    else
        @inbounds c[0] = a[0]
        γⁱ = one(γ)
        @inbounds for i ∈ 1:order(a)
            γⁱ *= γ
            c[i] = a[i]*γⁱ
        end
    end
    return c
end

function _apply!(C, 𝒮::Scale, space::Taylor, ::Val{D}, A) where {D}
    γ = value(𝒮)
    if !isone(γ)
        γⁱ = one(γ)
        @inbounds for i ∈ 1:order(space)
            γⁱ *= γ
            selectdim(C, D, i+1) .*= γⁱ
        end
    end
    return C
end

function _apply!(C::AbstractArray{T,N}, 𝒮::Scale, space::Taylor, A) where {T,N}
    γ = value(𝒮)
    if isone(γ)
        C .= A
    else
        @inbounds selectdim(C, N, 1) .= selectdim(A, N, 1)
        γⁱ = one(γ)
        @inbounds for i ∈ 1:order(space)
            γⁱ *= γ
            selectdim(C, N, i+1) .= γⁱ .* selectdim(A, N, i+1)
        end
    end
    return C
end

# Fourier

function _apply!(c::Sequence{<:Fourier}, 𝒮::Scale, a)
    γ = value(𝒮)
    if isone(γ)
        coefficients(c) .= coefficients(a)
    else
        @inbounds for k ∈ indices(space(c))
            if k % γ == 0
                c[k] = a[k ÷ γ]
            else
                c[k] = zero(eltype(c))
            end
        end
    end
    return c
end

function _apply!(C, 𝒮::Scale, ::Fourier, ::Val{D}, A) where {D}
    γ = value(𝒮)
    if !isone(γ)
        @inbounds for k ∈ indices(space(c))
            if k % γ == 0
                selectdim(C, D, k) .= selectdim(A, D, k ÷ γ)
            else
                selectdim(C, D, k) .= zero(eltype(C))
            end
        end
    end
    return C
end

function _apply!(C::AbstractArray{T,N}, 𝒮::Scale, ::Fourier, A) where {T,N}
    γ = value(𝒮)
    if isone(γ)
        C .= A
    else
        @inbounds for k ∈ indices(space(c))
            if k % γ == 0
                selectdim(C, N, k) .= selectdim(A, N, k ÷ γ)
            else
                selectdim(C, N, k) .= zero(eltype(C))
            end
        end
    end
    return C
end

# Chebyshev

function _apply!(c::Sequence{Chebyshev}, 𝒮::Scale, a)
    γ = value(𝒮)
    if isone(γ)
        coefficients(c) .= coefficients(a)
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return c
end

function _apply!(C, 𝒮::Scale, ::Chebyshev, ::Val{D}, A) where {D}
    γ = value(𝒮)
    isone(γ) || return throw(DomainError) # TODO: lift restriction
    return C
end

function _apply!(C::AbstractArray{T,N}, 𝒮::Scale, ::Chebyshev, A) where {T,N}
    γ = value(𝒮)
    if isone(γ)
        C .= A
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return C
end
