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
    return Fourier(Int(order(s) ÷ abs(γ)), frequency(s))
end

function codomain(𝒮::Scale{<:Number}, s::Fourier)
    γ = value(𝒮)
    isinteger(γ) || return throw(DomainError(𝒮, "Scale value must be an integer for Fourier spaces"))
    return Fourier(Int(order(s) * abs(γ)), frequency(s))
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
        return throw(DomainError(𝒮, "Scale is only defined for γ = 1 for Chebyshev spaces"))
    end
end

# Symmetric space

# on a Fourier axis the scale dilates the indices, k ↦ γk, and on a Taylor axis it multiplies the k-th
# coefficient by γ^k; a Chebyshev axis only admits γ = 1. Writing Γ for the diagonal matrix of the Fourier
# scales (1 elsewhere) and τ(k) = ∏ γ_l^{k_l} over the Taylor axes, the symmetry a_k = α(k) a_{βk} becomes
#     c_k = α(Γ⁻¹k) c_{ΓβΓ⁻¹k}    whenever τ∘β = τ,
# so the scaled sequence is invariant under the elements with ΓβΓ⁻¹ integer and τ∘β = τ, transformed into
# the automorphism ΓβΓ⁻¹ with the phase Γ⁻¹φ; the other elements are no longer symmetries and are dropped,
# as for the partial evaluation. The domain inverts the dilation, i.e. uses Γ⁻¹ in place of Γ.

function domain(𝒮::Scale, s::SymmetricSpace)
    V = domain(𝒮, desymmetrize(s))
    V isa UndefSpace && return UndefSpace()
    return SymmetricSpace(V, _scale_symmetry(symmetry(s), desymmetrize(s), 𝒮, -1))
end

function codomain(𝒮::Scale, s::SymmetricSpace)
    V = codomain(𝒮, desymmetrize(s))
    return SymmetricSpace(V, _scale_symmetry(symmetry(s), desymmetrize(s), 𝒮, 1))
end

_scale_symmetry(G::Group, s::BaseSpace, 𝒮::Scale{<:Number}, direction::Int) =
    _scale_symmetry(G, (s,), (value(𝒮),), direction)
_scale_symmetry(G::Group, s::TensorSpace{<:NTuple{N,BaseSpace}}, 𝒮::Scale{<:NTuple{N,Number}}, direction::Int) where {N} =
    _scale_symmetry(G, spaces(s), value(𝒮), direction)

function _scale_symmetry(G::Group{N,L}, s::NTuple{N,BaseSpace}, γ::NTuple{N,Number}, direction::Int) where {N,L}
    Γ = map(_index_dilation, s, γ) # k ↦ γk on a Fourier axis, the identity elsewhere
    τ = map(_coefficient_factor, s, γ) # a_k ↦ γ^k a_k on a Taylor axis, the identity elsewhere
    direction == -1 && (Γ = map(inv, Γ))
    return unsafe_group!(Set{GroupElement{N,L}}(_scale_groupelem(g, Γ) for g ∈ elements(G) if _scale_preserves(g, Γ, τ)))
end

function _index_dilation(::Fourier, γ::Number)
    isinteger(γ) || return throw(DomainError(γ, "Scale value must be an integer for Fourier spaces"))
    return Int(γ) // 1
end
_index_dilation(::Taylor, ::Number) = 1 // 1
function _index_dilation(::Chebyshev, γ::Number)
    isone(γ) || return throw(DomainError(γ, "Scale is only defined for γ = 1 for Chebyshev spaces"))
    return 1 // 1
end

_coefficient_factor(::Taylor, γ::Number) = γ
_coefficient_factor(::BaseSpace, γ::Number) = one(γ)

_conjugate_lattice_aut(β::StaticArrays.SMatrix{N,N,Int}, Γ::NTuple{N,Rational{Int}}) where {N} =
    StaticArrays.SMatrix{N,N,Rational{Int}}(Γ[l] * β[l,m] / Γ[m] for l ∈ 1:N, m ∈ 1:N) # ΓβΓ⁻¹

function _scale_preserves(g::GroupElement{N}, Γ::NTuple{N,Rational{Int}}, τ::NTuple{N,Number}) where {N}
    β = g.lattice_aut.matrix
    all(isinteger, _conjugate_lattice_aut(β, Γ)) || return false
    # τ∘β = τ, i.e. ∏_l τ_l^{β_{lm}} = τ_m for every m, compared with the negative powers moved across
    for m ∈ 1:N
        lhs, rhs = one(τ[m]), τ[m]
        for l ∈ 1:N
            β[l,m] > 0 && (lhs *= τ[l] ^ β[l,m])
            β[l,m] < 0 && (rhs *= τ[l] ^ (-β[l,m]))
        end
        _safe_isequal(lhs, rhs) || return false
    end
    return true
end

_scale_groupelem(g::GroupElement{N}, Γ::NTuple{N,Rational{Int}}) where {N} = GroupElement(
    LatticeAut(StaticArrays.SMatrix{N,N,Int}(_conjugate_lattice_aut(g.lattice_aut.matrix, Γ))),
    Cocycle(g.cocycle.angle, map(/, g.cocycle.phase, Γ)))

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

# the scale is an isometry onto the space whose weight is the pulled back one, and it maps the finite part
# onto the finite part of the codomain, so the three errors are carried over unchanged:
# - Fourier dilates the indices, k ↦ γk, and ν^{|γk|} = (ν^{|γ|})^{|k|} gives the rate ν^{1/|γ|}
# - Taylor multiplies the k-th coefficient by γ^k, which gives the rate ν/|γ|, only a weight when ≥ 1
# - Chebyshev only admits γ = 1
function scale(a::InfiniteSequence, γ::Union{Number,Tuple{Vararg{Number}}})
    c = scale(sequence(a), γ)
    X = Ell1(_scale_weight(weight(banachspace(a)), space(a), γ))
    return InfiniteSequence(c, finite_error(a), tail_error(a), total_error(a), X)
end

_scale_weight(w::NTuple{N,Weight}, s::TensorSpace{<:NTuple{N,BaseSpace}}, γ::NTuple{N,Number}) where {N} =
    map(_scale_weight, w, spaces(s), γ)
_scale_weight(w::Weight, s::TensorSpace{<:NTuple{N,BaseSpace}}, γ::NTuple{N,Number}) where {N} =
    _scale_weight(ntuple(_ -> w, Val(N)), s, γ)

_scale_weight(w::IdentityWeight, ::BaseSpace, ::Number) = w
_scale_weight(w::Weight, ::BaseSpace, ::Number) =
    throw(DomainError(w, "Scale of an InfiniteSequence is only implemented for IdentityWeight and GeometricWeight"))

function _scale_weight(w::GeometricWeight, ::Fourier, γ::Number)
    isone(abs(γ)) && return w
    return GeometricWeight(exp(log(rate(w)) / abs(γ)))
end

function _scale_weight(w::GeometricWeight, ::Taylor, γ::Number)
    isone(abs(γ)) && return w
    ν = rate(w) / abs(γ)
    inf(ν) ≥ 1 || return throw(DomainError(γ, "Scale by γ maps the geometric rate ν onto ν/|γ| on a Taylor space, which must be at least one"))
    return GeometricWeight(ν)
end

_scale_weight(w::GeometricWeight, ::Chebyshev, ::Number) = w

# Tensor space

function _apply!(c::Sequence{<:TensorSpace}, 𝒮::Scale, a)
    space_a = space(a)
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    C = _no_alloc_reshape(coefficients(c), dimensions(space(c)))
    _apply!(C, 𝒮, space_a, A)
    return c
end

_apply!(C, 𝒮::Scale, space::TensorSpace{<:NTuple{N₁,BaseSpace}}, A::AbstractArray{T,N₂}) where {N₁,T,N₂} =
    @inbounds _apply!(C, Scale(value(𝒮)[1]), space[1], _apply(Scale(Base.tail(value(𝒮))), Base.tail(space), A))
_apply!(C, 𝒮::Scale, space::TensorSpace{<:Tuple{BaseSpace}}, A::AbstractArray) =
    @inbounds _apply!(C, Scale(value(𝒮)[1]), space[1], A)

_apply(𝒮::Scale{<:NTuple{N₁,Number}}, space::TensorSpace{<:NTuple{N₁,BaseSpace}}, A::AbstractArray{T,N₂}) where {N₁,T,N₂} =
    @inbounds _apply(Scale(value(𝒮)[1]), space[1], Val(N₂-N₁+1), _apply(Scale(Base.tail(value(𝒮))), Base.tail(space), A))
_apply(𝒮::Scale{<:Tuple{Number}}, space::TensorSpace{<:Tuple{BaseSpace}}, A::AbstractArray{T,N}) where {T,N} =
    @inbounds _apply(Scale(value(𝒮)[1]), space[1], Val(N), A)

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

function _apply!(C::AbstractArray, 𝒮::Scale, space::Taylor, A)
    γ = value(𝒮)
    if isone(γ)
        C .= A
    else
        @inbounds selectdim(C, 1, 1) .= selectdim(A, 1, 1)
        γⁱ = one(γ)
        @inbounds for i ∈ 1:order(space)
            γⁱ *= γ
            selectdim(C, 1, i+1) .= γⁱ .* selectdim(A, 1, i+1)
        end
    end
    return C
end

function _apply(𝒮::Scale, space::Taylor, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    γ = value(𝒮)
    CoefType = _coeftype(𝒮, space, T)
    isone(γ) && return convert(Array{CoefType,N}, A)
    C = Array{CoefType,N}(undef, size(A))
    @inbounds selectdim(C, D, 1) .= selectdim(A, D, 1)
    γⁱ = one(γ)
    @inbounds for i ∈ 1:order(space)
        γⁱ *= γ
        selectdim(C, D, i+1) .= γⁱ .* selectdim(A, D, i+1)
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
                c[k] = a[Int(k ÷ γ)]
            else
                c[k] = zero(eltype(c))
            end
        end
    end
    return c
end

function _apply!(C::AbstractArray{T}, 𝒮::Scale, space::Fourier, A) where {T}
    γ = value(𝒮)
    if isone(γ)
        C .= A
    else
        ord_A = order(space)
        ord = Int(ord_A * abs(γ))
        @inbounds for k ∈ -ord:ord
            if k % γ == 0
                selectdim(C, 1, k+ord+1) .= selectdim(A, 1, Int(k ÷ γ)+ord_A+1)
            else
                selectdim(C, 1, k+ord+1) .= zero(T)
            end
        end
    end
    return C
end

function _apply(𝒮::Scale, space::Fourier, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    γ = value(𝒮)
    CoefType = _coeftype(𝒮, space, T)
    isone(γ) && return convert(Array{CoefType,N}, A)
    ord_A = order(space)
    ord = Int(ord_A * abs(γ))
    C = Array{CoefType,N}(undef, ntuple(i -> ifelse(i == D, 2ord+1, size(A, i)), Val(N)))
    @inbounds for k ∈ -ord:ord
        if k % γ == 0
            selectdim(C, D, k+ord+1) .= selectdim(A, D, Int(k ÷ γ)+ord_A+1)
        else
            selectdim(C, D, k+ord+1) .= zero(CoefType)
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
        return throw(DomainError(𝒮, "Scale is only defined for γ = 1 for Chebyshev spaces"))
    end
    return c
end

function _apply(𝒮::Scale, space::Chebyshev, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    γ = value(𝒮)
    CoefType = _coeftype(𝒮, space, T)
    isone(γ) && return convert(Array{CoefType,N}, A)
    return throw(DomainError(𝒮, "Scale is only defined for γ = 1 for Chebyshev spaces")) # TODO: lift restriction
end

function _apply!(C::AbstractArray{T,N}, 𝒮::Scale, ::Chebyshev, A) where {T,N}
    γ = value(𝒮)
    if isone(γ)
        C .= A
    else # TODO: lift restriction
        return throw(DomainError(𝒮, "Scale is only defined for γ = 1 for Chebyshev spaces"))
    end
    return C
end
