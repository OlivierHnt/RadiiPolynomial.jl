struct Shift{T}
    value :: T
end

(𝒮::Shift)(a::Sequence) = *(𝒮, a)
Base.:*(𝒮::Shift, a::Sequence) = shift(a, 𝒮.value)

function shift(a::Sequence, τ)
    𝒮 = Shift(τ)
    space_a = space(a)
    new_space = image(𝒮, space_a)
    CoefType = _coeftype(𝒮, space_a, eltype(a))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _apply!(c, 𝒮, a)
    return c
end

function shift!(c::Sequence, a::Sequence, τ)
    𝒮 = Shift(τ)
    space(c) == image(𝒮, space(a)) || return throw(DomainError)
    _apply!(c, 𝒮, a)
    return c
end

function project(𝒮::Shift, domain::VectorSpace, codomain::VectorSpace, ::Type{T}) where {T}
    _iscompatible(domain, codomain) || return throw(DimensionMismatch)
    ind_domain = _findposition_nzind_domain(𝒮, domain, codomain)
    ind_codomain = _findposition_nzind_codomain(𝒮, domain, codomain)
    C = LinearOperator(domain, codomain, sparse(ind_codomain, ind_domain, zeros(T, length(ind_domain)), dimension(codomain), dimension(domain)))
    _project!(C, 𝒮)
    return C
end

function project!(C::LinearOperator, 𝒮::Shift)
    _iscompatible(domain(C), codomain(C)) || return throw(DimensionMismatch)
    coefficients(C) .= zero(eltype(C))
    _project!(C, 𝒮)
    return C
end

_findposition_nzind_domain(𝒮::Shift, domain, codomain) =
    _findposition(_nzind_domain(𝒮, domain, codomain), domain)

_findposition_nzind_codomain(𝒮::Shift, domain, codomain) =
    _findposition(_nzind_codomain(𝒮, domain, codomain), codomain)

# Sequence spaces

image(𝒮::Shift{<:NTuple{N,Any}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((τᵢ, sᵢ) -> image(Shift(τᵢ), sᵢ), 𝒮.value, spaces(s)))

_coeftype(𝒮::Shift, s::TensorSpace, ::Type{T}) where {T} =
    @inbounds promote_type(_coeftype(Shift(𝒮.value[1]), s[1], T), _coeftype(Shift(Base.tail(𝒮.value)), Base.tail(s), T))
_coeftype(𝒮::Shift, s::TensorSpace{<:Tuple{BaseSpace}}, ::Type{T}) where {T} =
    @inbounds _coeftype(Shift(𝒮.value[1]), s[1], T)

function _apply!(c::Sequence{<:TensorSpace}, 𝒮::Shift, a)
    space_a = space(a)
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    C = _no_alloc_reshape(coefficients(c), dimensions(space(c)))
    _apply!(C, 𝒮, space_a, A)
    return c
end

_apply!(C, 𝒮::Shift, space::TensorSpace{<:NTuple{N₁,BaseSpace}}, A::AbstractArray{T,N₂}) where {N₁,T,N₂} =
    @inbounds _apply!(C, Shift(𝒮.value[1]), space[1], Val(N₂-N₁+1), _apply!(C, Shift(Base.tail(𝒮.value)), Base.tail(space), A))

_apply!(C, 𝒮::Shift, space::TensorSpace{<:Tuple{BaseSpace}}, A::AbstractArray) =
    @inbounds _apply!(C, Shift(𝒮.value[1]), space[1], A)

for (_f, __f) ∈ ((:_nzind_domain, :__nzind_domain), (:_nzind_codomain, :__nzind_codomain))
    @eval begin
        $_f(𝒮::Shift{<:NTuple{N,Any}}, domain::TensorSpace{<:NTuple{N,BaseSpace}}, codomain::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
            TensorIndices($__f(𝒮, domain, codomain))
        $__f(𝒮::Shift, domain::TensorSpace, codomain) =
            @inbounds ($_f(Shift(𝒮.value[1]), domain[1], codomain[1]), $__f(Shift(Base.tail(𝒮.value)), Base.tail(domain), Base.tail(codomain))...)
        $__f(𝒮::Shift, domain::TensorSpace{<:Tuple{BaseSpace}}, codomain) =
            @inbounds ($_f(Shift(𝒮.value[1]), domain[1], codomain[1]),)
    end
end

function _project!(C::LinearOperator{<:SequenceSpace,<:SequenceSpace}, 𝒮::Shift)
    domain_C = domain(C)
    codomain_C = codomain(C)
    CoefType = eltype(C)
    @inbounds for (α, β) ∈ zip(_nzind_codomain(𝒮, domain_C, codomain_C), _nzind_domain(𝒮, domain_C, codomain_C))
        C[α,β] = _nzval(𝒮, domain_C, codomain_C, CoefType, α, β)
    end
    return C
end

@generated function _nzval(𝒮::Shift{<:NTuple{N,Any}}, domain::TensorSpace{<:NTuple{N,BaseSpace}}, codomain::TensorSpace{<:NTuple{N,BaseSpace}}, ::Type{T}, α, β) where {N,T}
    p = :(_nzval(Shift(𝒮.value[1]), domain[1], codomain[1], T, α[1], β[1]))
    for i ∈ 2:N
        p = :(_nzval(Shift(𝒮.value[$i]), domain[$i], codomain[$i], T, α[$i], β[$i]) * $p)
    end
    return p
end

# Fourier

image(::Shift, s::Fourier) = s

_coeftype(::Shift{T}, s::Fourier, ::Type{S}) where {T,S} =
    promote_type(typeof(cis(frequency(s)*zero(T))), S)

function _apply!(c::Sequence{<:Fourier}, 𝒮::Shift, a)
    τ = 𝒮.value
    if iszero(τ)
        coefficients(c) .= coefficients(a)
    else
        @inbounds c[0] = a[0]
        eiωτ = cis(frequency(a)*τ)
        eiωτj = one(eiωτ)
        @inbounds for j ∈ 1:order(a)
            eiωτj *= eiωτ
            c[j] = eiωτj * a[j]
            c[-j] = conj(eiωτj) * a[-j]
        end
    end
    return c
end

function _apply!(C, 𝒮::Shift, space::Fourier, ::Val{D}, A) where {D}
    τ = 𝒮.value
    if !iszero(τ)
        ord = order(space)
        eiωτ = cis(frequency(space)*τ)
        eiωτj = one(eiωτ)
        @inbounds for j ∈ 1:ord
            eiωτj *= eiωτ
            selectdim(C, D, ord+1+j) .*= eiωτj
            selectdim(C, D, ord+1-j) .*= conj(eiωτj)
        end
    end
    return C
end

function _apply!(C::AbstractArray{T,N}, 𝒮::Shift, space::Fourier, A) where {T,N}
    τ = 𝒮.value
    if iszero(τ)
        C .= A
    else
        ord = order(space)
        @inbounds selectdim(C, N, 1) .= selectdim(A, N, 1)
        eiωτ = cis(frequency(space)*τ)
        eiωτj = one(eiωτ)
        @inbounds for j ∈ 1:ord
            eiωτj *= eiωτ
            selectdim(C, N, ord+1+j) .= eiωτj .* selectdim(A, N, ord+1+j)
            selectdim(C, N, ord+1-j) .= conj(eiωτj) .* selectdim(A, N, ord+1-j)
        end
    end
    return C
end

function _nzind_domain(::Shift, domain::Fourier, codomain::Fourier)
    ord = min(order(domain), order(codomain))
    return -ord:ord
end
function _nzind_codomain(::Shift, domain::Fourier, codomain::Fourier)
    ord = min(order(domain), order(codomain))
    return -ord:ord
end
function _nzval(𝒮::Shift, domain::Fourier, ::Fourier, ::Type{T}, i, j) where {T}
    τ = 𝒮.value
    if iszero(τ)
        return one(T)
    else
        return convert(T, cis(frequency(domain)*τ*i))
    end
end

# Cartesian spaces

image(𝒮::Shift, s::CartesianPower) =
    CartesianPower(image(𝒮, space(s)), nb_cartesian_product(s))

image(𝒮::Shift, s::CartesianProduct) =
    CartesianProduct(map(sᵢ -> image(𝒮, sᵢ), spaces(s)))

_coeftype(𝒮::Shift, s::CartesianPower, ::Type{T}) where {T} =
    _coeftype(𝒮, space(s), T)

_coeftype(𝒮::Shift, s::CartesianProduct, ::Type{T}) where {T} =
    @inbounds promote_type(_coeftype(𝒮, s[1], T), _coeftype(𝒮, Base.tail(s), T))
_coeftype(𝒮::Shift, s::CartesianProduct{<:Tuple{VectorSpace}}, ::Type{T}) where {T} =
    @inbounds _coeftype(𝒮, s[1], T)

function _apply!(c::Sequence{<:CartesianPower}, 𝒮::Shift, a)
    @inbounds for i ∈ 1:nb_cartesian_product(space(c))
        _apply!(component(c, i), 𝒮, component(a, i))
    end
    return c
end
function _apply!(c::Sequence{CartesianProduct{T}}, 𝒮::Shift, a) where {N,T<:NTuple{N,VectorSpace}}
    @inbounds _apply!(component(c, 1), 𝒮, component(a, 1))
    @inbounds _apply!(component(c, 2:N), 𝒮, component(a, 2:N))
    return c
end
function _apply!(c::Sequence{CartesianProduct{T}}, 𝒮::Shift, a) where {T<:Tuple{VectorSpace}}
    @inbounds _apply!(component(c, 1), 𝒮, component(a, 1))
    return c
end

function _findposition_nzind_domain(𝒮::Shift, domain::CartesianSpace, codomain::CartesianSpace)
    u = map((dom, codom) -> _findposition_nzind_domain(𝒮, dom, codom), spaces(domain), spaces(codomain))
    len = sum(length, u)
    v = Vector{Int}(undef, len)
    δ = δδ = 0
    @inbounds for (i, uᵢ) in enumerate(u)
        δ_ = δ
        δ += length(uᵢ)
        view(v, 1+δ_:δ) .= δδ .+ uᵢ
        δδ += dimension(domain[i])
    end
    return v
end

function _findposition_nzind_codomain(𝒮::Shift, domain::CartesianSpace, codomain::CartesianSpace)
    u = map((dom, codom) -> _findposition_nzind_codomain(𝒮, dom, codom), spaces(domain), spaces(codomain))
    len = sum(length, u)
    v = Vector{Int}(undef, len)
    δ = δδ = 0
    @inbounds for (i, uᵢ) in enumerate(u)
        δ_ = δ
        δ += length(uᵢ)
        view(v, 1+δ_:δ) .= δδ .+ uᵢ
        δδ += dimension(codomain[i])
    end
    return v
end

function _project!(C::LinearOperator{<:CartesianSpace,<:CartesianSpace}, 𝒮::Shift)
    @inbounds for i ∈ 1:nb_cartesian_product(domain(C))
        _project!(component(C, i, i), 𝒮)
    end
    return C
end
