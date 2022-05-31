struct Scale{T<:Union{Number,Tuple{Vararg{Number}}}}
    value :: T
end

# fallback arithmetic methods

function Base.:+(A::LinearOperator, 𝒮::Scale)
    domain_A = domain(A)
    return A + project(𝒮, domain_A, codomain(A), _coeftype(𝒮, domain_A, eltype(A)))
end
function Base.:+(𝒮::Scale, A::LinearOperator)
    domain_A = domain(A)
    return project(𝒮, domain_A, codomain(A), _coeftype(𝒮, domain_A, eltype(A))) + A
end
function Base.:-(A::LinearOperator, 𝒮::Scale)
    domain_A = domain(A)
    return A - project(𝒮, domain_A, codomain(A), _coeftype(𝒮, domain_A, eltype(A)))
end
function Base.:-(𝒮::Scale, A::LinearOperator)
    domain_A = domain(A)
    return project(𝒮, domain_A, codomain(A), _coeftype(𝒮, domain_A, eltype(A))) - A
end

add!(C::LinearOperator, A::LinearOperator, 𝒮::Scale) = add!(C, A, project(𝒮, domain(A), codomain(A), eltype(C)))
add!(C::LinearOperator, 𝒮::Scale, A::LinearOperator) = add!(C, project(𝒮, domain(A), codomain(A), eltype(C)), A)
sub!(C::LinearOperator, A::LinearOperator, 𝒮::Scale) = sub!(C, A, project(𝒮, domain(A), codomain(A), eltype(C)))
sub!(C::LinearOperator, 𝒮::Scale, A::LinearOperator) = sub!(C, project(𝒮, domain(A), codomain(A), eltype(C)), A)

radd!(A::LinearOperator, 𝒮::Scale) = radd!(A, project(𝒮, domain(A), codomain(A), eltype(A)))
rsub!(A::LinearOperator, 𝒮::Scale) = rsub!(A, project(𝒮, domain(A), codomain(A), eltype(A)))

ladd!(𝒮::Scale, A::LinearOperator) = ladd!(project(𝒮, domain(A), codomain(A), eltype(A)), A)
lsub!(𝒮::Scale, A::LinearOperator) = lsub!(project(𝒮, domain(A), codomain(A), eltype(A)), A)

function Base.:*(𝒮::Scale, A::LinearOperator)
    codomain_A = domain(A)
    return project(𝒮, codomain_A, image(𝒮, codomain_A), _coeftype(𝒮, codomain_A, eltype(A))) * A
end

LinearAlgebra.mul!(C::LinearOperator, 𝒮::Scale, A::LinearOperator, α::Number, β::Number) =
    mul!(C, project(𝒮, codomain(A), codomain(C), eltype(C)), A, α, β)
LinearAlgebra.mul!(C::LinearOperator, A::LinearOperator, 𝒮::Scale, α::Number, β::Number) =
    mul!(C, A, project(𝒮, domain(C), domain(A), eltype(C)), α, β)

#

(𝒮::Scale)(a::Sequence) = *(𝒮, a)
Base.:*(𝒮::Scale, a::Sequence) = scale(a, 𝒮.value)

function scale(a::Sequence, γ)
    𝒮 = Scale(γ)
    space_a = space(a)
    new_space = image(𝒮, space_a)
    CoefType = _coeftype(𝒮, space_a, eltype(a))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    _apply!(c, 𝒮, a)
    return c
end

function scale!(c::Sequence, a::Sequence, γ)
    𝒮 = Scale(γ)
    space_c = space(c)
    new_space = image(𝒮, space(a))
    space_c == new_space || return throw(ArgumentError("spaces must be equal: c has space $space_c, $𝒮(a) has space $new_space"))
    _apply!(c, 𝒮, a)
    return c
end

function project(𝒮::Scale, domain::VectorSpace, codomain::VectorSpace, ::Type{T}) where {T}
    _iscompatible(domain, codomain) || return throw(ArgumentError("spaces must be compatible: domain is $domain, codomain is $codomain"))
    ind_domain = _findposition_nzind_domain(𝒮, domain, codomain)
    ind_codomain = _findposition_nzind_codomain(𝒮, domain, codomain)
    C = LinearOperator(domain, codomain, sparse(ind_codomain, ind_domain, zeros(T, length(ind_domain)), dimension(codomain), dimension(domain)))
    _project!(C, 𝒮)
    return C
end

function project!(C::LinearOperator, 𝒮::Scale)
    domain_C = domain(C)
    codomain_C = codomain(C)
    _iscompatible(domain_C, codomain_C) || return throw(ArgumentError("spaces must be compatible: C has domain $domain_C, C has codomain $codomain_C"))
    coefficients(C) .= zero(eltype(C))
    _project!(C, 𝒮)
    return C
end

_findposition_nzind_domain(𝒮::Scale, domain, codomain) =
    _findposition(_nzind_domain(𝒮, domain, codomain), domain)

_findposition_nzind_codomain(𝒮::Scale, domain, codomain) =
    _findposition(_nzind_codomain(𝒮, domain, codomain), codomain)

# Sequence spaces

image(𝒮::Scale{<:NTuple{N,Number}}, s::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((γᵢ, sᵢ) -> image(Scale(γᵢ), sᵢ), 𝒮.value, spaces(s)))

_coeftype(𝒮::Scale, s::TensorSpace, ::Type{T}) where {T} =
    @inbounds promote_type(_coeftype(Scale(𝒮.value[1]), s[1], T), _coeftype(Scale(Base.tail(𝒮.value)), Base.tail(s), T))
_coeftype(𝒮::Scale, s::TensorSpace{<:Tuple{BaseSpace}}, ::Type{T}) where {T} =
    @inbounds _coeftype(Scale(𝒮.value[1]), s[1], T)

function _apply!(c::Sequence{<:TensorSpace}, 𝒮::Scale, a)
    space_a = space(a)
    A = _no_alloc_reshape(coefficients(a), dimensions(space_a))
    C = _no_alloc_reshape(coefficients(c), dimensions(space(c)))
    _apply!(C, 𝒮, space_a, A)
    return c
end

_apply!(C, 𝒮::Scale, space::TensorSpace{<:NTuple{N₁,BaseSpace}}, A::AbstractArray{T,N₂}) where {N₁,T,N₂} =
    @inbounds _apply!(C, Scale(𝒮.value[1]), space[1], Val(N₂-N₁+1), _apply!(C, Scale(Base.tail(𝒮.value)), Base.tail(space), A))

_apply!(C, 𝒮::Scale, space::TensorSpace{<:Tuple{BaseSpace}}, A::AbstractArray) =
    @inbounds _apply!(C, Scale(𝒮.value[1]), space[1], A)

for (_f, __f) ∈ ((:_nzind_domain, :__nzind_domain), (:_nzind_codomain, :__nzind_codomain))
    @eval begin
        $_f(𝒮::Scale{<:NTuple{N,Number}}, domain::TensorSpace{<:NTuple{N,BaseSpace}}, codomain::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
            TensorIndices($__f(𝒮, domain, codomain))
        $__f(𝒮::Scale, domain::TensorSpace, codomain) =
            @inbounds ($_f(Scale(𝒮.value[1]), domain[1], codomain[1]), $__f(Scale(Base.tail(𝒮.value)), Base.tail(domain), Base.tail(codomain))...)
        $__f(𝒮::Scale, domain::TensorSpace{<:Tuple{BaseSpace}}, codomain) =
            @inbounds ($_f(Scale(𝒮.value[1]), domain[1], codomain[1]),)
    end
end

function _project!(C::LinearOperator{<:SequenceSpace,<:SequenceSpace}, 𝒮::Scale)
    domain_C = domain(C)
    codomain_C = codomain(C)
    CoefType = eltype(C)
    @inbounds for (α, β) ∈ zip(_nzind_codomain(𝒮, domain_C, codomain_C), _nzind_domain(𝒮, domain_C, codomain_C))
        C[α,β] = _nzval(𝒮, domain_C, codomain_C, CoefType, α, β)
    end
    return C
end

_nzval(𝒮::Scale{<:NTuple{N,Number}}, domain::TensorSpace{<:NTuple{N,BaseSpace}}, codomain::TensorSpace{<:NTuple{N,BaseSpace}}, ::Type{T}, α, β) where {N,T} =
    @inbounds _nzval(Scale(𝒮.value[1]), domain[1], codomain[1], T, α[1], β[1]) * _nzval(Scale(Base.tail(𝒮.value)), Base.tail(domain), Base.tail(codomain), T, Base.tail(α), Base.tail(β))
_nzval(𝒮::Scale{<:Tuple{Number}}, domain::TensorSpace{<:Tuple{BaseSpace}}, codomain::TensorSpace{<:Tuple{BaseSpace}}, ::Type{T}, α, β) where {T} =
    @inbounds _nzval(Scale(𝒮.value[1]), domain[1], codomain[1], T, α[1], β[1])

# Taylor

image(::Scale, s::Taylor) = s

_coeftype(::Scale{T}, ::Taylor, ::Type{S}) where {T,S} = promote_type(T, S)

function _apply!(c::Sequence{Taylor}, 𝒮::Scale, a)
    γ = 𝒮.value
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
    γ = 𝒮.value
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
    γ = 𝒮.value
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

_nzind_domain(::Scale, domain::Taylor, codomain::Taylor) = 0:min(order(domain), order(codomain))
_nzind_codomain(::Scale, domain::Taylor, codomain::Taylor) = 0:min(order(domain), order(codomain))
function _nzval(𝒮::Scale, ::Taylor, ::Taylor, ::Type{T}, i, j) where {T}
    γ = 𝒮.value
    if isone(γ)
        return one(T)
    else
        return convert(T, γ^i)
    end
end

# Fourier

image(𝒮::Scale, s::Fourier) = Fourier(order(s), frequency(s)*𝒮.value)

_coeftype(::Scale, ::Fourier, ::Type{T}) where {T} = T

function _apply!(c::Sequence{<:Fourier}, ::Scale, a)
    coefficients(c) .= coefficients(a)
    return c
end

_apply!(C, ::Scale, ::Fourier, ::Val, A) = C

function _apply!(C, ::Scale, ::Fourier, A)
    C .= A
    return C
end

function _nzind_domain(::Scale, domain::Fourier, codomain::Fourier)
    ord = min(order(domain), order(codomain))
    return -ord:ord
end
function _nzind_codomain(::Scale, domain::Fourier, codomain::Fourier)
    ord = min(order(domain), order(codomain))
    return -ord:ord
end
_nzval(::Scale, ::Fourier, ::Fourier, ::Type{T}, i, j) where {T} = one(T)

# Chebyshev

image(::Scale, s::Chebyshev) = s

_coeftype(::Scale{T}, ::Chebyshev, ::Type{S}) where {T,S} = promote_type(T, S)

function _apply!(c::Sequence{Chebyshev}, 𝒮::Scale, a)
    γ = 𝒮.value
    if isone(γ)
        coefficients(c) .= coefficients(a)
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return c
end

function _apply!(C, 𝒮::Scale, space::Chebyshev, ::Val{D}, A) where {D}
    γ = 𝒮.value
    isone(γ) || return throw(DomainError) # TODO: lift restriction
    return C
end

function _apply!(C::AbstractArray{T,N}, 𝒮::Scale, space::Chebyshev, A) where {T,N}
    γ = 𝒮.value
    if isone(γ)
        C .= A
    else # TODO: lift restriction
        return throw(DomainError)
    end
    return C
end

_nzind_domain(::Scale, domain::Chebyshev, codomain::Chebyshev) = 0:min(order(domain), order(codomain))
_nzind_codomain(::Scale, domain::Chebyshev, codomain::Chebyshev) = 0:min(order(domain), order(codomain))
function _nzval(𝒮::Scale, ::Chebyshev, ::Chebyshev, ::Type{T}, i, j) where {T}
    γ = 𝒮.value
    if isone(γ)
        return one(T)
    else # TODO: lift restriction
        return throw(DomainError)
    end
end

# Cartesian spaces

image(𝒮::Scale, s::CartesianPower) =
    CartesianPower(image(𝒮, space(s)), nb_cartesian_product(s))

image(𝒮::Scale, s::CartesianProduct) =
    CartesianProduct(map(sᵢ -> image(𝒮, sᵢ), spaces(s)))

_coeftype(𝒮::Scale, s::CartesianPower, ::Type{T}) where {T} =
    _coeftype(𝒮, space(s), T)

_coeftype(𝒮::Scale, s::CartesianProduct, ::Type{T}) where {T} =
    @inbounds promote_type(_coeftype(𝒮, s[1], T), _coeftype(𝒮, Base.tail(s), T))
_coeftype(𝒮::Scale, s::CartesianProduct{<:Tuple{VectorSpace}}, ::Type{T}) where {T} =
    @inbounds _coeftype(𝒮, s[1], T)

function _apply!(c::Sequence{<:CartesianPower}, 𝒮::Scale, a)
    @inbounds for i ∈ 1:nb_cartesian_product(space(c))
        _apply!(component(c, i), 𝒮, component(a, i))
    end
    return c
end
function _apply!(c::Sequence{CartesianProduct{T}}, 𝒮::Scale, a) where {N,T<:NTuple{N,VectorSpace}}
    @inbounds _apply!(component(c, 1), 𝒮, component(a, 1))
    @inbounds _apply!(component(c, 2:N), 𝒮, component(a, 2:N))
    return c
end
function _apply!(c::Sequence{CartesianProduct{T}}, 𝒮::Scale, a) where {T<:Tuple{VectorSpace}}
    @inbounds _apply!(component(c, 1), 𝒮, component(a, 1))
    return c
end

function _findposition_nzind_domain(𝒮::Scale, domain::CartesianSpace, codomain::CartesianSpace)
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

function _findposition_nzind_codomain(𝒮::Scale, domain::CartesianSpace, codomain::CartesianSpace)
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

function _project!(C::LinearOperator{<:CartesianSpace,<:CartesianSpace}, 𝒮::Scale)
    @inbounds for i ∈ 1:nb_cartesian_product(domain(C))
        _project!(component(C, i, i), 𝒮)
    end
    return C
end
