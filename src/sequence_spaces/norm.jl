## fallback methods for standard norm/opnorm

LinearAlgebra.norm(a::Sequence, p::Real=1) = norm(a.coefficients, p)

LinearAlgebra.opnorm(A::Operator, p::Real=1) = opnorm(A.coefficients, p)

## CARTESIAN SPACE

# standard norm/opnorm

LinearAlgebra.norm(a::Sequence{<:CartesianSpace}, p::Real=2) =
    norm(map(norm, eachcomponent(a)), p)

LinearAlgebra.opnorm(A::Operator{<:CartesianSpace,<:CartesianSpace}, p::Real=2) =
    opnorm(map(opnorm, eachcomponent(A)), p)
LinearAlgebra.opnorm(A::Operator{<:CartesianSpace,<:VectorSpace}, p::Real=2) =
    opnorm(map(opnorm, eachcomponent(A)), p)
LinearAlgebra.opnorm(A::Operator{<:VectorSpace,<:CartesianSpace}, p::Real=2) =
    opnorm(map(opnorm, eachcomponent(A)), p)

## SEQUENCE SPACE

# weighted ℓ¹

checkweight(ν) = ν > 0
checkweight(ν::Tuple) = all(checkweight, ν)

#

norm_weighted_ℓ¹(a::Sequence{<:SequenceSpace}) = norm(a.coefficients, 1)

function norm_weighted_ℓ¹(a::Sequence{<:SequenceSpace}, ν)
    checkweight(ν) || return throw(DomainError(ν, "norm_weighted_ℓ¹ is only defined for strictly positive weights"))
    return _norm_weighted_ℓ¹(a.space, a.coefficients, ν)
end

function _norm_weighted_ℓ¹(space::Taylor, A, ν)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * one(ν)
    @inbounds for i ∈ ord-1:-1:0
        s = s * ν + abs(A[i+1])
    end
    return s
end

function _norm_weighted_ℓ¹(space::Fourier, A, ν)
    ord = order(space)
    @inbounds s = (abs(A[1]) + abs(A[2ord+1])) * one(ν)
    @inbounds for i ∈ ord-1:-1:1
        s = s * ν + abs(A[ord+1-i]) + abs(A[ord+1+i])
    end
    @inbounds s = s * ν + abs(A[ord+1])
    return s
end

function _norm_weighted_ℓ¹(space::Chebyshev, A, ν)
    ord = order(space)
    @inbounds s = abs(A[ord+1]) * one(ν)
    @inbounds for i ∈ ord-1:-1:0
        s = s * ν + abs(A[i+1])
    end
    return s
end

function _norm_weighted_ℓ¹(space::TensorSpace, A, ν)
    A_ = reshape(A, dimensions(space))
    return @inbounds _apply_norm_weighted_ℓ¹(space, A_, ν)[1]
end

_apply_norm_weighted_ℓ¹(space::TensorSpace{<:NTuple{N₁,UnivariateSpace}}, A::AbstractArray{T,N₂}, ν::NTuple{N₁,Any}) where {N₁,T,N₂} =
    @inbounds _norm_weighted_ℓ¹(space[1], Val(N₂-N₁+1), _apply_norm_weighted_ℓ¹(Base.tail(space), A, Base.tail(ν)), ν[1])

_apply_norm_weighted_ℓ¹(space::TensorSpace{<:NTuple{2,UnivariateSpace}}, A::AbstractArray{T,N}, ν::NTuple{2,Any}) where {T,N} =
    @inbounds _norm_weighted_ℓ¹(space[1], Val(N-1), _norm_weighted_ℓ¹(space[2], Val(N), A, ν[2]), ν[1])

function _norm_weighted_ℓ¹(space::Taylor, ::Val{D}, A, ν) where {D}
    ord = order(space)
    @inbounds s = abs.(selectdim(A, D, ord+1:ord+1)) .* one(ν)
    @inbounds for i ∈ ord-1:-1:0
        Aᵢ = selectdim(A, D, i+1:i+1)
        @. s = s * ν + abs(Aᵢ)
    end
    return s
end

function _norm_weighted_ℓ¹(space::Fourier, ::Val{D}, A, ν) where {D}
    ord = order(space)
    @inbounds s = (abs.(selectdim(A, D, 1:1)) .+ abs.(selectdim(A, D, 2ord+1:2ord+1))) .* one(ν)
    @inbounds for i ∈ ord-1:-1:1
        Aᵢ = selectdim(A, D, ord+1+i:ord+1+i)
        A₋ᵢ = selectdim(A, D, ord+1-i:ord+1-i)
        @. s = s * ν + abs(A₋ᵢ) + abs(Aᵢ)
    end
    @inbounds A₀ = selectdim(A, D, ord+1:ord+1)
    @. s = s * ν + abs(A₀)
    return s
end

function _norm_weighted_ℓ¹(space::Chebyshev, ::Val{D}, A, ν) where {D}
    ord = order(space)
    @inbounds s = abs.(selectdim(A, D, ord+1:ord+1)) .* one(ν)
    @inbounds for i ∈ ord-1:-1:0
        Aᵢ = selectdim(A, D, i+1:i+1)
        @. s = s * ν + abs(Aᵢ)
    end
    return s
end

#

opnorm_weighted_ℓ¹(A::Operator{<:SequenceSpace,<:SequenceSpace}) = opnorm(A.coefficients, 1)

function opnorm_weighted_ℓ¹(A::Operator{<:SequenceSpace,<:SequenceSpace}, μ, ν)
    checkweight(μ) && checkweight(ν) || throw(DomainError((μ, ν), "norm_weighted_ℓ¹ is only defined for strictly positive weights"))
    return _functional_opnorm_weighted_ℓ¹(A.domain, map(Aᵢ -> _norm_weighted_ℓ¹(A.codomain, Aᵢ, ν), eachcol(A.coefficients)), μ)
end

function _functional_opnorm_weighted_ℓ¹(domain::Taylor, A, μ)
    μ⁻¹ = inv(μ)
    μ⁻ⁱ = one(μ⁻¹)
    @inbounds s = abs(A[1]) * μ⁻ⁱ
    @inbounds for i ∈ 1:order(domain)
        μ⁻ⁱ *= μ⁻¹
        s = max(s, abs(A[i+1]) * μ⁻ⁱ)
    end
    return s
end

function _functional_opnorm_weighted_ℓ¹(domain::Fourier, A, μ)
    μ⁻¹ = inv(μ)
    μ⁻ⁱ = one(μ⁻¹)
    ord = order(domain)
    @inbounds s = abs(A[ord+1]) * μ⁻ⁱ
    @inbounds for i ∈ 1:ord
        μ⁻ⁱ *= μ⁻¹
        s = max(s, abs(A[ord+1+i]) * μ⁻ⁱ, abs(A[ord+1-i]) * μ⁻ⁱ)
    end
    return s
end

function _functional_opnorm_weighted_ℓ¹(domain::Chebyshev, A, μ)
    μ⁻¹ = inv(μ)
    μ⁻ⁱ = one(μ⁻¹)
    @inbounds s = abs(A[1]) * μ⁻ⁱ
    @inbounds for i ∈ 1:order(domain)
        μ⁻ⁱ *= μ⁻¹
        s = max(s, abs(A[i+1]) * μ⁻ⁱ)
    end
    return s
end

function _functional_opnorm_weighted_ℓ¹(domain::TensorSpace, A, μ)
    A_ = reshape(A, dimensions(domain))
    return @inbounds _apply_functional_opnorm_weighted_ℓ¹(domain, A_, μ)[1]
end

_apply_functional_opnorm_weighted_ℓ¹(space::TensorSpace{<:NTuple{N₁,UnivariateSpace}}, A::AbstractArray{T,N₂}, μ::NTuple{N₁,Real}) where {N₁,T,N₂} =
    @inbounds _functional_opnorm_weighted_ℓ¹(space[1], Val(N₂-N₁+1), _apply_functional_opnorm_weighted_ℓ¹(Base.tail(space), A, Base.tail(μ)), μ[1])

_apply_functional_opnorm_weighted_ℓ¹(space::TensorSpace{<:NTuple{2,UnivariateSpace}}, A::AbstractArray{T,N}, μ::NTuple{2,Real}) where {T,N} =
    @inbounds _functional_opnorm_weighted_ℓ¹(space[1], Val(N-1), _functional_opnorm_weighted_ℓ¹(space[2], Val(N), A, μ[2]), μ[1])

function _functional_opnorm_weighted_ℓ¹(space::Taylor, ::Val{D}, A, μ) where {D}
    μ⁻¹ = inv(μ)
    μ⁻ⁱ = one(μ⁻¹)
    @inbounds s = abs.(selectdim(A, D, 1:1)) .* μ⁻ⁱ
    @inbounds for i ∈ 1:order(space)
        μ⁻ⁱ *= μ⁻¹
        Aᵢ = selectdim(A, D, i+1:i+1)
        @. s = max(s, abs(Aᵢ) * μ⁻ⁱ)
    end
    return s
end

function _functional_opnorm_weighted_ℓ¹(space::Fourier, ::Val{D}, A, μ) where {D}
    μ⁻¹ = inv(μ)
    μ⁻ⁱ = one(μ⁻¹)
    ord = order(space)
    @inbounds s = abs.(selectdim(A, D, ord+1:ord+1)) .* μ⁻ⁱ
    @inbounds for i ∈ 1:ord
        μ⁻ⁱ *= μ⁻¹
        Aᵢ = selectdim(A, D, ord+1+i:ord+1+i)
        A₋ᵢ = selectdim(A, D, ord+1-i:ord+1-i)
        @. s = max(s, abs(A₋ᵢ) * μ⁻ⁱ, abs(Aᵢ) * μ⁻ⁱ)
    end
    return s
end

function _functional_opnorm_weighted_ℓ¹(space::Chebyshev, ::Val{D}, A, μ) where {D}
    μ⁻¹ = inv(μ)
    μ⁻ⁱ = one(μ⁻¹)
    @inbounds s = abs.(selectdim(A, D, 1:1)) .* μ⁻ⁱ
    @inbounds for i ∈ 1:order(space)
        μ⁻ⁱ *= μ⁻¹
        Aᵢ = selectdim(A, D, i+1:i+1)
        @. s = max(s, abs(Aᵢ) * μ⁻ⁱ)
    end
    return s
end
