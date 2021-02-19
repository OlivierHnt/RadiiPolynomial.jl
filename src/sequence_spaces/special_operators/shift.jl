struct Shift{T}
    from :: T
    to :: T
end

## arithmetic

function +̄(𝒮::Shift, A::Operator{Fourier{T},Fourier{S}}) where {T,S}
    frequency(domain) == frequency(codomain) || return throw(DomainError)
    eiωΔτ = cis(frequency(domain)*one(S)*(𝒮.to-𝒮.from))
    CoefType = typeof(eiωΔτ)
    C = Operator(domain, codomain, Matrix{CoefType}(undef, size(A)))
    @. C.coefficients = A.coefficients
    eiωΔτj = one(CoefType)
    @inbounds C[0,0] += eiωΔτj
    @inbounds for j ∈ 1:min(order(domain), order(codomain))
        eiωΔτj *= eiωΔτ
        C[-j,-j] += conj(eiωΔτj)
        C[j,j] += eiωΔτj
    end
    return C
end

+̄(A::Operator{Fourier{T},Fourier{S}}, 𝒮::Shift) where {T,S} = +̄(𝒮, A)

function -̄(𝒮::Shift, A::Operator{Fourier{T},Fourier{S}}) where {T,S}
    frequency(domain) == frequency(codomain) || return throw(DomainError)
    eiωΔτ = cis(frequency(domain)*one(S)*(𝒮.to-𝒮.from))
    CoefType = typeof(eiωΔτ)
    C = Operator(domain, codomain, Matrix{CoefType}(undef, size(A)))
    @. C.coefficients = -A.coefficients
    eiωΔτj = one(CoefType)
    @inbounds C[0,0] += eiωΔτj
    @inbounds for j ∈ 1:min(order(domain), order(codomain))
        eiωΔτj *= eiωΔτ
        C[-j,-j] += conj(eiωΔτj)
        C[j,j] += eiωΔτj
    end
    return C
end

function -̄(A::Operator{Fourier{T},Fourier{S}}, 𝒮::Shift) where {T,S}
    frequency(domain) == frequency(codomain) || return throw(DomainError)
    eiωΔτ = cis(frequency(domain)*one(S)*(𝒮.to-𝒮.from))
    CoefType = typeof(eiωΔτ)
    C = Operator(domain, codomain, Matrix{CoefType}(undef, size(A)))
    @. C.coefficients = A.coefficients
    eiωΔτj = one(CoefType)
    @inbounds C[0,0] -= eiωΔτj
    @inbounds for j ∈ 1:min(order(domain), order(codomain))
        eiωΔτj *= eiωΔτ
        C[-j,-j] -= conj(eiωΔτj)
        C[j,j] -= eiωΔτj
    end
    return C
end

#

function project(𝒮::Shift, domain::Fourier{T}, codomain::Fourier{S}) where {T,S}
    frequency(domain) == frequency(codomain) || return throw(DomainError)
    eiωΔτ = cis(frequency(domain)*one(S)*(𝒮.to-𝒮.from))
    CoefType = typeof(eiωΔτ)
    A = Operator(domain, codomain, zeros(CoefType, dimension(codomain), dimension(domain)))
    eiωΔτj = one(CoefType)
    @inbounds A[0,0] = eiωΔτj
    @inbounds for j ∈ 1:min(order(domain), order(codomain))
        eiωΔτj *= eiωΔτ
        A[-j,-j] = conj(eiωΔτj)
        A[j,j] = eiωΔτj
    end
    return A
end

## action

(𝒮::Shift)(a::Sequence) = *(𝒮, a)
# signature needed to resolve ambiguity due to *(b, a::Sequence)
Base.:*(𝒮::Shift, a::Sequence) = shift(a, 𝒮.from, 𝒮.to)

function shift(a::Sequence{<:Fourier}, from, to)
    eiωΔτ = cis(frequency(a)*(to-from))
    CoefType = promote_type(eltype(a), typeof(eiωΔτ))
    c = Sequence(space(a), Vector{CoefType}(undef, length(a)))
    @. c.coefficients = a.coefficients
    return shift!(c, from, to)
end

function shift!(a::Sequence{<:Fourier}, from, to)
    from == to && return a
    eiωΔτ = cis(frequency(a)*(to-from))
    eiωΔτj = one(eiωΔτ)
    @inbounds for j ∈ 1:order(a)
        eiωΔτj *= eiωΔτ
        a[-j] *= conj(eiωΔτj)
        a[j] *= eiωΔτj
    end
    return a
end
