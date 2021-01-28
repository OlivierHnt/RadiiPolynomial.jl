struct Shift{T}
    from :: T
    to :: T
end

(𝒮::Shift)(a) = *(𝒮, a)

# signature needed to resolve ambiguity due to *(b, a::Sequence)
Base.:*(𝒮::Shift, a::Sequence) = shift(a, 𝒮.from, 𝒮.to)
# signature needed to resolve ambiguity due to *(b, A::Operator)
Base.:*(𝒮::Shift, A::Operator) = shift(A, 𝒮.from, 𝒮.to)

# sequence space

function Operator(domain::Fourier{T}, codomain::Fourier{S}, 𝒮::Shift) where {T,S}
    @assert domain.frequency == codomain.frequency
    iωΔτ = im*domain.frequency*one(S)*(𝒮.to-𝒮.from)
    CoefType = float(typeof(iωΔτ))
    A = Operator(domain, codomain, Matrix{CoefType}(undef, dimension(codomain), dimension(domain)))
    A.coefficients .= zero(CoefType)
    @inbounds A[0,0] = one(CoefType)
    @inbounds for j ∈ 1:min(domain.order, codomain.order)
        eiωΔτj = exp(iωΔτ*j)
        A[-j,-j] = conj(eiωΔτj)
        A[j,j] = eiωΔτj
    end
    return A
end

##

# sequence space

function shift(a::Sequence{<:Fourier}, from, to)
    iωΔτ = im*a.space.frequency*(to-from)
    CoefType = float(promote_type(eltype(a), typeof(iωΔτ)))
    c = Sequence(a.space, Vector{CoefType}(undef, dimension(a.space)))
    @inbounds c[0] = a[0]
    @inbounds for j ∈ 1:order(a.space)
        eiωΔτj = exp(iωΔτ*j)
        c[-j] = a[-j]*conj(eiωΔτj)
        c[j] = a[j]*eiωΔτj
    end
    return c
end
