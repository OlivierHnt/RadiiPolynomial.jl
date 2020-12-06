## differentiation

function differentiate(a::Sequence{Taylor})
    CoefType = eltype(a)
    ord = order(a)
    ord == 0 && return Sequence(a.space, [zero(CoefType)])
    c = Sequence(Taylor(ord-1), Vector{CoefType}(undef, ord))
    @inbounds for i ∈ 1:ord
        c[i-1] = i*a[i]
    end
    return c
end

function differentiate(a::Sequence{<:Fourier})
    iω = im*a.space.frequency
    CoefType = promote_type(eltype(a), typeof(iω))
    c = Sequence(a.space, Vector{CoefType}(undef, length(a.space)))
    @inbounds c[0] = zero(CoefType)
    @inbounds for j ∈ 1:order(a.space)
        iωj = iω*j
        c[j] = iωj*a[j]
        c[-j] = -iωj*a[-j]
    end
    return c
end

function differentiate(a::Sequence{Chebyshev})
    CoefType = eltype(a)
    ord = order(a.order)
    ord == 0 && return Sequence(a.space, [zero(CoefType)])
    c = Sequence(Chebyshev(ord-1), Vector{CoefType}(undef, ord))
    @inbounds c[0] = zero(CoefType)
    @inbounds for i ∈ 1:2:ord
        c[0] += i*a[i]
    end
    @inbounds for i ∈ 1:ord-1
        c[i] = zero(CoefType)
        @inbounds for j ∈ i+1:2:ord
            c[i] += j*a[j]
        end
        c[i] *= 2
    end
    return c
end

## integration

function integrate(a::Sequence{Taylor})
    CoefType = float(eltype(a))
    ord = order(a)
    c = Sequence(Taylor(ord+1), Vector{CoefType}(undef, ord+2))
    @inbounds c[0] = zero(CoefType)
    @inbounds c[1] = convert(CoefType, a[0])
    @inbounds for i ∈ 2:ord+1
        c[i] = a[i-1]/i
    end
    return c
end

function integrate(a::Sequence{<:Fourier})
    @assert iszero(a[0])
    iω⁻¹ = im*inv(a.space.frequency)
    CoefType = promote_type(eltype(a), typeof(iω⁻¹))
    c = Sequence(a.space, Vector{CoefType}(undef, length(a.space)))
    @inbounds c[0] = zero(CoefType)
    @inbounds for j ∈ 1:order(a.order)
        iω⁻¹j⁻¹ = iω⁻¹/j
        c[j] = -iω⁻¹j⁻¹*a[j]
        c[-j] = iω⁻¹j⁻¹*a[-j]
    end
    return c
end

function integrate(a::Sequence{Chebyshev})
    CoefType = float(eltype(a))
    ord = order(a.order)
    if ord == 0
        @inbounds a₀ = convert(CoefType, a[0])
        return Sequence(Chebyshev(1), [a₀, a₀])
    elseif ord == 1
        @inbounds a₀ = convert(CoefType, a[0])
        return Sequence(Chebyshev(2), [a₀ - a[1]/4, a₀, a[1]/4])
    else
        c = Sequence(Chebyshev(ord+1), Vector{CoefType}(undef, ord+2))
        @inbounds c[0] = a[0] - a[1] / 4
        @inbounds for i ∈ 2:2:ord-1
            c[0] -= a[i] / (i^2-1)
            c[0] += a[i+1] / ((i+1)^2-1)
        end
        if iseven(ord)
            @inbounds c[0] -= a[ord] / (ord^2-1)
        end
        @inbounds c[1] = a[0] - a[2] / 2
        @inbounds for i ∈ 2:ord-1
            c[i] = (a[i-1] - a[i+1]) / (2i)
        end
        @inbounds c[ord] = a[ord-1] / (2ord)
        @inbounds c[ord+1] = a[ord] / (2(ord+1))
        return c
    end
end
