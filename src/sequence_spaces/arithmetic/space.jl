##

function Base.:+(s₁::ParameterSpace, s₂::ParameterSpace)
    s₁.dimension == s₂.dimension || return throw(DomainError)
    return s₁
end

Base.:+(s₁::Taylor, s₂::Taylor) = ifelse(s₁.order ≤ s₂.order, s₂, s₁)

function Base.:+(s₁::Fourier{T}, s₂::Fourier{S}) where {T,S}
    s₁.frequency == s₂.frequency || return throw(DomainError)
    NewType = promote_type(T, S)
    s₁.order ≤ s₂.order && return Fourier(s₂.order, convert(NewType, s₂.frequency))
    return Fourier(s₁.order, convert(NewType, s₁.frequency))
end

Base.:+(s₁::Chebyshev, s₂::Chebyshev) = ifelse(s₁.order ≤ s₂.order, s₂, s₁)

Base.:+(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    TensorSpace(map(+, s₁.spaces, s₂.spaces))

Base.:+(s₁::CartesianSpace{<:NTuple{N,SingleSpace}}, s₂::CartesianSpace{<:NTuple{N,SingleSpace}}) where {N} =
    CartesianSpace(map(+, s₁.spaces, s₂.spaces))

##

+̄(s₁::ParameterSpace, s₂::ParameterSpace) = +(s₁, s₂)

+̄(s₁::Taylor, s₂::Taylor) = ifelse(s₁.order ≤ s₂.order, s₁, s₂)

function +̄(s₁::Fourier{T}, s₂::Fourier{S}) where {T,S}
    s₁.frequency == s₂.frequency || return throw(DomainError)
    NewType = promote_type(T, S)
    s₁.order ≤ s₂.order && return Fourier(s₁.order, convert(NewType, s₁.frequency))
    return Fourier(s₂.order, convert(NewType, s₂.frequency))
end

+̄(s₁::Chebyshev, s₂::Chebyshev) = ifelse(s₁.order ≤ s₂.order, s₁, s₂)

+̄(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    TensorSpace(map(+̄, s₁.spaces, s₂.spaces))

+̄(s₁::CartesianSpace{<:NTuple{N,SingleSpace}}, s₂::CartesianSpace{<:NTuple{N,SingleSpace}}) where {N} =
    CartesianSpace(map(+̄, s₁.spaces, s₂.spaces))

##

Base.:*(s₁::Taylor, s₂::Taylor) = Taylor(s₁.order + s₂.order)

function Base.:*(s₁::Fourier{T}, s₂::Fourier{S}) where {T,S}
    s₁.frequency == s₂.frequency || return throw(DomainError)
    NewType = promote_type(T, S)
    return Fourier(s₁.order + s₂.order, convert(NewType, s₁.frequency))
end

Base.:*(s₁::Chebyshev, s₂::Chebyshev) = Chebyshev(s₁.order + s₂.order)

Base.:*(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    TensorSpace(map(*, s₁.spaces, s₂.spaces))

function Base.:^(s::SequenceSpace, n::Int)
    n < 0 && return throw(DomainError(n, "^ is only defined for positive integers"))
    n == 0 && return s
    n == 1 && return s
    n == 2 && return *(s, s)
    # power by squaring
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        s = *(s, s)
    end
    new_s = s
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            s = *(s, s)
        end
        new_s = *(new_s, s)
    end
    return new_s
end

##

*̄(s₁::Taylor, s₂::Taylor) = +̄(s₁, s₂)

*̄(s₁::Fourier{T}, s₂::Fourier{S}) where {T,S} = +̄(s₁, s₂)

*̄(s₁::Chebyshev, s₂::Chebyshev) = +̄(s₁, s₂)

*̄(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    +̄(s₁, s₂)

function ^̄(s::SequenceSpace, n::Int)
    n < 0 && return throw(DomainError(n, "^ is only defined for positive integers"))
    n == 0 && return s
    n == 1 && return s
    n == 2 && return *̄(s, s)
    # power by squaring
    t = trailing_zeros(n) + 1
    n >>= t
    while (t -= 1) > 0
        s = *̄(s, s)
    end
    new_s = s
    while n > 0
        t = trailing_zeros(n) + 1
        n >>= t
        while (t -= 1) ≥ 0
            s = *̄(s, s)
        end
        new_s = *̄(new_s, s)
    end
    return new_s
end
