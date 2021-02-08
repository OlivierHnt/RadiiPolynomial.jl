##

function addition_range(s₁::CartesianPowerSpace, s₂::CartesianPowerSpace)
    s₁.dim == s₂.dim || return throw(DimensionMismatch)
    return CartesianPowerSpace(addition_range(s₁.space, s₂.space), s₁.dim)
end
function addition_bar_range(s₁::CartesianPowerSpace, s₂::CartesianPowerSpace)
    s₁.dim == s₂.dim || return throw(DimensionMismatch)
    return CartesianPowerSpace(addition_bar_range(s₁.space, s₂.space), s₁.dim)
end

addition_range(s₁::CartesianProductSpace{<:NTuple{N,VectorSpace}}, s₂::CartesianProductSpace{<:NTuple{N,VectorSpace}}) where {N} =
    CartesianProductSpace(map(addition_range, s₁.spaces, s₂.spaces))
addition_bar_range(s₁::CartesianProductSpace{<:NTuple{N,VectorSpace}}, s₂::CartesianProductSpace{<:NTuple{N,VectorSpace}}) where {N} =
    CartesianProductSpace(map(addition_bar_range, s₁.spaces, s₂.spaces))

##

addition_range(s₁::ParameterSpace, s₂::ParameterSpace) = ParameterSpace()
addition_bar_range(s₁::ParameterSpace, s₂::ParameterSpace) = addition_space(s₁, s₂)

##

addition_range(s₁::Taylor, s₂::Taylor) = ifelse(s₁.order ≤ s₂.order, s₂, s₁)
addition_bar_range(s₁::Taylor, s₂::Taylor) = ifelse(s₁.order ≤ s₂.order, s₁, s₂)
convolution_range(s₁::Taylor, s₂::Taylor) = Taylor(s₁.order + s₂.order)
convolution_bar_range(s₁::Taylor, s₂::Taylor) = addition_bar_range(s₁, s₂)

function addition_range(s₁::Fourier{T}, s₂::Fourier{S}) where {T,S}
    s₁.frequency == s₂.frequency || return throw(DomainError)
    R = promote_type(T, S)
    return Fourier(max(s₁.order, s₂.order), convert(R, s₁.frequency))
end
function addition_bar_range(s₁::Fourier{T}, s₂::Fourier{S}) where {T,S}
    s₁.frequency == s₂.frequency || return throw(DomainError)
    R = promote_type(T, S)
    return Fourier(min(s₁.order, s₂.order), convert(R, s₁.frequency))
end
function convolution_range(s₁::Fourier{T}, s₂::Fourier{S}) where {T,S}
    s₁.frequency == s₂.frequency || return throw(DomainError)
    R = promote_type(T, S)
    return Fourier(s₁.order + s₂.order, convert(R, s₁.frequency))
end
convolution_bar_range(s₁::Fourier{T}, s₂::Fourier{S}) where {T,S} = addition_bar_range(s₁, s₂)

addition_range(s₁::Chebyshev, s₂::Chebyshev) = ifelse(s₁.order ≤ s₂.order, s₂, s₁)
addition_bar_range(s₁::Chebyshev, s₂::Chebyshev) = ifelse(s₁.order ≤ s₂.order, s₁, s₂)
convolution_range(s₁::Chebyshev, s₂::Chebyshev) = Chebyshev(s₁.order + s₂.order)
convolution_bar_range(s₁::Chebyshev, s₂::Chebyshev) = addition_bar_range(s₁, s₂)

addition_range(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    TensorSpace(map(addition_range, s₁.spaces, s₂.spaces))
addition_bar_range(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    TensorSpace(map(addition_bar_range, s₁.spaces, s₂.spaces))
convolution_range(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    TensorSpace(map(convolution_range, s₁.spaces, s₂.spaces))
convolution_bar_range(s₁::TensorSpace{<:NTuple{N,UnivariateSpace}}, s₂::TensorSpace{<:NTuple{N,UnivariateSpace}}) where {N} =
    TensorSpace(map(convolution_bar_range, s₁.spaces, s₂.spaces))
