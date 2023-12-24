_image_trunc(::typeof(inv), s::SequenceSpace) = s

function Base.inv(a::Sequence{<:SequenceSpace})
    space_c = _image_trunc(inv, space(a))
    A = fft(a, fft_size(space_c))
    C = inv.(A)
    c = _call_ifft!(C, space_c, eltype(a))
    X = Ell1(weight(c))
    banach_rounding!(c, norm(c, X), X)
    return c
end

_image_trunc(::typeof(/), s₁::TensorSpace{<:NTuple{N,BaseSpace}}, s₂::TensorSpace{<:NTuple{N,BaseSpace}}) where {N} =
    TensorSpace(map((s₁ᵢ, s₂ᵢ) -> _image_trunc(/, s₁ᵢ, s₂ᵢ), spaces(s₁), spaces(s₂)))

_image_trunc(::typeof(/), s₁::Taylor, s₂::Taylor) = union(s₁, s₂)

_image_trunc(::typeof(/), s₁::Fourier, s₂::Fourier) = union(s₁, s₂)

_image_trunc(::typeof(/), s₁::Chebyshev, s₂::Chebyshev) = union(s₁, s₂)

function Base.:/(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace})
    space_c = _image_trunc(/, space(a), space(b))
    A = fft(a, fft_size(space_c))
    B = fft(b, fft_size(space_c))
    C = A ./ B
    c = _call_ifft!(C, space_c, promote_type(eltype(a), eltype(b)))
    X = Ell1(weight(c))
    banach_rounding!(c, norm(c, X), X)
    return c
end
function Base.:/(a::Number, b::Sequence{<:SequenceSpace})
    space_c = _image_trunc(inv, space(b))
    B = fft(b, fft_size(space_c))
    C = a ./ B
    c = _call_ifft!(C, space_c, promote_type(typeof(a), eltype(b)))
    X = Ell1(weight(c))
    banach_rounding!(c, norm(c, X), X)
    return c
end

_image_trunc(::typeof(\), s₁::SequenceSpace, s₂::SequenceSpace) = image(/, s₁, s₂)

Base.:\(a::Sequence{<:SequenceSpace}, b::Sequence{<:SequenceSpace}) = b / a
Base.:\(a::Sequence{<:SequenceSpace}, b::Number) = b / a

#

_image_trunc(::typeof(sqrt), s::SequenceSpace) = s

function Base.sqrt(a::Sequence{<:SequenceSpace})
    space_c = _image_trunc(sqrt, space(a))
    A = fft(a, fft_size(space_c))
    C = sqrt.(A)
    c = _call_ifft!(C, space_c, eltype(a))
    _banach_rounding!(c)
    return c
end

#

Base.abs(a::Sequence{<:SequenceSpace}) = sqrt(a^2)
Base.abs2(a::Sequence{<:SequenceSpace}) = a^2