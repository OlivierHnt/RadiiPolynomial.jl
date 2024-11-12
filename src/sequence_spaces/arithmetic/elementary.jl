_image_trunc(::typeof(inv), s::SequenceSpace) = s

function Base.inv(a::Sequence{<:SequenceSpace})
    space_c = _image_trunc(inv, space(a))
    A = fft(a, fft_size(space_c))
    C = inv.(A)
    c = _call_ifft!(C, space_c, eltype(a))
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
    return c
end
function Base.:/(a::Number, b::Sequence{<:SequenceSpace})
    space_c = _image_trunc(inv, space(b))
    B = fft(b, fft_size(space_c))
    C = a ./ B
    c = _call_ifft!(C, space_c, promote_type(typeof(a), eltype(b)))
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
    return c
end

#

_image_trunc(::typeof(cbrt), s::SequenceSpace) = s

function Base.cbrt(a::Sequence{<:SequenceSpace})
    space_c = _image_trunc(cbrt, space(a))
    A = fft(a, fft_size(space_c))
    C = A .^ ExactReal(1//3)
    c = _call_ifft!(C, space_c, eltype(a))
    return c
end

#

Base.abs(a::Sequence{<:SequenceSpace}) = sqrt(a^2)
Base.abs2(a::Sequence{<:SequenceSpace}) = a^2

#

_image_trunc(::typeof(^), s::SequenceSpace) = s

function Base.:^(a::Sequence{<:SequenceSpace}, p::Real)
    space_c = _image_trunc(^, space(a))
    A = fft(a, fft_size(space_c))
    C = A .^ p
    c = _call_ifft!(C, space_c, eltype(a))
    ν̄ = 0.5 .* (max.(_geometric_rate(space(a), coefficients(a))[1], _geometric_rate(space(c), coefficients(c))[1]) .- 1.0) .+ 1.0
    _resolve_saturation!(x -> ^(x, p), c, a, ν̄)
    return c
end

#

for f ∈ (:exp, :cos, :sin, :cosh, :sinh)
    @eval begin
        _image_trunc(::typeof($f), s::SequenceSpace) = s

        function Base.$f(a::Sequence{<:SequenceSpace})
            _isconstant(a) && return _at_value($f, a)
            space_c = _image_trunc($f, space(a))
            A = fft(a, fft_size(space_c))
            C = $f.(A)
            c = _call_ifft!(C, space_c, eltype(a))
            ν̄ = 0.5 .* (max.(_geometric_rate(space(a), coefficients(a))[1], _geometric_rate(space(c), coefficients(c))[1]) .- 1.0) .+ 1.0
            _resolve_saturation!($f, c, a, ν̄)
            return c
        end

        function _at_value(::typeof($f), a)
            c = one(a)
            idx = _findindex_constant(space(a))
            c[idx] = $f(a[idx])
            return c
        end
    end
end

#

function _isconstant(a::Sequence)
    s = space(a)
    idx = _findindex_constant(s) # throws for `SinFourier`
    return all(k -> ifelse(k == idx, true, iszero(a[k])), indices(s))
end



function _resolve_saturation!(f, c, a, ν)
    ν⁻¹ = inv(ν)
    C = max(_contour(f, a, ν), _contour(f, a, ν⁻¹))
    CoefType = eltype(c)
    min_ord = order(c)
    for k ∈ indices(space(c))
        if abs(c[k]) > C / ν ^ abs(k)
            min_ord = min(min_ord, abs(k))
            c[k] = zero(CoefType)
        end
    end
    return c, min_ord
end

function _resolve_saturation!(f, c, a, ν::NTuple{N}) where {N}
    ν⁻¹ = inv.(ν)
    _tuple_ = tuple(ν, ν⁻¹)
    _mix_ = Iterators.product(ntuple(i -> getindex.(_tuple_, i), Val(N))...)
    C = maximum(μ -> _contour(f, a, μ), _mix_)
    CoefType = eltype(c)
    min_ord = order(c)
    for k ∈ indices(space(c))
        if abs(c[k]) > C / prod(ν .^ abs.(k))
            min_ord = min.(min_ord, abs.(k))
            c[k] = zero(CoefType)
        end
    end
    return c, min_ord
end



function _contour(f, a, ν)
    # N_fft = min(fft_size(space(a)), prevpow(2, log( ifelse(ν < 1, floatmin(ν), floatmax(ν)) ) / log(ν))) # maybe there is a better N_fft value to consider
    N_fft = fft_size(space(a))

    CoefType = complex(IntervalArithmetic.numtype(eltype(a)))
    grid_a_δ = zeros(CoefType, N_fft)

    A = coefficients(a)
    view(grid_a_δ, eachindex(A)) .= mid.(A)
    _preprocess!(grid_a_δ, space(a))
    _boxes!(grid_a_δ, ν)

    _fft_pow2!(grid_a_δ)
    contour_integral = sum(abs ∘ f, grid_a_δ)

    return contour_integral / N_fft
end

function _contour(f, a, ν::Tuple)
    # N_fft = min.(fft_size(space(a)), prevpow.(2, log.( ifelse.(mid.(ν) .< 1, floatmin.(ν), floatmax.(ν)) ) ./ log.(ν)))
    N_fft = fft_size(space(a))

    CoefType = complex(IntervalArithmetic.numtype(eltype(a)))
    grid_a_δ = zeros(CoefType, N_fft)

    A = _no_alloc_reshape(coefficients(a), dimensions(space(a)))
    view(grid_a_δ, axes(A)...) .= mid.(A)
    _apply_preprocess!(grid_a_δ, space(a))
    _apply_boxes!(grid_a_δ, ν)

    _fft_pow2!(grid_a_δ)
    contour_integral = sum(abs ∘ f, grid_a_δ)

    return contour_integral / prod(N_fft)
end



_apply_boxes!(C::AbstractArray{S,N₁}, ν::NTuple{N₂,Number}) where {S,N₁,N₂} =
    @inbounds _boxes!(_apply_boxes!(C, Base.tail(ν)), ν[1], Val(N₁-N₂+1))

_apply_boxes!(C::AbstractArray{S,N}, ν::Tuple{Number}) where {S,N} =
    @inbounds _boxes!(C, ν[1], Val(N))

function _boxes!(C, ν)
    len = length(C)
    for k ∈ 1:len÷2-1
        C[k+1]     *= ν ^ ExactReal(-k)
        C[len+1-k] *= ν ^ ExactReal( k)
    end
    return C
end

function _boxes!(C, ν, ::Val{D}) where {D}
    len = size(C, D)
    for k ∈ 1:len÷2-1
        selectdim(C, D, k+1)     .*= ν ^ ExactReal(-k)
        selectdim(C, D, len+1-k) .*= ν ^ ExactReal( k)
    end
    return C
end
