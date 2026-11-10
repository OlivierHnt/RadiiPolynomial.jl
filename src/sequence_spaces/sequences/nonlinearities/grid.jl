_maybe_interval(::Type, a) = a
_maybe_interval(::Type{<:RealOrComplexI}, a) = interval(a)

_isguaranteed(a::InfiniteSequence) = all(isguaranteed, sequence(a)) & isguaranteed(sequence_norm(a)) & isguaranteed(finite_error(a)) & isguaranteed(tail_error(a)) & isguaranteed(total_error(a))

_mag(x::Union{Interval,Complex{<:Interval}}) = mag(x)
_mag(x) = abs(x)

_oforder(::Taylor, n::Int) = Taylor(n)
_oforder(::Chebyshev, n::Int) = Chebyshev(n)
_oforder(s::Fourier, n::Int) = Fourier(n, frequency(s))
_oforder(s::TensorSpace, n::Tuple) = TensorSpace(map(_oforder, spaces(s), n))
_oforder(s::BaseSpace, n::Tuple{Int}) = _oforder(s, n[1])
_oforder(s::SymmetricSpace, n) = SymmetricSpace(_oforder(desymmetrize(s), n), symmetry(s))

# grid of the contour constant and of the analyticity sweep: doubled, to halve the arcs enclosing f ∘ a
_oversampled_grid_size(s::SequenceSpace) = fast_grid_size(2 .* grid_size(s), s)
_oversampled_fft_size(s::SequenceSpace) = _full_fft_size(_oversampled_grid_size(s), s)

# grid of the coefficients of f ∘ a: that of `codomain`, or of an input if larger
_coef_grid_size(codomain::SequenceSpace, s::SequenceSpace...) =
    fast_grid_size(max.(grid_size(codomain), map(grid_size, s)...), codomain)
_coef_fft_size(codomain::SequenceSpace, s::SequenceSpace...) = _full_fft_size(_coef_grid_size(codomain, s...), codomain)

# f at the grid values of the inputs, transformed back to `codomain`
function _grid_coefficients(f, codomain::SequenceSpace, ::Type{T}, a::Sequence...; grid::SequenceSpace = codomain) where {T}
    sz = _coef_grid_size(grid, map(space, a)...)
    return _call_to_coef!(f.(map(aᵢ -> to_grid(aᵢ, sz), a)...), codomain, T)
end

# real coefficients only when f maps real arguments to real values
_image_eltype(f, ::Type{T}) where {T} = ifelse(Base.promote_op(f, T) <: Real, T, complex(T))

# f ∘ a on spaces of doubling order until, in each direction, the coefficients past half the order are
# below the noise floor, or small and not decreasing
function _resolve(evaluate, s::SequenceSpace)
    N = max.(Tuple(order(s)), 8)
    tails_prev = map(_ -> Inf, N)
    while true
        c = evaluate(_oforder(s, N))
        tails = _tails(c)
        small = sqrt(eps(one(_magnitude(c)))) * _magnitude(c)
        done = (tails .≤ _noise_floor(c)) .| ((tails .> tails_prev ./ 4) .& (tails .≤ small)) .| (N .≥ 2^12)
        (all(done) || prod(N .+ 1) > 2^18) && return c
        N = ifelse.(done, N, 2 .* N)
        tails_prev = tails
    end
end

# largest coefficient past half the order, per direction
function _tails(c::Sequence)
    N = Tuple(order(space(c)))
    m(k) = abs(mid(c[k]))
    return ntuple(i -> maximum(k -> ifelse(abs(Tuple(k)[i]) > N[i] ÷ 2, m(k), zero(m(k))), indices(space(c))), length(N))
end

# space of f ∘ a: the full lattice, or the symmetric space of a group
_output_space(s::SequenceSpace, ::Nothing) = desymmetrize(s)
_output_space(s::SequenceSpace, G) = SymmetricSpace(desymmetrize(s), G)

# f ∘ a in `codomain`, or with its order resolved
function _apply(f, a::Sequence{<:SequenceSpace}, codomain, symmetry)
    (codomain === nothing) | (symmetry === nothing) || return throw(ArgumentError("a codomain fixes the symmetry: give either a codomain or a symmetry"))
    _isconstant(a) && return _at_value(f, a, something(codomain, _output_space(space(a), symmetry)))
    T = _image_eltype(f, eltype(a))
    evaluate(s) = _grid_coefficients(f, s, T, a)
    return codomain === nothing ? _noise_trim(_resolve(evaluate, _output_space(space(a), symmetry)), NoiseFloor()) : evaluate(codomain)
end

# a constant input gives the constant f(a₀)
function _isconstant(a::Sequence)
    s = space(a)
    idx = _findindex_constant(s)
    return all(k -> ifelse(k == idx, true, _safe_iszero(a[k])), indices(s))
end
_isexactconstant(a::InfiniteSequence) = _isconstant(sequence(a)) & _safe_iszero(total_error(a))

function _at_value(f, a::Sequence, s::SequenceSpace)
    v = f(a[_findindex_constant(space(a))])
    c = zeros(promote_type(eltype(a), typeof(v)), s)
    c[_findindex_constant(s)] = v
    return c
end
_at_value(f, a::InfiniteSequence, codomain) =
    InfiniteSequence(_at_value(f, sequence(a), something(codomain, _output_space(space(a), nothing))), banachspace(a))
