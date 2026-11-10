# The errors of a `Nonlinearity` on an `InfiniteSequence` against an independent reference: f ∘ x is sampled
# in BigFloat on the unit circle, every space being read as a Laurent series (a Chebyshev coefficient sits
# at ±k), and its coefficients are recovered by a direct DFT

_ref_laurent(::Taylor, k) = (k,)
_ref_laurent(::Fourier, k) = (k,)
_ref_laurent(::Chebyshev, k) = k == 0 ? (0,) : (k, -k)

_ref_index(s::Taylor, ℓ) = 0 ≤ ℓ ≤ order(s) ? ℓ : nothing
_ref_index(s::Fourier, ℓ) = abs(ℓ) ≤ order(s) ? ℓ : nothing
_ref_index(s::Chebyshev, ℓ) = abs(ℓ) ≤ order(s) ? abs(ℓ) : nothing

_ref_spaces(s::TensorSpace) = spaces(s)
_ref_spaces(s::BaseSpace) = (s,)

# Laurent coefficients of f ∘ (x₁ + x₂ + ⋯) on the box |ℓᵢ| ≤ Kᵢ, from Mᵢ samples per direction, at a
# precision covering the largest weight ν^K
function _ref_coefficients(f, xs::Tuple, ν::NTuple{D}, K::NTuple{D,Int}, M::NTuple{D,Int}) where {D}
    return setprecision(BigFloat, 128 + ceil(Int, sum(K .* log2.(ν)))) do
        L = ntuple(i -> maximum(x -> order(_ref_spaces(space(x))[i]), xs), Val(D))
        A = zeros(Complex{BigFloat}, map(Lᵢ -> 2Lᵢ + 1, L))
        for x ∈ xs, k ∈ indices(space(x))
            c = Complex{BigFloat}(x[k])
            iszero(c) && continue
            for ℓ ∈ Iterators.product(map(_ref_laurent, _ref_spaces(space(x)), Tuple(k))...)
                A[(ℓ .+ L .+ 1)...] += c
            end
        end
        # the samples e^{iℓθⱼ} are read off the M roots of unity
        ω = map(m -> [cispi(2 * big(j) / m) for j ∈ 0:m-1], M)
        samples(n, ωᵢ) = [ωᵢ[mod(ℓ * j, length(ωᵢ)) + 1] for ℓ ∈ -n:n, j ∈ 0:length(ωᵢ)-1]
        E = map(samples, L, ω)
        Ē = map(conj ∘ samples, K, ω)
        if D == 1
            return Ē[1] * f.(transpose(E[1]) * A) / M[1]
        else
            return Ē[1] * f.(transpose(E[1]) * A * E[2]) * transpose(Ē[2]) / (M[1] * M[2])
        end
    end
end

# distance from a point to an interval, and to a box in the complex plane
_ref_gap(t, x::Interval) = max(zero(t), BigFloat(inf(x)) - t, t - BigFloat(sup(x)))
_ref_dist(z, x::Interval) = hypot(_ref_gap(real(z), x), abs(imag(z)))
_ref_dist(z, x::Complex{<:Interval}) = hypot(_ref_gap(real(z), real(x)), _ref_gap(imag(z), imag(x)))

function _ref_check(r::InfiniteSequence, C::AbstractArray, ν::Tuple)
    c = sequence(r)
    ss = _ref_spaces(space(c))
    K = map(n -> (n - 1) ÷ 2, size(C))
    finite = tail = shell = zero(BigFloat)
    for ℓ ∈ Iterators.product(map(Kᵢ -> -Kᵢ:Kᵢ, K)...)
        w = prod(BigFloat.(ν) .^ abs.(ℓ))
        ref = C[(ℓ .+ K .+ 1)...]
        k = map(_ref_index, ss, ℓ)
        if any(isnothing, k)
            tail += w * abs(ref)
        else
            finite += w * _ref_dist(ref, c[length(k) == 1 ? k[1] : k])
        end
        any(abs.(ℓ) .== K) && (shell += w * abs(ref))
    end
    # the reference is accurate to about 1e-36, and the box holds it up to its outer shell
    @test shell < 1e-28
    @test isfinite(sup(total_error(r)))
    @test finite ≤ sup(finite_error(r)) + 1e-26
    @test tail ≤ sup(tail_error(r)) + 1e-26
    @test finite + tail ≤ sup(total_error(r)) + 1e-26
end

_oforder_like(s::Taylor, n) = Taylor(n)
_oforder_like(s::Fourier, n) = Fourier(n, frequency(s))
_oforder_like(s::Chebyshev, n) = Chebyshev(n)

_ref_chebyshev_interpolant(f, N) =
    real.(coefficients(RadiiPolynomial.to_coef(reverse(f.([-cospi(j / N) for j ∈ 0:N])), Chebyshev(N))))

@testset "Nonlinearity: soundness against a BigFloat reference" begin
    pow = Nonlinearity(x -> x ^ 1.5, Complex{Interval{Float64}}[], interval(-Inf, 0))
    pole = Nonlinearity(x -> inv(3 - x), [interval(3.0)], emptyinterval())
    # not real on the real axis: the coefficients of a real input's image are complex
    rotation = Nonlinearity(x -> exp(im * x), Complex{Interval{Float64}}[], emptyinterval())

    @testset "base spaces" begin
        X(ν) = Ell1(GeometricWeight(interval(ν)))
        cases = (
            # (nonlinearity, input, rate, codomains)
            (Nonlinearity(exp), Sequence(Taylor(10), [0.5 ; 0.8 ; zeros(9)]), 1.2, (Taylor(10), Taylor(20))),
            (pow, Sequence(Taylor(6), [2.0, 0.5, 0.1, 0, 0, 0, 0]), 1.2, (Taylor(6), Taylor(12))),
            (Nonlinearity(sin), Sequence(Fourier(3, 1.0), [0, 0.1, 0.3, 0, 0.3, 0.1, 0]), 1.2, (Fourier(3, 1.0), Fourier(10, 1.0))),
            (Nonlinearity(exp), Sequence(Fourier(2, 1.0), [0, 0.2im, 0.1 + 0.2im, 0.3, 0]), 1.3, (Fourier(8, 1.0),)),
            (Nonlinearity(cosh), Sequence(Chebyshev(10), [1.0 ; 0.15 ; zeros(9)]), 1.2, (Chebyshev(10), Chebyshev(4), Chebyshev(30))),
            (pole, Sequence(Chebyshev(8), [0.0 ; 0.25 ; zeros(7)]), 1.2, (Chebyshev(8), Chebyshev(16))),
            (Nonlinearity(exp), Sequence(Chebyshev(16), _ref_chebyshev_interpolant(t -> log(2 + t / 2), 16)), 1.2, (Chebyshev(16), Chebyshev(32))),
            (Nonlinearity(exp), Sequence(Chebyshev(10), [0.0 ; 0.25 ; zeros(9)]), 3.0, (Chebyshev(10), Chebyshev(20))),
            (rotation, Sequence(Chebyshev(10), [0.2 ; 0.5 ; zeros(9)]), 1.2, (Chebyshev(10), Chebyshev(20))),
            (rotation, Sequence(Fourier(3, 1.0), [0, 0.1, 0.3, 0, 0.3, 0.1, 0]), 1.2, (Fourier(3, 1.0),)),
            (rotation, Sequence(Taylor(6), [0.2 ; 0.5 ; zeros(5)]), 1.2, (Taylor(6),)),
        )
        for (nl, x, ν, codomains) ∈ cases
            C = _ref_coefficients(nl.f, (x,), (ν,), (150,), (600,))
            for codomain ∈ codomains
                _ref_check(nl(InfiniteSequence(interval.(x), X(ν)); codomain = codomain), C, (ν,))
            end
            _ref_check(nl(InfiniteSequence(interval.(x), X(ν))), C, (ν,)) # order resolved adaptively
        end
    end

    @testset "tensor spaces" begin
        X(ν₁, ν₂) = Ell1((GeometricWeight(interval(ν₁)), GeometricWeight(interval(ν₂))))
        function input(s, entries)
            x = Sequence(s, zeros(dimension(s)))
            foreach(((k, v),) -> x[k] = v, entries)
            return x
        end
        noise_floor = _ref_chebyshev_interpolant(t -> log(2 + t / 2), 16)
        cases = (
            # the tail lies in the slab k > N, l ≤ N
            (Nonlinearity(exp), input(Taylor(5) ⊗ Taylor(5), ((1, 0) => 2.0, (0, 1) => 0.01)), (1.2, 1.2),
                (Taylor(5) ⊗ Taylor(5),), (40, 20), (160, 80)),
            (Nonlinearity(exp), input(Chebyshev(6) ⊗ Fourier(3, 1.0), ((0, 0) => 0.5, (1, 0) => 0.1, (0, 1) => 0.15, (0, -1) => 0.15)), (1.2, 1.5),
                (Chebyshev(6) ⊗ Fourier(3, 1.0), Chebyshev(12) ⊗ Fourier(6, 1.0)), (30, 30), (120, 120)),
            (pole, input(Fourier(2, 1.0) ⊗ Fourier(2, 1.0), ((1, 0) => 0.3, (-1, 0) => 0.3, (0, 1) => 0.3, (0, -1) => 0.3)), (1.1, 1.3),
                (Fourier(2, 1.0) ⊗ Fourier(2, 1.0), Fourier(6, 1.0) ⊗ Fourier(6, 1.0)), (40, 40), (160, 160)),
            (Nonlinearity(exp), input(Chebyshev(16) ⊗ Chebyshev(4), ((0, 1) => 0.05, ((k, 0) => noise_floor[k+1] for k ∈ 0:16)...)), (1.2, 1.2),
                (Chebyshev(16) ⊗ Chebyshev(4),), (48, 24), (192, 96)),
        )
        for (nl, x, ν, codomains, K, M) ∈ cases
            C = _ref_coefficients(nl.f, (x,), ν, K, M)
            for codomain ∈ codomains
                _ref_check(nl(InfiniteSequence(interval.(x), X(ν...)); codomain = codomain), C, ν)
            end
            _ref_check(nl(InfiniteSequence(interval.(x), X(ν...))), C, ν) # order resolved adaptively
        end
    end

    @testset "input error: every input within ε" begin
        # the input is the midpoint plus h of norm 0.9ε, on the constant, on a mode, and beyond the order
        X = Ell1(GeometricWeight(interval(1.2)))
        cases = (
            (Nonlinearity(exp), Sequence(Chebyshev(10), [1.0 ; 0.15 ; zeros(9)]), 1e-6,
                (0 => 1.0, 3 => 1 / (2 * 1.2^3), 20 => 1 / (2 * 1.2^20)), Chebyshev(20)),
            (pow, Sequence(Taylor(8), [2.0 ; 0.5 ; zeros(7)]), 1e-8,
                (0 => 1.0, 5 => 1 / 1.2^5, 20 => 1 / 1.2^20), Taylor(20)),
        )
        for (nl, x, ε, perturbations, s) ∈ cases
            r = nl(InfiniteSequence(interval.(x), X; total_error = interval(ε)); codomain = space(x))
            r_adaptive = nl(InfiniteSequence(interval.(x), X; total_error = interval(ε)))
            for (k, scale) ∈ perturbations
                h = Sequence(s, zeros(dimension(s)))
                h[k] = 0.9 * ε * scale
                C = _ref_coefficients(nl.f, (x, h), (1.2,), (150,), (600,))
                _ref_check(r, C, (1.2,))
                _ref_check(r_adaptive, C, (1.2,))
            end
        end

        s = Chebyshev(6) ⊗ Fourier(3, 1.0)
        x = Sequence(s, zeros(dimension(s)))
        x[(0, 0)], x[(1, 0)], x[(0, 1)], x[(0, -1)] = 0.5, 0.1, 0.15, 0.15
        X₂ = Ell1((GeometricWeight(interval(1.2)), GeometricWeight(interval(1.5))))
        r = exp(InfiniteSequence(interval.(x), X₂; total_error = interval(1e-6)))
        h = Sequence(s, zeros(dimension(s)))
        h[(2, 1)] = 0.9e-6 / (2 * 1.2^2 * 1.5)
        _ref_check(r, _ref_coefficients(exp, (x, h), (1.2, 1.5), (30, 30), (120, 120)), (1.2, 1.5))
    end

    @testset "inv, sqrt, cbrt and division" begin
        X = Ell1(GeometricWeight(interval(1.2)))
        operations = ((inv, inv), (sqrt, sqrt), (cbrt, z -> z ^ (big(1) / 3)), (x -> x / (x + interval(1.0)), z -> z / (z + 1)))
        inputs = (Sequence(Chebyshev(1), [2.0, 0.25]), Sequence(Taylor(3), [3.0, 0.5, 0.1, 0.0]), Sequence(Fourier(1, 1.0), [0.2, 2.0, 0.2]))
        for (i, (F, ref)) ∈ enumerate(operations), x ∈ inputs
            C = _ref_coefficients(ref, (x,), (1.2,), (150,), (600,))
            _ref_check(F(InfiniteSequence(interval.(x), X)), C, (1.2,))
            # the division takes no codomain
            i ≤ 3 && _ref_check(F(InfiniteSequence(interval.(x), X); codomain = _oforder_like(space(x), 6)), C, (1.2,))
        end
    end

    @testset "input intervals: every choice within them" begin
        X = Ell1(GeometricWeight(interval(1.2)))
        a = Sequence(Chebyshev(10), [interval(1 - 1e-6, 1 + 1e-6) ; interval(0.15) ; zeros(Interval{Float64}, 9)])
        r = sin(InfiniteSequence(a, X))
        for x ∈ (Sequence(space(a), inf.(coefficients(a))), Sequence(space(a), sup.(coefficients(a))))
            _ref_check(r, _ref_coefficients(sin, (x,), (1.2,), (150,), (600,)), (1.2,))
        end
    end
end
