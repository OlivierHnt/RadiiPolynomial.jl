# sequence on `s` vanishing outside of the given `index => value` entries
function _el_seq(s, entries::Pair...)
    a = zeros(s)
    foreach(((k, v),) -> a[k] = v, entries)
    return a
end

@testset "Elementary nonlinearities" begin

    𝒯, ℱ, 𝒞 = Taylor(4), Fourier(2, 1.0), Chebyshev(4)
    ℓ¹ν(ν...) = Ell1(length(ν) == 1 ? GeometricWeight(interval(ν[1])) : map(νᵢ -> GeometricWeight(interval(νᵢ)), ν))
    pow = Nonlinearity(x -> x^0.7, Complex{Interval{Float64}}[], interval(-Inf, 0))
    far_pole = Nonlinearity(x -> 1/(x - interval(100.0)), Complex{Interval{Float64}}[interval(100.0)+0im], emptyinterval(Float64))
    entire = (exp, cos, sin, cosh, sinh)

    # The nonlinearities sample on a grid and interpolate back, so their image is aliased. Only the
    # constant shortcut is exact.

    @testset "a constant sequence is mapped exactly" begin
        for (f, v, w) ∈ ((inv, 4.0, 0.25), (sqrt, 4.0, 2.0), (cbrt, 8.0, 2.0), (cbrt, -8.0, -2.0)), s ∈ (𝒯, ℱ, 𝒞)
            a, b = _el_seq(s, 0 => v), _el_seq(s, 0 => w)
            @test f(a) == b
            @test f(interval.(a)) == interval.(b)
        end
        @test inv(zeros(𝒯))[0] == Inf # the reciprocal has no domain guard
        @test_throws DomainError sqrt(_el_seq(𝒯, 0 => -4.0))
        # the guard of sqrt covers the constant shortcut only
        @test coefficients(sqrt(_el_seq(𝒯, 0 => -1.0, 1 => 0.01))) isa Vector{Float64}
    end

    @testset "the image evaluates to f(a(x))" begin
        cos_small = Sequence(ℱ, [0.0, 0.05, 0.0, 0.05, 0.0]) # 0.1cos(θ)
        cases = (
            # (f, a, evaluation points, tolerance)
            (inv, _el_seq(𝒯, 0 => 2.0, 1 => 0.3), (0.1, 0.3, -0.2), 1e-6),
            (inv, Sequence(ℱ, [0.0, 0.05, 1.0, 0.05, 0.0]), (0.0, 0.7, 2.0), 1e-3),
            (inv, _el_seq(𝒯 ⊗ 𝒯, (0, 0) => 2.0, (1, 0) => 0.1), ((0.05, 0.0),), 1e-6),
            (inv, Sequence(𝒯, ComplexF64[1.0+0.2im, 0.1, 0.0, 0.0, 0.0]), (0.1,), 1e-6),
            (sqrt, _el_seq(𝒯, 0 => 4.0, 1 => 0.2), (0.1, 0.2, -0.1), 1e-6),
            (sqrt, _el_seq(𝒞, 0 => 4.0, 1 => 0.1), (0.0, 0.5, -0.3), 1e-6),
            (cbrt, _el_seq(𝒯, 0 => 8.0, 1 => 0.3), (0.1, -0.1, 0.05), 1e-8),
            (x -> x^1.5, _el_seq(𝒯, 0 => 1.0, 1 => 0.05), (0.0, 0.1, -0.05), 1e-8),
            (x -> x^0.5, _el_seq(Taylor(2) ⊗ Taylor(2), (0, 0) => 1.0, (1, 0) => 0.05, (0, 1) => 0.02), ((0.1, 0.0),), 1e-6),
            (exp, cos_small, (0.3, -0.7, 1.4), 1e-3),
            (sin, cos_small, (0.3, -0.7, 1.4), 1e-4),
            (cos, cos_small, (0.3, -0.7, 1.4), 1e-4),
            (sinh, cos_small, (0.3, -0.7, 1.4), 1e-4),
            (cosh, cos_small, (0.3, -0.7, 1.4), 1e-4),
            (exp, _el_seq(𝒞, 1 => 0.1, 2 => 0.02), (0.0, 0.5, -0.3), 1e-3),
            (exp, _el_seq(Taylor(2) ⊗ Taylor(2), (0, 0) => 0.1, (1, 0) => 0.05), ((0.1, 0.2), (0.0, 0.0)), 1e-6),
            (exp, Sequence(𝒯, ComplexF64[0.5+0.1im, 0.2, 0.0, 0.0, 0.0]), (0.1, -0.2), 1e-6),
        )
        for (f, a, xs, tol) ∈ cases
            b = f(a)
            for x ∈ xs
                @test b(x...) ≈ f(a(x...)) atol=tol
            end
        end
    end

    @testset "division / and \\" begin
        a, b = _el_seq(𝒯, 0 => 1.0, 1 => 0.5), _el_seq(𝒯, 0 => 2.0)
        q = _el_seq(𝒯, 0 => 0.5, 1 => 0.25)

        # a constant divisor short-circuits, without mutating the dividend
        @test a / b == b \ a == q
        @test a == _el_seq(𝒯, 0 => 1.0, 1 => 0.5)
        @test interval.(a) / interval.(b) == interval.(q)

        @test 3.0 / b == b \ 3.0 == _el_seq(𝒯, 0 => 1.5)
        @test 1 / b == inv(b)

        # a / b is no polynomial: its order exceeds those of a and b
        c, d = Sequence(Taylor(2), [1.0, 0.1, 0.0]), Sequence(Taylor(3), [2.0, 0.1, 0.0, 0.0])
        @test order(space(c / d)) > 3
        @test (c / d)(0.3) ≈ c(0.3) / d(0.3)

        e, f = _el_seq(𝒞, 0 => 3.0, 1 => 0.1), _el_seq(𝒞, 0 => 1.0, 1 => 0.05)
        for x ∈ (0.0, 0.5, -0.4)
            @test (e / f)(x) ≈ e(x) / f(x) atol=1e-5
        end
    end

    @testset "power ^ dispatches onto the integer power, sqrt and cbrt" begin
        a = _el_seq(𝒯, 0 => 1.0, 1 => 0.05)
        # an integer-valued Real forwards to the exact convolution power
        @test a^2.0 == a^2
        @test a^(3//1) == a^3
        @test a^(1//2) == sqrt(a) == a^0.5
        @test a^(1//3) == cbrt(a)
    end

    @testset "entire functions" begin
        maclaurin = (
            (exp, k -> 1 / factorial(k)),
            (cos, k -> iseven(k) ? (-1.0)^(k÷2) / factorial(k) : 0.0),
            (sin, k -> isodd(k) ? (-1.0)^((k-1)÷2) / factorial(k) : 0.0),
            (cosh, k -> iseven(k) ? 1 / factorial(k) : 0.0),
            (sinh, k -> isodd(k) ? 1 / factorial(k) : 0.0),
        )
        @testset "Maclaurin coefficients of f(x)" begin
            a = _el_seq(Taylor(8), 1 => 1.0)
            for (f, c) ∈ maclaurin
                @test coefficients(project(f(a), Taylor(8))) ≈ c.(0:8) atol=1e-10
            end

            # a plain sequence carries no error bound: its enclosures hold up to the aliasing
            ai = interval.(_el_seq(Taylor(16), 1 => 1.0))
            @test all(isguaranteed, coefficients(exp(ai)))
            for (f, c) ∈ maclaurin[1:2]
                b = f(ai)
                @test all(k -> in_interval(c(k), b[k]), 0:8)
            end
        end

        @testset "SymmetricSpace: the full lattice by default, the symmetry on request" begin
            # exp of an even function is even
            seven = evensym(Fourier(4, 1.0))
            ae = _el_seq(seven, 0 => 1.0, 1 => 0.1)
            b_exp = exp(ae)
            @test space(b_exp) isa Fourier
            @test b_exp[1] ≈ b_exp[-1] atol=1e-12
            @test b_exp[2] ≈ b_exp[-2] atol=1e-12

            # the symmetry can be imposed, a codomain fixes the space, and not both at once
            b_even = Nonlinearity(exp)(ae; symmetry = symmetry(seven))
            @test space(b_even) isa SymmetricSpace
            @test b_even[1] ≈ b_exp[1] atol=1e-12
            @test space(Nonlinearity(exp)(ae; codomain = desymmetrize(seven))) == desymmetrize(seven)
            @test_throws ArgumentError Nonlinearity(exp)(ae; codomain = seven, symmetry = symmetry(seven))

            # sin of an odd function is odd, cos of an odd function is even
            ao = _el_seq(oddsym(ℱ), 1 => 0.1, 2 => 0.02)
            b_sin = Nonlinearity(sin)(ao; codomain = ℱ)
            b_cos = Nonlinearity(cos)(ao; codomain = ℱ)
            @test b_sin[1] ≈ -b_sin[-1] atol=1e-12
            @test b_cos[1] ≈  b_cos[-1] atol=1e-12
        end

        @testset "CartesianSpace: unsupported" begin
            a = zeros(Taylor(2) × Taylor(2))
            component(a, 1)[1] = 0.1
            @test_throws MethodError exp(a)
        end
    end

    @testset "Nonlinearity struct" begin
        @test Nonlinearity(exp) isa Nonlinearity{typeof(exp), Complex{Interval{Float64}}, Interval{Float64}}
        @test isempty(Nonlinearity(exp).poles)
        @test isempty_interval(Nonlinearity(exp).branch_cut)

        # the poles are only checked on an InfiniteSequence
        a = Sequence(Taylor(2), [0.5, 0.01, 0.0])
        nl = Nonlinearity(x -> 1/(x - 1.0), Complex{Interval{Float64}}[], emptyinterval(Float64))
        @test nl(a; codomain = Taylor(2))(0.0) ≈ 1/(a(0.0) - 1.0) atol=1e-6
    end

    @testset "InfiniteSequence: the constant shortcut has zero error" begin
        c = InfiniteSequence(Sequence(Fourier(0, 1.0), [interval(2.0)]), ℓ¹ν(1.2))
        for f ∈ (inv, sqrt, cbrt, entire...)
            r = f(c)
            @test isthinzero(finite_error(r))
            @test isthinzero(tail_error(r))
            @test isthinzero(total_error(r))
        end
    end

    @testset "InfiniteSequence: inv, sqrt, cbrt and division enclose the exact value" begin
        a = InfiniteSequence(Sequence(ℱ, interval.([0.0, 0.01, 1.0, 0.01, 0.0])), ℓ¹ν(1.2)) # a(0) = 1.02
        ball(r) = r(interval(0.0)) + interval(-1, 1) * total_error(r)

        for op ∈ (inv, sqrt, cbrt)
            r = op(a)
            @test inf(total_error(r)) ≥ 0
            @test isfinite(sup(total_error(r)))
            @test in_interval(op(1.02), ball(r))
            @test RadiiPolynomial._isguaranteed(r)
        end
        @test in_interval(1.0, ball(a / a))

        # the constant divisor shortcut does not mutate the dividend
        a1 = InfiniteSequence(Sequence(Taylor(2), interval.([1.0, 0.5, 0.0])), ℓ¹ν(1.2))
        b1 = InfiniteSequence(Sequence(Taylor(2), interval.([2.0, 0.0, 0.0])), ℓ¹ν(1.2))
        @test sequence(a1 / b1) == Sequence(Taylor(2), interval.([0.5, 0.25, 0.0]))
        @test sequence(a1) == Sequence(Taylor(2), interval.([1.0, 0.5, 0.0]))

        z = complex(interval(0.0))
        ac = InfiniteSequence(Sequence(Taylor(3), [complex(interval(1.2), interval(0.05)), complex(interval(0.05)), z, z]), ℓ¹ν(1.3))
        @test RadiiPolynomial._isguaranteed(sqrt(ac))

        # the second-order constant of cbrt is the sharp ‖ū⁻²‖(2‖ū‖ + R) on the ball of radius R,
        # which certifies ≈ 0.074602 where a weaker constant stops at ≈ 0.076069
        r = cbrt(InfiniteSequence(sequence(a), ℓ¹ν(1.2); total_error = interval(0.2)))
        @test sup(total_error(r)) < 0.0755
        @test in_interval(cbrt(1.02), ball(r))
        @test RadiiPolynomial._isguaranteed(r)
    end

    @testset "InfiniteSequence: exp on Fourier" begin
        a_seq = Sequence(ℱ, interval.([0.0, 0.05, 0.0, 0.05, 0.0])) # 0.1cos(θ)

        # the error bounds are enclosures, so only their supremum is meaningful
        b = exp(InfiniteSequence(a_seq, ℓ¹ν(1.2)))
        @test sup(finite_error(b)) > 0
        @test sup(tail_error(b)) > 0
        @test isequal_interval(total_error(b), finite_error(b) + tail_error(b))
        @test in_interval(exp(0.1), b(interval(0.0)))
        @test RadiiPolynomial._isguaranteed(b)

        # the perturbation term proportional to the input error is added once to the total error,
        # but twice across the finite and tail errors
        b2 = exp(InfiniteSequence(a_seq, ℓ¹ν(1.2); total_error = interval(1e-8)))
        @test sup(total_error(b2)) < sup(finite_error(b2) + tail_error(b2))
        @test sup(finite_error(b2)) > 0
        @test sup(tail_error(b2)) > 0
    end

    @testset "InfiniteSequence: the input-error bound sweeps every corner of the polyannulus" begin
        function corner_bounds(seq, ν, ν̄, ε)
            ρ = interval(1) + interval(ε)
            circle = ρ * cispi(interval(-1, 1))
            transfer = prod((ν̄ .+ ν) ./ (ν̄ .- ν))
            W = [sup(RadiiPolynomial._contour(exp, seq + circle, μ) * transfer) for μ ∈ RadiiPolynomial._polyannulus_corners(ν̄)]
            return W, sup(RadiiPolynomial._perturbation_bound(exp, seq, ρ, ν, ν̄))
        end

        # the negative modes dominate: the corner 1/ν̄ carries the largest bound, W[1] being the corner ν̄
        W, W_all = corner_bounds(Sequence(ℱ, interval.([0.05, 0.3, 1.0, 0.0, 0.0])), (interval(1.2),), (interval(2.0),), 1e-2)
        @test W_all == maximum(W)
        @test W_all > W[1]

        # the mode (-1, 1) dominates: the corner (1/ν̄, ν̄) carries the largest bound
        sq = interval.(_el_seq(Fourier(3, 1.0) ⊗ Fourier(3, 1.0), (0, 0) => 1.0, (-1, 1) => 0.3, (1, 0) => 0.02, (0, 1) => 0.02))
        W, W_all = corner_bounds(sq, (interval(1.3), interval(1.3)), (interval(2.0), interval(2.0)), 1e-6)
        @test W_all == maximum(W)
        @test W_all > 1.5 * W[1]
    end

    @testset "InfiniteSequence: custom Nonlinearity on a TensorSpace" begin
        sT = Fourier(4, 1.0) ⊗ Fourier(4, 1.0)
        nl = Nonlinearity(x -> x^3, Complex{Interval{Float64}}[], emptyinterval(Float64))
        bT = nl(InfiniteSequence(interval.(_el_seq(sT, (1, 0) => 0.05)), ℓ¹ν(1.3, 1.3)); codomain = sT)
        @test isfinite(sup(bT.full_norm))
        @test inf(tail_error(bT)) ≥ 0
        @test inf(finite_error(bT)) ≥ 0
    end

    @testset "InfiniteSequence: poles and branch cut" begin
        seq = interval.(_el_seq(𝒯, 0 => 1.0, 1 => 0.01))

        # a pole at the value a(0) = 1 lies on the image of the contour
        nl_hit = Nonlinearity(x -> 1/(x - interval(1.0)), Complex{Interval{Float64}}[interval(1.0)+0im], emptyinterval(Float64))
        @test_throws AnalyticityError nl_hit(InfiniteSequence(seq, ℓ¹ν(1.5)); codomain = 𝒯)

        # a pole far away from the image, and from the disc of the input error around it
        for ε ∈ (0.0, 1e-8)
            r = far_pole(InfiniteSequence(seq, ℓ¹ν(1.5); total_error = interval(ε)); codomain = 𝒯)
            @test inf(total_error(r)) ≥ 0
            @test isfinite(sup(total_error(r)))
        end

        # the constant term 1 keeps the image of the Taylor disk off (-∞, 0], the constant term 0 does not
        r = InfiniteSequence(seq, ℓ¹ν(1.5))^0.7
        @test isfinite(sup(total_error(r)))
        @test inf(total_error(r)) ≥ 0
        @test_throws AnalyticityError InfiniteSequence(interval.(_el_seq(𝒯, 1 => 0.01)), ℓ¹ν(1.5))^0.7

        # f(z) = 2 - 3z⁻¹ is positive on |z| = ν but hits (-∞, 0] on |z| = 1/ν, so both radii must be swept
        coeffs = complex.(interval.([0.0, -3.0, 2.0, 0.0, 0.0]))
        @test_throws AnalyticityError InfiniteSequence(Sequence(ℱ, coeffs), ℓ¹ν(1.5))^0.7

        # an input error needs f analytic within 1 + ε of the image: here the branch point is closer
        near = Sequence(Taylor(2), interval.([0.8, 0.1, 0.01]))
        @test isfinite(sup(total_error(InfiniteSequence(near, ℓ¹ν(1.2))^0.7)))
        @test_throws AnalyticityError InfiniteSequence(near, ℓ¹ν(1.2); total_error = interval(1e-8))^0.7

        # an entire function is analytic within any radius
        e = InfiniteSequence(Sequence(ℱ, interval.([0.0, 0.05, 1.0, 0.05, 0.0])), ℓ¹ν(1.2); total_error = interval(1e-8))
        @test isfinite(sup(total_error(exp(e))))
    end

    @testset "InfiniteSequence: Fourier and Chebyshev need ν > 1, Taylor admits ν = 1" begin
        for s ∈ (Chebyshev(4), Fourier(4, 1.0)), f ∈ (entire..., x -> x^0.7)
            a = InfiniteSequence(interval.(_el_seq(s, 0 => 1.0, 1 => 0.01)), ℓ¹ν(1.0))
            @test_throws ArgumentError f(a)
        end

        # the disk keeps its interior, and the strict inequalities of the bounds involve only ν̄ > ν
        aT = InfiniteSequence(Sequence(Taylor(3), interval.([1.0, 0.1, 0.01, 0.001])), ℓ¹ν(1.0))
        for f ∈ entire
            r = f(aT)
            @test isfinite(sup(total_error(r)))
            @test RadiiPolynomial._isguaranteed(r)
        end
        r = InfiniteSequence(interval.(_el_seq(𝒯, 0 => 1.0, 1 => 0.01)), ℓ¹ν(1.0))^0.7
        @test isfinite(sup(total_error(r)))
        @test inf(total_error(r)) ≥ 0

        # ν = 1 on a single TensorSpace factor is enough to reject
        sT = Fourier(4, 1.0) ⊗ Fourier(4, 1.0)
        aT_bad = InfiniteSequence(interval.(_el_seq(sT, (0, 0) => 1.0, (1, 0) => 0.01)), ℓ¹ν(1.3, 1.0))
        @test_throws ArgumentError exp(aT_bad)
        @test_throws ArgumentError pow(aT_bad; codomain = sT)
    end

    @testset "InfiniteSequence: Nonlinearity requires a geometric weight" begin
        seqT = Sequence(Taylor(3), interval.([1.0, 0.1, 0.01, 0.001]))
        @test_throws ArgumentError exp(InfiniteSequence(seqT, Ell1(AlgebraicWeight(interval(2.0)))))
        # the identity weight is the rate ν = 1
        @test isfinite(sup(total_error(exp(InfiniteSequence(seqT, Ell1())))))
        @test_throws ArgumentError exp(InfiniteSequence(Sequence(Chebyshev(3), coefficients(seqT)), Ell1()))

        seqF = Sequence(Fourier(3, 1.0), interval.([0.001, 0.01, 0.1, 1.0, 0.1, 0.01, 0.001]))
        @test_throws ArgumentError exp(InfiniteSequence(seqF, Ell1(BesselWeight(interval(2.0)))))

        # a geometric weight on one factor and an algebraic one on the other
        sT = Fourier(4, 1.0) ⊗ Fourier(4, 1.0)
        XT = Ell1((GeometricWeight(interval(1.3)), AlgebraicWeight(interval(2.0))))
        @test_throws ArgumentError exp(InfiniteSequence(interval.(_el_seq(sT, (1, 0) => 0.05)), XT))
    end

    @testset "InfiniteSequence: float coefficients run the whole pipeline, without rigor" begin
        Xf = Ell1(GeometricWeight(1.2))
        seq = Sequence(Taylor(3), [1.5, 0.1, 0.01, 0.001])
        for f ∈ (exp, x -> x^0.7) # entire, and a branch cut
            r = f(InfiniteSequence(seq, Xf))
            @test isfinite(total_error(r))
            @test total_error(r) ≥ 0
            @test eltype(r) === Float64
            @test !RadiiPolynomial._isguaranteed(r)
        end
        rf = inv(InfiniteSequence(Sequence(ℱ, [0.0, 0.05, 1.0, 0.05, 0.0]), Xf))
        @test total_error(rf) ≥ 0
        @test !RadiiPolynomial._isguaranteed(rf)

        # with an input error the perturbation circle is sampled: the errors stay floats
        r_err = exp(InfiniteSequence(seq, Xf; total_error = 1e-8))
        @test total_error(r_err) isa Float64
        @test isfinite(total_error(r_err))
        @test total_error(r_err) > total_error(exp(InfiniteSequence(seq, Xf)))

        @test !RadiiPolynomial._isguaranteed(InfiniteSequence(Sequence(Taylor(2), [1.0, 0.0, 0.0]), Xf))
        @test RadiiPolynomial._isguaranteed(InfiniteSequence(Sequence(Taylor(2), interval.([1.0, 0.0, 0.0])), ℓ¹ν(1.2)))
    end

    @testset "the nonlinearity grid is oversampled 2x" begin
        # aliasing decays like ν̄^{-m} in the grid size m, truncation like (ν/ν̄)^N in the order N:
        # doubling the grid the order alone would need squares the aliasing term
        for s ∈ (Taylor(8), Fourier(8, 1.0), Chebyshev(8))
            @test RadiiPolynomial._oversampled_grid_size(s) ==
                  RadiiPolynomial.fast_grid_size(2 .* grid_size(s), s)
            @test all(RadiiPolynomial._oversampled_grid_size(s) .≥ 2 .* grid_size(s))
        end
        @test RadiiPolynomial._oversampled_grid_size(Taylor(8)) == (32,)
        @test RadiiPolynomial._oversampled_grid_size(Fourier(4, 1.0) ⊗ Fourier(4, 1.0)) == (32, 32)
    end

    @testset "InfiniteSequence: the ν̄ search" begin
        seq(a₀) = InfiniteSequence(Sequence(Taylor(2), interval.([a₀, 0.1, 0.01])), ℓ¹ν(1.2))
        for a₀ ∈ (0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
            @test isfinite(sup(total_error(seq(a₀)^0.7)))
        end

        # the radius returned passes the interval analyticity sweep
        for a₀ ∈ (0.3, 0.5, 0.8, 1.0, 1.5, 3.0), a₁ ∈ (0.05, 0.2, 0.5)
            a = InfiniteSequence(Sequence(Taylor(3), interval.([a₀, a₁, 0.01, 1e-3])), ℓ¹ν(1.3))
            r = try
                a^0.7
            catch e
                @test e isa AnalyticityError # refused outright, which is a certified answer too
                continue
            end
            @test isfinite(sup(total_error(r)))
            ν̄ = RadiiPolynomial._optimize_decay(pow, sequence(r), sequence(a), (interval(1.3),), total_error(a), total_error(a))
            @test RadiiPolynomial._check_branch_cut_poles(pow, sequence(a), interval.(ν̄), total_error(a))
        end

        # a branch cut swallowing the whole plane leaves no radius feasible
        a = InfiniteSequence(Sequence(Taylor(3), interval.([1.5, 0.1, 0.01, 1e-3])), ℓ¹ν(1.3))
        nl = Nonlinearity(x -> x^0.7, Complex{Interval{Float64}}[], interval(-Inf, Inf))
        @test_throws AnalyticityError RadiiPolynomial._optimize_decay(nl, sequence(a), sequence(a), (interval(1.3),), total_error(a), total_error(a))
    end

    @testset "the plateau sets the order of the output" begin
        # the box of kept indices stops before the first noisy coefficient, the rest is charged to the tail
        a = Sequence(Taylor(3), [1.0, 0.1, 0.01, 0.001])
        ν̄ = (2.0,)
        C = maximum(μ -> RadiiPolynomial._contour(exp, a, μ), RadiiPolynomial._polyannulus_corners(ν̄))
        trim(c) = RadiiPolynomial._noise_trim(c, RadiiPolynomial.CauchyCeiling(C, ν̄), RadiiPolynomial.ChargeTail((1.5,) ./ ν̄))
        # a violator at k = 2 and a quiet coefficient below the envelope at k = 3
        @test trim(Sequence(Taylor(3), [1.0, 0.1, 3C / ν̄[1]^2, 1e-20])) == (1,)

        # a plateau reaching the constant term empties the box, which `_alias_and_tail` must still price
        N₀ = trim(Sequence(Taylor(3), [2C, 0.0, 0.0, 0.0]))
        @test N₀ == (-1,)
        finite_alias, tail_alias = RadiiPolynomial._alias_and_tail(RadiiPolynomial._coef_fft_size(Taylor(3), Taylor(3)), (1.5,), ν̄, N₀)
        @test iszero(finite_alias)
        @test isfinite(C * (finite_alias + tail_alias))

        # a noisy corner coefficient of a tensor space is left out along one direction only
        s = Chebyshev(16) ⊗ Chebyshev(4)
        c = Sequence(s, [1e-3 * 0.1^(k + l) for (k, l) ∈ indices(s)])
        c[(16, 4)] = 1.0
        @test RadiiPolynomial._noise_trim(c, RadiiPolynomial.CauchyCeiling(1.0, (4.0, 4.0)), RadiiPolynomial.ChargeTail((0.3, 0.3))) ∈ ((15, 4), (16, 3))
    end

    @testset "InfiniteSequence: coefficients below the noise floor belong to the tail" begin
        K = 50
        a = InfiniteSequence(Sequence(Taylor(K), interval.([5.0^-k for k ∈ 0:K])), ℓ¹ν(1.0))
        r = exp(a)
        N = order(space(sequence(r)))
        @test N < K
        @test sup(finite_error(r)) < sup(tail_error(r))

        # projecting back onto the codomain prices the band by the tail budget, which encloses the
        # true coefficients there
        p = project(r, Taylor(K), Interval{Float64})
        truth = exp(big.(mid.(sequence(a))))
        for k ∈ (N + 1, N + 5, K)
            @test in_interval(Float64(truth[k]), p[k])
        end
    end

    @testset "InfiniteSequence: every direction gets room above ν" begin
        # the input is constant along the second direction, which has no decay to observe
        for (n, ν) ∈ ((2, 1.3), (2, 1.5), (4, 1.3))
            sT = Fourier(n, 1.0) ⊗ Fourier(n, 1.0)
            a = InfiniteSequence(interval.(_el_seq(sT, (0, 0) => 1.0, (1, 0) => 0.01)), ℓ¹ν(ν, ν))
            @test isfinite(sup(total_error(exp(a))))
        end
    end

    @testset "InfiniteSequence: output resolved to a noise floor" begin
        # exp of the interpolant of log(2 + t/2) is the polynomial 2 + t/2, up to noise
        N = 16
        nodes = [-cospi(j / N) for j ∈ 0:N]
        a = interval(real(RadiiPolynomial.to_coef(reverse(log.(2 .+ nodes ./ 2)), Chebyshev(N))))
        for codomain ∈ (Chebyshev(N), Chebyshev(2N))
            r = Nonlinearity(exp)(InfiniteSequence(a, ℓ¹ν(1.2)); codomain = codomain)
            @test sup(total_error(r)) < 1e-12
        end
    end

    @testset "InfiniteSequence: roots of b xᵖ = a share one proof" begin
        X = ℓ¹ν(1.2)
        a = InfiniteSequence(Sequence(Taylor(2), interval.([1.0, 0.5, 0.1])), X; total_error = interval(0.01))
        # an exact constant divisor keeps the error of the numerator
        b = InfiniteSequence(Sequence(Taylor(2), interval.([2.0, 0.0, 0.0])), X)
        @test sup(total_error(a / b)) ≥ 0.005
        # p exactly 1/q is a q-th root, whatever q
        x = InfiniteSequence(Sequence(Chebyshev(1), interval.([2.0, 0.25])), X)
        for p ∈ (0.25, 1//5)
            r = x ^ p
            @test sup(total_error(r)) < 1e-10
            @test sup(norm(sequence(r) ^ Int(inv(p)) - sequence(x), X)) < 1e-10
        end
        # the error is only as guaranteed as the input, and a Float64 factor drops the flag
        ng = InfiniteSequence(Sequence(Chebyshev(1), [interval(2.0) * 1.0, interval(0.25)]), X)
        for F ∈ (inv, sqrt, cbrt, y -> y / (y + interval(1.0)), y -> x / y)
            @test RadiiPolynomial._isguaranteed(F(x))
            @test !RadiiPolynomial._isguaranteed(F(ng))
            @test !isguaranteed(total_error(F(ng)))
        end
        # a quotient far smoother than 1 / b
        bs = Sequence(Fourier(120, 1.0), [(k == 0) + 0.5 * 0.6^abs(k) for k ∈ -120:120])
        Bs = InfiniteSequence(interval.(bs), ℓ¹ν(1.05))
        @test sup(total_error(Bs / Bs)) < 1e-10
        # an exponent that is no 1/q for a small q goes through the general power
        @test isfinite(sup(total_error(x ^ 1e-20)))
        @test isfinite(sup(total_error(x ^ (1 // 40))))
        # a constant input gives the constant f(a₀) in the requested space
        c = InfiniteSequence(Sequence(Taylor(2), interval.([4.0, 0.0, 0.0])), X)
        @test space(sequence(sqrt(c; codomain = Taylor(5)))) == Taylor(5)
        @test in_interval(2.0, sequence(sqrt(c; codomain = Taylor(5)))[0])
        @test space(Nonlinearity(exp)(sequence(c); codomain = Taylor(7))) == Taylor(7)
    end

    @testset "a slow decay is not mistaken for a plateau" begin
        # 1 / (1 + 0.99t) decays like 1.15^-k
        a = Sequence(Chebyshev(1), [1.0, 0.495])
        x = inv(a)
        @test order(space(x)) > 100
        @test maximum(abs, coefficients(x * a - 1)) < 1e-10
        @test sup(total_error(inv(InfiniteSequence(interval.(a), ℓ¹ν(1.001))))) < 1e-8
        # an order too low for the proof to succeed
        @test_throws ProofError inv(InfiniteSequence(interval.(a), ℓ¹ν(1.001)); codomain = Chebyshev(8))
    end

    @testset "InfiniteSequence: a symmetric input is read on the full lattice" begin
        F = Fourier(4, 1.0)
        pow15 = Nonlinearity(x -> x ^ 1.5, Complex{Interval{Float64}}[], interval(-Inf, 0))
        cases = ((evensym(F), [1.0, 0.1, 0.02, 0.0, 0.0], exp), (oddsym(F), [0.1, 0.02, 0.0, 0.0], sin), (evensym(F), [2.0, 0.1, 0.02, 0.0, 0.0], pow15))
        for (s, coeffs, f) ∈ cases, ε ∈ (0.0, 1e-8)
            a = Sequence(s, interval.(coeffs))
            r = f(InfiniteSequence(a, ℓ¹ν(1.2); total_error = interval(ε)))
            r_full = f(InfiniteSequence(Projection(F) * a, ℓ¹ν(1.2); total_error = interval(ε)))
            @test space(sequence(r)) == space(sequence(r_full))
            @test all(isequal_interval.(coefficients(sequence(r)), coefficients(sequence(r_full))))
            @test isequal_interval(total_error(r), total_error(r_full))
        end
        F² = Fourier(1, 1.0) ⊗ Fourier(1, 1.0)
        a = Sequence(d4sym(F²), interval.([1.0, 0.05, 0.02]))
        @test isequal_interval(total_error(exp(InfiniteSequence(a, ℓ¹ν(1.2, 1.2)))),
                               total_error(exp(InfiniteSequence(Projection(F²) * a, ℓ¹ν(1.2, 1.2)))))
    end

    @testset "InfiniteSequence: a function not real on the real axis keeps its imaginary part" begin
        nl = Nonlinearity(x -> exp(im * x), Complex{Interval{Float64}}[], emptyinterval())
        r = nl(InfiniteSequence(Sequence(Chebyshev(1), interval.([0.0, 0.25])), ℓ¹ν(1.2)); codomain = Chebyshev(12))
        @test eltype(sequence(r)) == Complex{Interval{Float64}}
        @test abs(mid.(sequence(r))(0.3) - exp(0.15im)) ≤ sup(total_error(r)) + 1e-12 # a(0.3) = 0.15
        # the constant shortcut too
        r₀ = nl(InfiniteSequence(Sequence(Chebyshev(1), interval.([0.5, 0.0])), ℓ¹ν(1.2)); codomain = Chebyshev(1))
        @test mid(sequence(r₀)[0]) ≈ exp(0.5im)
    end

    @testset "a failed verification is a VerificationFailure" begin
        @test AnalyticityError <: VerificationFailure
        @test ProofError <: VerificationFailure
        @test sprint(showerror, AnalyticityError("pole met")) == "AnalyticityError: pole met"
        @test sprint(showerror, ProofError("Z₁ ≥ 1")) == "ProofError: Z₁ ≥ 1"
        a = InfiniteSequence(interval.(Sequence(Taylor(1), [0.0, 0.01])), ℓ¹ν(1.5))
        @test_throws VerificationFailure a^0.7
    end
end
