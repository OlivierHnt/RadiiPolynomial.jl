@testset "Elementary nonlinearities" begin

    𝒯 = Taylor(4)
    ℱ = Fourier(2, 1.0)
    𝒞 = Chebyshev(4)

    # All FFT-based nonlinearities compute the *aliased* pointwise image on a
    # grid of size fft_size(space) and interpolate back; away from the exact
    # constant-sequence shortcut this is an approximation, so non-constant
    # checks below use ≈ / eval cross-checks with explicit tolerances, while
    # the constant shortcut (_isconstant/_at_value) is checked with exact ==.

    @testset "inv" begin
        @testset "Taylor / Fourier / Chebyshev: constant shortcut is exact" begin
            for s ∈ (Taylor(2), Fourier(1, 1.0), Chebyshev(2))
                a = Sequence(s, zeros(Float64, dimension(s)))
                a[0] = 4.0
                expected = Sequence(s, zeros(Float64, dimension(s)))
                expected[0] = 0.25
                @test inv(a) == expected
            end
        end

        @test inv(Sequence(𝒯, [4.0, 0.0, 0.0, 0.0, 0.0])) == Sequence(𝒯, [0.25, 0.0, 0.0, 0.0, 0.0])
        # inv(0) is IEEE Inf, not an error: no domain guard on the reciprocal
        @test inv(Sequence(𝒯, zeros(5)))[0] == Inf

        @testset "Taylor: non-constant, eval cross-check 1/a(x) ≈ (1/a)(x)" begin
            a = Sequence(𝒯, [2.0, 0.3, 0.0, 0.0, 0.0])
            b = inv(a)
            for x ∈ (0.1, 0.3, -0.2)
                @test b(x) ≈ 1 / a(x) atol=1e-6
            end
        end

        @testset "Fourier: non-constant, eval cross-check" begin
            a = Sequence(ℱ, [0.0, 0.05, 1.0, 0.05, 0.0]) # 1 + 0.1cos(θ)
            b = inv(a)
            for θ ∈ (0.0, 0.7, 2.0)
                @test b(θ) ≈ 1 / a(θ) atol=1e-3
            end
        end

        @testset "TensorSpace" begin
            s = Taylor(2) ⊗ Taylor(2)
            a = Sequence(s, zeros(Float64, dimension(s)))
            a[(0,0)] = 2.0
            a[(1,0)] = 0.1
            b = inv(a)
            @test b(0.05, 0.0) ≈ 1 / a(0.05, 0.0) atol=1e-6
        end

        @testset "ComplexF64" begin
            a = Sequence(𝒯, ComplexF64[1.0+0.2im, 0.1, 0.0, 0.0, 0.0])
            b = inv(a)
            @test b(0.1) ≈ 1 / a(0.1) atol=1e-6
        end

        @testset "Interval{Float64}: constant shortcut is an exact enclosure" begin
            a = Sequence(𝒯, interval.([4.0, 0.0, 0.0, 0.0, 0.0]))
            @test inv(a) == Sequence(𝒯, interval.([0.25, 0.0, 0.0, 0.0, 0.0])) # Sequence == Sequence is interval-safe
        end
    end

    @testset "sqrt" begin
        @test sqrt(Sequence(𝒯, [4.0, 0.0, 0.0, 0.0, 0.0])) == Sequence(𝒯, [2.0, 0.0, 0.0, 0.0, 0.0])

        @testset "negative constant term throws DomainError (guarded via _at_value)" begin
            a = Sequence(𝒯, [-4.0, 0.0, 0.0, 0.0, 0.0])
            @test_throws DomainError sqrt(a)
        end

        @testset "negative non-constant term does NOT throw (guard only covers the constant shortcut)" begin
            a = Sequence(𝒯, [-1.0, 0.01, 0.0, 0.0, 0.0])
            b = sqrt(a) # silently takes the real part of the complex grid-based sqrt
            @test coefficients(b) isa Vector{Float64}
        end

        @testset "Taylor: non-constant, eval cross-check" begin
            a = Sequence(𝒯, [4.0, 0.2, 0.0, 0.0, 0.0])
            b = sqrt(a)
            for x ∈ (0.1, 0.2, -0.1)
                @test b(x) ≈ sqrt(a(x)) atol=1e-6
            end
        end

        @testset "Chebyshev: non-constant, eval cross-check" begin
            a = Sequence(𝒞, [4.0, 0.1, 0.0, 0.0, 0.0])
            b = sqrt(a)
            for x ∈ (0.0, 0.5, -0.3)
                @test b(x) ≈ sqrt(a(x)) atol=1e-6
            end
        end

        @testset "Interval{Float64}" begin
            a = Sequence(𝒯, interval.([4.0, 0.0, 0.0, 0.0, 0.0]))
            @test sqrt(a) == Sequence(𝒯, interval.([2.0, 0.0, 0.0, 0.0, 0.0]))
        end
    end

    @testset "cbrt" begin
        @test cbrt(Sequence(𝒯, [8.0, 0.0, 0.0, 0.0, 0.0])) == Sequence(𝒯, [2.0, 0.0, 0.0, 0.0, 0.0])
        # cbrt is odd and defined on all of ℝ: no domain guard, unlike sqrt
        @test cbrt(Sequence(𝒯, [-8.0, 0.0, 0.0, 0.0, 0.0])) == Sequence(𝒯, [-2.0, 0.0, 0.0, 0.0, 0.0])

        @testset "Taylor: non-constant, eval cross-check" begin
            a = Sequence(𝒯, [8.0, 0.3, 0.0, 0.0, 0.0])
            b = cbrt(a)
            for x ∈ (0.1, -0.1, 0.05)
                @test b(x) ≈ cbrt(a(x)) atol=1e-8
            end
        end

        @testset "Interval{Float64}" begin
            a = Sequence(𝒯, interval.([8.0, 0.0, 0.0, 0.0, 0.0]))
            @test cbrt(a) == Sequence(𝒯, interval.([2.0, 0.0, 0.0, 0.0, 0.0]))
        end
    end

    @testset "division / and \\" begin
        @testset "constant divisor short-circuits, without mutating the dividend" begin
            a = Sequence(𝒯, [1.0, 0.5, 0.0, 0.0, 0.0])
            b = Sequence(𝒯, [2.0, 0.0, 0.0, 0.0, 0.0])
            q = a / b
            @test q == Sequence(𝒯, [0.5, 0.25, 0.0, 0.0, 0.0])
            @test a == Sequence(𝒯, [1.0, 0.5, 0.0, 0.0, 0.0])

            a2 = Sequence(𝒯, [1.0, 0.5, 0.0, 0.0, 0.0])
            @test b \ a2 == Sequence(𝒯, [0.5, 0.25, 0.0, 0.0, 0.0])
            @test a2 == Sequence(𝒯, [1.0, 0.5, 0.0, 0.0, 0.0])
        end

        @testset "Number / Sequence and Sequence \\ Number both reduce to inv" begin
            b = Sequence(𝒯, [2.0, 0.0, 0.0, 0.0, 0.0])
            @test 3.0 / b == b \ 3.0 == Sequence(𝒯, [1.5, 0.0, 0.0, 0.0, 0.0])
            @test 1 / b == inv(b)
        end

        @testset "codomain is the union of orders" begin
            a = Sequence(Taylor(2), [1.0, 0.1, 0.0])
            b = Sequence(Taylor(3), [2.0, 0.1, 0.0, 0.0]) # non-constant: goes through the grid path
            q = a / b
            @test space(q) == Taylor(3)
        end

        @testset "Chebyshev: non-constant, eval cross-check a(x)/b(x)" begin
            a = Sequence(𝒞, [3.0, 0.1, 0.0, 0.0, 0.0])
            b = Sequence(𝒞, [1.0, 0.05, 0.0, 0.0, 0.0])
            q = a / b
            for x ∈ (0.0, 0.5, -0.4)
                @test q(x) ≈ a(x) / b(x) atol=1e-5
            end
        end

        @testset "Interval{Float64}: constant divisor is exact" begin
            a = Sequence(𝒯, interval.([1.0, 0.5, 0.0, 0.0, 0.0]))
            b = Sequence(𝒯, interval.([2.0, 0.0, 0.0, 0.0, 0.0]))
            @test a / b == Sequence(𝒯, interval.([0.5, 0.25, 0.0, 0.0, 0.0]))
        end
    end

    @testset "power ^ (dispatch onto ^Integer / sqrt / cbrt / Nonlinearity)" begin
        a = Sequence(𝒯, [1.0, 0.05, 0.0, 0.0, 0.0])

        # a mathematically-integer Real bypasses the FFT approximation entirely
        # and forwards to the exact convolution-based integer power
        @test a^2.0 == a^2
        @test a^(3//1) == a^3

        @test a^(1//2) == sqrt(a) == a^0.5
        @test a^(1//3) == cbrt(a)

        @testset "general real power: eval cross-check a(x)^p ≈ (a^p)(x)" begin
            b = a^1.5
            for x ∈ (0.0, 0.1, -0.05)
                @test b(x) ≈ a(x)^1.5 atol=1e-8
            end
        end

        @testset "TensorSpace" begin
            s = Taylor(2) ⊗ Taylor(2)
            c = Sequence(s, zeros(Float64, dimension(s)))
            c[(0,0)] = 1.0
            c[(1,0)] = 0.05
            c[(0,1)] = 0.02
            b = c^0.5
            @test b(0.1, 0.0) ≈ sqrt(c(0.1, 0.0)) atol=1e-6
        end
    end

    @testset "entire functions: exp, cos, sin, cosh, sinh" begin
        @testset "Taylor: exp/cos/sin/cosh/sinh of x — hand-computed Maclaurin coefficients" begin
            # a(x) = x on Taylor(8); fft_size(Taylor(8)) = 32, so the FFT aliasing
            # error is O(1/32!) ≈ 4·10⁻³⁶ — negligible at Float64 precision
            a = Sequence(Taylor(8), vcat([0.0, 1.0], zeros(7)))

            expected_exp   = [1 / factorial(k) for k ∈ 0:8]
            expected_cos   = [iseven(k) ? (-1.0)^(k÷2) / factorial(k) : 0.0 for k ∈ 0:8]
            expected_sin   = [isodd(k)  ? (-1.0)^((k-1)÷2) / factorial(k) : 0.0 for k ∈ 0:8]
            expected_cosh  = [iseven(k) ? 1 / factorial(k) : 0.0 for k ∈ 0:8]
            expected_sinh  = [isodd(k)  ? 1 / factorial(k) : 0.0 for k ∈ 0:8]

            @test coefficients(exp(a))  ≈ expected_exp  atol=1e-10
            @test coefficients(cos(a))  ≈ expected_cos  atol=1e-10
            @test coefficients(sin(a))  ≈ expected_sin  atol=1e-10
            @test coefficients(cosh(a)) ≈ expected_cosh atol=1e-10
            @test coefficients(sinh(a)) ≈ expected_sinh atol=1e-10
        end

        @testset "Fourier: eval cross-check f(a(θ)) ≈ f(a)(θ)" begin
            a = Sequence(ℱ, [0.0, 0.05, 0.0, 0.05, 0.0]) # 0.1cos(θ)
            for (f, tol) ∈ ((exp, 1e-3), (sin, 1e-4), (cos, 1e-4), (sinh, 1e-4), (cosh, 1e-4))
                b = f(a)
                for θ ∈ (0.3, -0.7, 1.4)
                    @test b(θ) ≈ f(a(θ)) atol=tol
                end
            end
        end

        @testset "Chebyshev: eval cross-check" begin
            a = Sequence(𝒞, [0.0, 0.1, 0.02, 0.0, 0.0])
            b = exp(a)
            for x ∈ (0.0, 0.5, -0.3)
                @test b(x) ≈ exp(a(x)) atol=1e-3
            end
        end

        @testset "TensorSpace: eval cross-check" begin
            s = Taylor(2) ⊗ Taylor(2)
            a = Sequence(s, zeros(Float64, dimension(s)))
            a[(0,0)] = 0.1
            a[(1,0)] = 0.05
            b = exp(a)
            for pt ∈ ((0.1, 0.2), (0.0, 0.0))
                @test b(pt...) ≈ exp(a(pt...)) atol=1e-6
            end
        end

        @testset "ComplexF64: eval cross-check" begin
            a = Sequence(Taylor(4), ComplexF64[0.5+0.1im, 0.2, 0.0, 0.0, 0.0])
            b = exp(a)
            for x ∈ (0.1, -0.2)
                @test b(x) ≈ exp(a(x)) atol=1e-6
            end
        end

        @testset "Interval{Float64}: hand-computed Maclaurin coefficients, rigorous enclosure" begin
            a = Sequence(Taylor(8), interval.(vcat([0.0, 1.0], zeros(7))))
            b = exp(a)
            @test all(isguaranteed, coefficients(b))
            @test all(k -> in_interval(1 / factorial(k), b[k]), 0:8)

            bc = cos(a)
            expected_cos = [iseven(k) ? (-1.0)^(k÷2) / factorial(k) : 0.0 for k ∈ 0:8]
            @test all(k -> in_interval(expected_cos[k+1], bc[k]), 0:8)
        end

        @testset "SymmetricSpace: requires an explicit (desymmetrized) codomain" begin
            # `exp`/`cos`/`sin`/... call Nonlinearity with the *default* codomain
            # kwarg `_codomain(f, space(a))`, which has no method for SymmetricSpace:
            # calling them directly on a symmetric-domain Sequence is a MethodError.
            seven = evensym(Fourier(4, 1.0))
            ae = Sequence(seven, zeros(Float64, dimension(seven)))
            ae[0] = 1.0
            ae[1] = 0.1
            @test_throws MethodError exp(ae)

            # Supplying codomain = desymmetrize(space(a)) explicitly to the
            # Nonlinearity object works, and exp of an even function is even
            b_exp = Nonlinearity(exp)(ae; codomain = desymmetrize(seven))
            @test b_exp[1] ≈ b_exp[-1] atol=1e-12
            @test b_exp[2] ≈ b_exp[-2] atol=1e-12

            # sin of an odd function is odd; cos of an odd function is even
            sodd = oddsym(ℱ)
            ao = Sequence(sodd, zeros(Float64, dimension(sodd)))
            ao[1] = 0.1
            ao[2] = 0.02
            b_sin = Nonlinearity(sin)(ao; codomain = desymmetrize(sodd))
            b_cos = Nonlinearity(cos)(ao; codomain = desymmetrize(sodd))
            @test b_sin[1] ≈ -b_sin[-1] atol=1e-12
            @test b_cos[1] ≈  b_cos[-1] atol=1e-12
        end

        @testset "CartesianSpace: unsupported (no _codomain method), out of scope for this file" begin
            s = Taylor(2) × Taylor(2)
            a = Sequence(s, zeros(Float64, dimension(s)))
            component(a, 1)[1] = 0.1
            @test_throws MethodError exp(a)
        end
    end

    @testset "Nonlinearity struct" begin
        @test Nonlinearity(exp) isa Nonlinearity{typeof(exp), Complex{Interval{Float64}}, Interval{Float64}}
        @test isempty(Nonlinearity(exp).poles)
        @test isempty_interval(Nonlinearity(exp).branch_cut) # entire functions: no branch cut

        @testset "custom pole list, plain Sequence: no guard (guard is InfiniteSequence-only)" begin
            a = Sequence(Taylor(2), [0.5, 0.01, 0.0])
            nl = Nonlinearity(x -> 1/(x - 1.0), Complex{Interval{Float64}}[], emptyinterval(Float64))
            b = nl(a; codomain = Taylor(2))
            @test b(0.0) ≈ 1/(a(0.0) - 1.0) atol=1e-6
        end
    end

    @testset "InfiniteSequence: constant shortcut has zero error, for every operator" begin
        X = Ell1(GeometricWeight(interval(1.2)))
        c = InfiniteSequence(Sequence(Fourier(0, 1.0), [interval(2.0)]), X)
        for f ∈ (inv, sqrt, cbrt, exp, sin, cos, sinh, cosh)
            r = f(c)
            @test isthinzero(finite_error(r))
            @test isthinzero(tail_error(r))
            @test isthinzero(total_error(r))
        end
    end

    @testset "InfiniteSequence: inv / sqrt / cbrt / division, rigorous enclosure" begin
        X = Ell1(GeometricWeight(interval(1.2)))
        a_seq = Sequence(Fourier(2, 1.0), interval.([0.0, 0.01, 1.0, 0.01, 0.0])) # a(0) = sum of coeffs = 1.02

        a = InfiniteSequence(a_seq, X)
        b = InfiniteSequence(a_seq, X)
        a0 = 1.02

        for op ∈ (inv, sqrt, cbrt)
            r = op(a)
            @test inf(total_error(r)) ≥ 0
            @test isfinite(sup(total_error(r)))
            # the exactly-known value at θ=0 lies within the computed ball
            @test in_interval(op(a0), r(interval(0.0)) + interval(-1, 1) * total_error(r))
            @test RadiiPolynomial._isguaranteed(r)
        end

        q = a / b
        @test in_interval(1.0, q(interval(0.0)) + interval(-1, 1) * total_error(q))

        @testset "constant-divisor shortcut does not mutate the dividend" begin
            X1 = Ell1(GeometricWeight(interval(1.2)))
            a1 = InfiniteSequence(Sequence(Taylor(2), interval.([1.0, 0.5, 0.0])), X1)
            b1 = InfiniteSequence(Sequence(Taylor(2), interval.([2.0, 0.0, 0.0])), X1)
            q1 = a1 / b1
            @test sequence(q1) == Sequence(Taylor(2), interval.([0.5, 0.25, 0.0]))
            @test sequence(a1) == Sequence(Taylor(2), interval.([1.0, 0.5, 0.0]))
        end

        @testset "non-interval (plain Float64) InfiniteSequence: works but is not rigorous" begin
            Xf = Ell1(GeometricWeight(1.2))
            af = InfiniteSequence(Sequence(Fourier(2, 1.0), [0.0, 0.05, 1.0, 0.05, 0.0]), Xf)
            rf = inv(af)
            @test total_error(rf) ≥ 0
            @test !RadiiPolynomial._isguaranteed(rf) # isguaranteed(::Float64) is always false
        end

        @testset "Complex{Interval{Float64}} coefficients" begin
            Xc = Ell1(GeometricWeight(interval(1.3)))
            z1, z2, z0 = complex(interval(1.2), interval(0.05)), complex(interval(0.05), interval(0.0)), complex(interval(0.0), interval(0.0))
            ac = InfiniteSequence(Sequence(Taylor(3), [z1, z2, z0, z0]), Xc)
            rc = sqrt(ac)
            @test RadiiPolynomial._isguaranteed(rc)
        end
    end

    @testset "InfiniteSequence: exp on thin Fourier" begin
        X = Ell1(GeometricWeight(interval(1.2)))
        a_seq = Sequence(Fourier(2, 1.0), interval.([0.0, 0.05, 0.0, 0.05, 0.0])) # 0.1cos(θ)

        @testset "Case A: no input error" begin
            a = InfiniteSequence(a_seq, X)
            b = exp(a)
            # the error bounds are enclosures, so only their supremum is meaningful;
            # asserting on the infimum would tie the test to how tightly they happen
            # to be computed (see `set_fft_algorithm`)
            @test sup(finite_error(b)) > 0
            @test sup(tail_error(b)) > 0
            @test isequal_interval(total_error(b), finite_error(b) + tail_error(b))
            @test in_interval(exp(0.1), b(interval(0.0)))
            @test RadiiPolynomial._isguaranteed(b)
        end

        @testset "Case B: with input error" begin
            a2 = InfiniteSequence(a_seq, X; total_error = interval(1e-8))
            b2 = exp(a2)
            # the W·sequence_error(a) perturbation term is added once to total,
            # twice to finite_error + tail_error (see the source comment)
            @test sup(total_error(b2)) < sup(finite_error(b2) + tail_error(b2))
            @test sup(finite_error(b2)) > 0
            @test sup(tail_error(b2)) > 0
        end
    end

    @testset "InfiniteSequence: TensorSpace Nonlinearity (custom f = x -> x^3)" begin
        sT = Fourier(4, 1.0) ⊗ Fourier(4, 1.0)
        XT = Ell1(ntuple(_ -> GeometricWeight(interval(1.3)), 2))
        seq = Sequence(sT, zeros(Interval{Float64}, dimension(sT)))
        seq[(1, 0)] = interval(0.05)
        aT = InfiniteSequence(seq, XT)
        nl = Nonlinearity(x -> x^3, Complex{Interval{Float64}}[], emptyinterval(Float64))
        bT = nl(aT; codomain = sT)
        @test isfinite(sup(bT.full_norm))
        @test inf(tail_error(bT)) ≥ 0
        @test inf(finite_error(bT)) ≥ 0
    end

    @testset "InfiniteSequence: custom Nonlinearity with an explicit pole list" begin
        X = Ell1(GeometricWeight(interval(1.5)))
        seq = Sequence(Taylor(4), interval.([1.0, 0.01, 0.0, 0.0, 0.0]))
        a = InfiniteSequence(seq, X)

        # pole sitting right at the DC value a(0) ≈ 1.0: the Cauchy-contour sweep
        # must detect the intersection and refuse to certify analyticity
        nl_hit = Nonlinearity(x -> 1/(x - interval(1.0)), Complex{Interval{Float64}}[interval(1.0)+0im], emptyinterval(Float64))
        @test_throws ArgumentError nl_hit(a; codomain = Taylor(4))

        # a pole far away from the image does not trigger the guard
        nl_far = Nonlinearity(x -> 1/(x - interval(100.0)), Complex{Interval{Float64}}[interval(100.0)+0im], emptyinterval(Float64))
        r = nl_far(a; codomain = Taylor(4))
        @test inf(total_error(r)) ≥ 0
    end

    @testset "InfiniteSequence: branch-cut / pole guard requires ν > 1" begin
        for s ∈ (Fourier(4, 1.0), Chebyshev(4), Taylor(4))
            X1 = Ell1(GeometricWeight(interval(1.0))) # ν = 1: not enough decay to certify analyticity
            seq = Sequence(s, zeros(Interval{Float64}, dimension(s)))
            seq[0] = interval(1.0)
            seq[1] = interval(0.01) # non-constant, so the _isconstant shortcut is bypassed
            a = InfiniteSequence(seq, X1)
            @test_throws ArgumentError a^0.7 # x -> x^0.7 has a branch cut on (-∞, 0]
        end

        # ν = 1 on a single TensorSpace factor is enough to reject
        sT = Fourier(4, 1.0) ⊗ Fourier(4, 1.0)
        XT_bad = Ell1((GeometricWeight(interval(1.3)), GeometricWeight(interval(1.0))))
        seqT = Sequence(sT, zeros(Interval{Float64}, dimension(sT)))
        seqT[(0, 0)] = interval(1.0)
        seqT[(1, 0)] = interval(0.01)
        aT_bad = InfiniteSequence(seqT, XT_bad)
        nl = Nonlinearity(x -> x^0.7, Complex{Interval{Float64}}[], interval(-Inf, 0))
        @test_throws ArgumentError nl(aT_bad; codomain = sT)
    end

    @testset "InfiniteSequence: Taylor disk vs. branch cut" begin
        Xt = Ell1(GeometricWeight(interval(1.5)))

        # constant term 1.0 keeps the image of the disk off (-∞, 0]
        seq_ok = Sequence(Taylor(4), interval.([1.0, 0.01, 0.0, 0.0, 0.0]))
        r = (InfiniteSequence(seq_ok, Xt))^0.7
        @test isfinite(sup(total_error(r)))
        @test inf(total_error(r)) ≥ 0

        # constant term 0.0: the image straddles the cut
        seq_bad = Sequence(Taylor(4), interval.([0.0, 0.01, 0.0, 0.0, 0.0]))
        @test_throws ArgumentError (InfiniteSequence(seq_bad, Xt))^0.7
    end

    @testset "InfiniteSequence: Fourier branch-cut check over the full annulus" begin
        # f(z) = 2 - 3z⁻¹ is positive on |z| = ν but hits (-∞, 0] on |z| = 1/ν:
        # the check sweeps both radii (no Hermitian-symmetry shortcut)
        ν = 1.5
        Xν = Ell1(GeometricWeight(interval(ν)))
        coeffs = Complex{Interval{Float64}}[interval(0)+0im, -interval(3)+0im, interval(2)+0im, interval(0)+0im, interval(0)+0im]
        a = InfiniteSequence(Sequence(Fourier(2, 1.0), coeffs), Xν)
        @test_throws ArgumentError a^0.7
    end

    @testset "_isguaranteed" begin
        Xf = Ell1(GeometricWeight(1.2))
        af = InfiniteSequence(Sequence(Taylor(2), [1.0, 0.0, 0.0]), Xf)
        @test !RadiiPolynomial._isguaranteed(af) # plain Float64: not a guaranteed enclosure

        Xi = Ell1(GeometricWeight(interval(1.2)))
        ai = InfiniteSequence(Sequence(Taylor(2), interval.([1.0, 0.0, 0.0])), Xi)
        @test RadiiPolynomial._isguaranteed(ai)
    end
end
