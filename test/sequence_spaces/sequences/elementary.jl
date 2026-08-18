@testset "Elementary nonlinearities" begin

    𝒯 = Taylor(4)
    ℱ = Fourier(2, 1.0)
    𝒞 = Chebyshev(4)

    # The nonlinearities below sample on a grid and interpolate back, so their image is
    # *aliased*: non-constant checks use ≈ or evaluation cross-checks with an explicit
    # tolerance, while the exact constant-sequence shortcut is checked with ==.

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
        # the reciprocal has no domain guard: 1/0 gives Inf rather than an error
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
            #= The order of `a` sets the grid size, hence how far the coefficients alias: a
               grid of m folds k onto k+m. For 1/(2 + x/10) the exact coefficients are
               0.5(-0.05)^k, so at m = 4 the leading aliased term is 0.5·0.05^4 ≈ 3.1e-6,
               above the tolerance below; padding `a` to order 4 raises the grid to 8 and
               the aliasing to 0.5·0.05^8 ≈ 2e-11. =#
            s = Taylor(4) ⊗ Taylor(4)
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
            @test inv(a) == Sequence(𝒯, interval.([0.25, 0.0, 0.0, 0.0, 0.0])) # `==` compares interval bounds
        end
    end

    @testset "sqrt" begin
        @test sqrt(Sequence(𝒯, [4.0, 0.0, 0.0, 0.0, 0.0])) == Sequence(𝒯, [2.0, 0.0, 0.0, 0.0, 0.0])

        @testset "negative constant term throws DomainError" begin
            a = Sequence(𝒯, [-4.0, 0.0, 0.0, 0.0, 0.0])
            @test_throws DomainError sqrt(a)
        end

        @testset "negative non-constant term does NOT throw (guard only covers the constant shortcut)" begin
            a = Sequence(𝒯, [-1.0, 0.01, 0.0, 0.0, 0.0])
            b = sqrt(a) # the real part of the complex grid values is taken silently
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

        # an integer-valued Real forwards to the exact convolution power, not to the grid
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
            # a(x) = x on Taylor(8) samples 32 nodes, so the aliasing error is
            # O(1/32!) ≈ 4e-36, negligible at Float64 precision
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

        @testset "Interval{Float64}: hand-computed Maclaurin coefficients" begin
            #= On a grid of m nodes the transform returns the aliased coefficients Σⱼ cₖ₊ⱼₘ,
               and a plain sequence carries no aliasing bound (only an infinite sequence does),
               so these match the Maclaurin coefficients only while the aliasing stays below
               the width of the enclosure. For exp the leading aliased term is 1/m!: ≈ 4.8e-14
               at m = 16, which already moves b[0] off 1.0, against ≈ 3.8e-36 at m = 32 — hence
               the padding to order 16, the smallest order whose grid rounds up to 32. =#
            a = Sequence(Taylor(16), interval.(vcat([0.0, 1.0], zeros(15))))
            b = exp(a)
            @test all(isguaranteed, coefficients(b))
            @test all(k -> in_interval(1 / factorial(k), b[k]), 0:8)

            bc = cos(a)
            expected_cos = [iseven(k) ? (-1.0)^(k÷2) / factorial(k) : 0.0 for k ∈ 0:8]
            @test all(k -> in_interval(expected_cos[k+1], bc[k]), 0:8)
        end

        @testset "SymmetricSpace: requires an explicit (desymmetrized) codomain" begin
            # the codomain is derived from the space of `a`, and is undefined for a
            # symmetric space, so a direct call throws
            seven = evensym(Fourier(4, 1.0))
            ae = Sequence(seven, zeros(Float64, dimension(seven)))
            ae[0] = 1.0
            ae[1] = 0.1
            @test_throws MethodError exp(ae)

            # supplying the desymmetrized codomain works, and exp of an even function is even
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

        @testset "CartesianSpace: unsupported" begin
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
            # the exact value at θ = 0 lies in the computed ball
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

    @testset "cbrt: second-order constant is the sharp ‖ū⁻²‖(2‖ū‖ + R)" begin
        #= For the cubic map F(u) = u³ - a with A ≈ (3ū²)⁻¹ = ū⁻²/3,
             ‖A(DF(u) - DF(ū))‖ = ‖ū⁻²(u - ū)(u + ū)‖ ≤ ‖ū⁻²‖‖u-ū‖(2‖ū‖ + ‖u-ū‖),
           so the sharp second-order constant on B_R(ū) is Z₂ = ‖ū⁻²‖(2‖ū‖ + R), not the
           weaker (but still valid) ‖ū⁻²‖(2‖ū‖ + 2R). The gap is second order and shows up
           in the certified radius only once Y is sizeable, hence the large input error. =#
        X = Ell1(GeometricWeight(interval(1.2)))
        a = InfiniteSequence(Sequence(Fourier(2, 1.0), interval.([0.0, 0.01, 1.0, 0.01, 0.0])),
                             X; total_error = interval(0.2))
        r = cbrt(a)
        # the sharp constant certifies ≈ 0.074602; the weaker one stops at ≈ 0.076069
        @test sup(total_error(r)) < 0.0755
        # whichever constant is used, the ball must still enclose the true cube root
        @test in_interval(cbrt(1.02), r(interval(0.0)) + interval(-1, 1) * total_error(r))
        @test RadiiPolynomial._isguaranteed(r)
    end

    @testset "InfiniteSequence: exp on thin Fourier" begin
        X = Ell1(GeometricWeight(interval(1.2)))
        a_seq = Sequence(Fourier(2, 1.0), interval.([0.0, 0.05, 0.0, 0.05, 0.0])) # 0.1cos(θ)

        @testset "Case A: no input error" begin
            a = InfiniteSequence(a_seq, X)
            b = exp(a)
            # the error bounds are enclosures, so only their supremum is meaningful:
            # asserting on the infimum would pin the test to how tightly they come out
            @test sup(finite_error(b)) > 0
            @test sup(tail_error(b)) > 0
            @test isequal_interval(total_error(b), finite_error(b) + tail_error(b))
            @test in_interval(exp(0.1), b(interval(0.0)))
            @test RadiiPolynomial._isguaranteed(b)
        end

        @testset "Case B: with input error" begin
            a2 = InfiniteSequence(a_seq, X; total_error = interval(1e-8))
            b2 = exp(a2)
            # the perturbation term proportional to the input error is added once to
            # the total error, but twice across the finite and tail errors
            @test sup(total_error(b2)) < sup(finite_error(b2) + tail_error(b2))
            @test sup(finite_error(b2)) > 0
            @test sup(tail_error(b2)) > 0
        end
    end

    @testset "InfiniteSequence: the input-error bound sweeps every corner of the polyannulus" begin
        #= The perturbation term is W·ε with W ≥ max_θ ‖f(ū + r_⋆e^{iθ})‖_ν, and that ℓ¹_ν norm
           comes from the contour values through the Cauchy transfer ∏(ν̄+ν)/(ν̄−ν), so the
           contour must be swept at all 2^d corners of the ν̄-polyannulus (radius ν̄ᵢ or 1/ν̄ᵢ
           per direction). A single corner under-estimates W — by 5.6x in one direction and
           324x in two — hence the deliberately asymmetric spectra below. =#
        X = Ell1(GeometricWeight(interval(1.2)))
        a1 = InfiniteSequence(Sequence(Fourier(2, 1.0), interval.([0.05, 0.3, 1.0, 0.0, 0.0])),
                              X; total_error = interval(1e-2))
        # W is 5.6x larger at the 1/ν̄ corner, and ε is large enough for W·ε to dominate
        # the ε-independent aliasing floor of ≈ 0.558
        @test sup(total_error(exp(a1))) > 1.0   # one corner stops at ≈ 0.730, the sweep gives ≈ 1.52

        sT = Fourier(3, 1.0) ⊗ Fourier(3, 1.0)
        XT = Ell1(ntuple(_ -> GeometricWeight(interval(1.3)), 2))
        sq = Sequence(sT, zeros(Interval{Float64}, dimension(sT)))
        sq[(0,0)] = interval(1.0); sq[(1,0)] = interval(0.3)
        sq[(0,1)] = interval(0.2); sq[(-1,0)] = interval(0.02)
        aT = InfiniteSequence(sq, XT; total_error = interval(1e-6))
        # the single corner certifies ≈ 2.3e-5, the 4-corner maximum is ≈ 324x larger
        @test sup(total_error(exp(aT))) > 1e-3
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

        # a pole sitting at the value a(0) ≈ 1.0: the contour sweep must detect the
        # intersection and refuse to certify analyticity
        nl_hit = Nonlinearity(x -> 1/(x - interval(1.0)), Complex{Interval{Float64}}[interval(1.0)+0im], emptyinterval(Float64))
        @test_throws ArgumentError nl_hit(a; codomain = Taylor(4))

        # a pole far away from the image does not trigger the guard
        nl_far = Nonlinearity(x -> 1/(x - interval(100.0)), Complex{Interval{Float64}}[interval(100.0)+0im], emptyinterval(Float64))
        r = nl_far(a; codomain = Taylor(4))
        @test inf(total_error(r)) ≥ 0
    end

    @testset "InfiniteSequence: branch-cut / pole guard and the rate ν = 1" begin
        X1 = Ell1(GeometricWeight(interval(1.0))) # ν = 1: strip and ellipse have empty interior
        for s ∈ (Fourier(4, 1.0), Chebyshev(4))
            seq = Sequence(s, zeros(Interval{Float64}, dimension(s)))
            seq[0] = interval(1.0)
            seq[1] = interval(0.01) # non-constant, so the constant shortcut is bypassed
            a = InfiniteSequence(seq, X1)
            @test_throws ArgumentError a^0.7 # x -> x^0.7 has a branch cut on (-∞, 0]
        end

        # Taylor admits ν = 1: the disk keeps its interior, and the strict inequalities of the
        # bounds involve only the auxiliary radius ν̄ > ν
        seqT1 = Sequence(Taylor(4), zeros(Interval{Float64}, 5))
        seqT1[0] = interval(1.0)
        seqT1[1] = interval(0.01)
        rT1 = InfiniteSequence(seqT1, X1)^0.7
        @test isfinite(sup(total_error(rT1)))
        @test inf(total_error(rT1)) ≥ 0 # (the float exponent of ^0.7 makes the result NG regardless of ν)

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

    @testset "InfiniteSequence: float coefficients run the whole pipeline, non-rigorously" begin
        #= The ν̄ the search returns is promoted to intervals only when the coefficients are
           intervals: certified bounds need interval radii, while a float pipeline cannot even
           write them into its grids — and certifies nothing to begin with. A float
           InfiniteSequence therefore runs search, saturation and error bounds entirely in
           floating point, and must run rather than throw. =#
        Xf = Ell1(GeometricWeight(1.2))
        af = InfiniteSequence(Sequence(Taylor(3), [1.5, 0.1, 0.01, 0.001]), Xf)
        for f ∈ (exp, x -> x^0.7) # entire, and a branch cut
            r = f(af)
            @test isfinite(total_error(r))
            @test total_error(r) ≥ 0
            @test eltype(r) === Float64
            @test !RadiiPolynomial._isguaranteed(r)
        end

        # with an input error the perturbation coefficient W is sampled on the circle rather
        # than enclosed, so the errors stay floats instead of NG intervals
        af_err = InfiniteSequence(Sequence(Taylor(3), [1.5, 0.1, 0.01, 0.001]), Xf; total_error = 1e-8)
        r_err = exp(af_err)
        @test total_error(r_err) isa Float64
        @test isfinite(total_error(r_err))
        @test total_error(r_err) > total_error(exp(InfiniteSequence(sequence(af_err), Xf))) # ε contributes
    end

    @testset "the nonlinearity grid is oversampled 2x" begin
        # aliasing decays like ν̄^{-m} in the grid size m, truncation like (ν/ν̄)^N in the
        # order N; doubling the grid the order alone would need squares the aliasing term
        for s ∈ (Taylor(8), Fourier(8, 1.0), Chebyshev(8))
            @test RadiiPolynomial._oversampled_grid_size(s) ==
                  RadiiPolynomial.fast_grid_size(2 .* grid_size(s), s)
            @test all(RadiiPolynomial._oversampled_grid_size(s) .≥ 2 .* grid_size(s))
        end
        @test RadiiPolynomial._oversampled_grid_size(Taylor(8)) == (32,)
        @test RadiiPolynomial._oversampled_grid_size(Fourier(4, 1.0) ⊗ Fourier(4, 1.0)) == (32, 32)
    end

    # `@test_throws` has no `@test_broken` counterpart, hence this predicate
    _throws_argument(f) = try (f(); false) catch e; e isa ArgumentError end

    @testset "InfiniteSequence: entire functions validate ν too" begin
        #= The decay rate is a hypothesis of the Cauchy estimates regardless of where the
           singularities are, so it is validated even for entire functions, which have neither
           poles nor branch cuts to sweep. Fourier and Chebyshev need ν > 1 — at ν = 1 their
           strip/ellipse of analyticity has empty interior — while Taylor admits ν = 1, the
           open unit disk being as good a disk as any. =#
        X1 = Ell1(GeometricWeight(interval(1.0))) # ν = 1
        for (s, coeffs) ∈ ((Chebyshev(3),   [1.0, 0.1, 0.01, 0.001]),
                           (Fourier(3, 1.0),[0.001, 0.01, 0.1, 1.0, 0.1, 0.01, 0.001]))
            a = InfiniteSequence(Sequence(s, interval.(coeffs)), X1)
            for f ∈ (exp, cos, sin, cosh, sinh)
                @test _throws_argument(() -> f(a))
            end
        end

        aT1 = InfiniteSequence(Sequence(Taylor(3), interval.([1.0, 0.1, 0.01, 0.001])), X1)
        for f ∈ (exp, cos, sin, cosh, sinh)
            r = f(aT1)
            @test isfinite(sup(total_error(r)))
            @test RadiiPolynomial._isguaranteed(r)
        end

        # ν = 1 on a single TensorSpace factor is enough to reject
        sT = Fourier(4, 1.0) ⊗ Fourier(4, 1.0)
        seqT = Sequence(sT, zeros(Interval{Float64}, dimension(sT)))
        seqT[(1, 0)] = interval(0.05)
        XT_bad = Ell1((GeometricWeight(interval(1.3)), GeometricWeight(interval(1.0))))
        @test _throws_argument(() -> exp(InfiniteSequence(seqT, XT_bad)))
    end

    @testset "InfiniteSequence: Nonlinearity requires a geometric weight" begin
        #= The Cauchy estimates assume ω_k = ν^{|k|} everywhere, so an algebraic or Bessel
           weight would be unsound rather than merely inaccurate. The identity weight is the
           ν = 1 geometric weight: admitted on Taylor like any other ν = 1 rate, rejected on
           Fourier and Chebyshev. =#
        seqT = Sequence(Taylor(3), interval.([1.0, 0.1, 0.01, 0.001]))
        a = InfiniteSequence(seqT, Ell1(AlgebraicWeight(interval(2.0))))
        @test _throws_argument(() -> exp(a))
        rI = exp(InfiniteSequence(seqT, Ell1())) # IdentityWeight ≡ ν = 1 on Taylor
        @test isfinite(sup(total_error(rI)))
        seqC = Sequence(Chebyshev(3), interval.([1.0, 0.1, 0.01, 0.001]))
        @test _throws_argument(() -> exp(InfiniteSequence(seqC, Ell1())))

        # BesselWeight is only indexable on Fourier
        seqF = Sequence(Fourier(3, 1.0), interval.([0.001, 0.01, 0.1, 1.0, 0.1, 0.01, 0.001]))
        aB = InfiniteSequence(seqF, Ell1(BesselWeight(interval(2.0))))
        @test _throws_argument(() -> exp(aB))

        # a geometric weight on one factor and an algebraic one on the other is rejected
        sT = Fourier(4, 1.0) ⊗ Fourier(4, 1.0)
        seqTT = Sequence(sT, zeros(Interval{Float64}, dimension(sT)))
        seqTT[(1, 0)] = interval(0.05)
        XT = Ell1((GeometricWeight(interval(1.3)), AlgebraicWeight(interval(2.0))))
        @test _throws_argument(() -> exp(InfiniteSequence(seqTT, XT)))
    end

    @testset "InfiniteSequence: the perturbation radius must fit inside the analyticity domain" begin
        #= The input-error term is W·ε with W ≥ max_θ ‖f(ū + r_⋆e^{iθ})‖_ν, valid only while
           0 < r_⋆ < dist(ū(S_ν̄), ∂Ω). The radius is taken as r_⋆ = 1 + ε, which makes the
           quotient ε/(r_⋆ − ε) collapse to ε. Here ū(0) ≈ 0.8 sits closer to the branch
           point of z^0.7 at the origin than r_⋆ ≈ 1, so the estimate does not apply and
           must be refused rather than silently degenerate. =#
        Xt = Ell1(GeometricWeight(interval(1.2)))
        seq = Sequence(Taylor(2), interval.([0.8, 0.1, 0.01]))

        # with no input error the perturbation branch is skipped and the bound is finite
        @test isfinite(sup(total_error(InfiniteSequence(seq, Xt)^0.7)))

        a = InfiniteSequence(seq, Xt; total_error = interval(1e-8))
        @test _throws_argument(() -> a^0.7) # would otherwise return [0.019, ∞)

        # an entire f has Ω = ℂ, so the hypothesis is vacuous and nothing is refused
        Xf = Ell1(GeometricWeight(interval(1.2)))
        e = InfiniteSequence(Sequence(Fourier(2, 1.0), interval.([0.0, 0.05, 1.0, 0.05, 0.0])),
                             Xf; total_error = interval(1e-8))
        @test isfinite(sup(total_error(exp(e))))

        # a pole far enough from the r_⋆-disc around the contour is likewise fine
        nl_far = Nonlinearity(x -> 1/(x - interval(100.0)), Complex{Interval{Float64}}[interval(100.0)+0im], emptyinterval(Float64))
        far = InfiniteSequence(Sequence(Taylor(4), interval.([1.0, 0.01, 0.0, 0.0, 0.0])),
                               Ell1(GeometricWeight(interval(1.5))); total_error = interval(1e-8))
        @test isfinite(sup(total_error(nl_far(far; codomain = Taylor(4)))))
    end

    @testset "InfiniteSequence: the ν̄ search returns a point it actually evaluated" begin
        #= The auxiliary radius ν̄ is chosen by a golden-section search, and feasibility is not
           an interval property here: the contour of x^p at one ν̄ can straddle the branch cut
           (infinite bound) while neighbouring radii are fine. Returning the midpoint of the
           final bracket — a point never evaluated — could therefore land on exactly the ν̄ the
           search spent its budget avoiding, so the radius returned must be one it evaluated. =#
        X = Ell1(GeometricWeight(interval(1.2)))
        seq(a₀) = InfiniteSequence(Sequence(Taylor(2), interval.([a₀, 0.1, 0.01])), X)
        for a₀ ∈ (1.5, 4.0, 6.0)
            @test isfinite(sup(total_error(seq(a₀)^0.7)))
        end
        # neighbouring constant terms must keep working too
        for a₀ ∈ (0.8, 1.0, 2.0, 3.0, 8.0)
            @test isfinite(sup(total_error(seq(a₀)^0.7)))
        end
    end

    @testset "InfiniteSequence: the ν̄ the search settles on is certified, not merely sampled" begin
        #= The search itself is run in floating point, feasibility included: it splits the
           polyannulus at finitely many radii and samples the transform there, where the covering
           sweep encloses it with interval boxes. A sampled test can only ever indicate, so the
           radius it settles on is put through the covering sweep before it is returned — an
           uncertified ν̄ would not announce itself, `f` evaluated in interval arithmetic across
           a branch cut silently returning its intersection with the domain of `f`. =#
        X = Ell1(GeometricWeight(interval(1.3)))
        nl = Nonlinearity(x -> x^0.7, Complex{Interval{Float64}}[], interval(-Inf, 0))
        for a₀ ∈ (0.3, 0.5, 0.8, 1.0, 1.5, 3.0), a₁ ∈ (0.05, 0.2, 0.5)
            a = InfiniteSequence(Sequence(Taylor(3), interval.([a₀, a₁, 0.01, 1e-3])), X)
            r = try
                a^0.7
            catch e
                @test e isa ArgumentError # refused outright, which is a certified answer too
                continue
            end
            @test isfinite(sup(total_error(r)))
            ν̄ = RadiiPolynomial._optimize_decay(nl, sequence(r), a, total_error(a))
            @test RadiiPolynomial._check_branch_cut_poles(nl, sequence(a), interval.(ν̄), total_error(a))
        end
    end

    @testset "InfiniteSequence: a search with nothing feasible left says so" begin
        #= Both passes of the search — the float one and the rerun that tests feasibility with
           the covering sweep — can come back empty handed, and the golden search then returns
           the midpoint of its final bracket, a radius nothing ever certified. Handing that back
           would produce a finite-looking bound resting on `f` evaluated across its own branch
           cut, so the exhausted search must throw instead. =#
        X = Ell1(GeometricWeight(interval(1.3)))
        a = InfiniteSequence(Sequence(Taylor(3), interval.([1.5, 0.1, 0.01, 1e-3])), X)
        # a branch cut swallowing the whole plane leaves no radius feasible, at any sweep
        nl = Nonlinearity(x -> x^0.7, Complex{Interval{Float64}}[], interval(-Inf, Inf))
        @test_throws ArgumentError RadiiPolynomial._optimize_decay(nl, sequence(a), a, total_error(a))
    end

    @testset "the plateau sets the order of the output" begin
        #= `_error` prices every index inside the box [-N_v, N_v] as the aliasing of a computed
           coefficient and every index outside as truncated tail. `_saturation_order` places
           the box just before the first envelope violation — quiet stragglers beyond it are
           as much tail as the violators — and the caller truncates its output to the box, so
           the stored sequence is exactly what the finite budget covers. =#
        a = Sequence(Taylor(3), [1.0, 0.1, 0.01, 0.001])
        ν̄ = (2.0,)
        C = maximum(μ -> RadiiPolynomial._contour(exp, a, μ), RadiiPolynomial._polyannulus_corners(ν̄))
        # a violator at k = 2 and a quiet, envelope-respecting coefficient at k = 3
        c = Sequence(Taylor(3), [1.0, 0.1, 3C / ν̄[1]^2, 1e-20])
        @test RadiiPolynomial._saturation_order(exp, c, a, ν̄) == 1

        # a plateau reaching the constant term empties the box; `_error` must still price it
        c₀ = Sequence(Taylor(3), [2C, 0.0, 0.0, 0.0])
        N_v₀ = RadiiPolynomial._saturation_order(exp, c₀, a, ν̄)
        @test N_v₀ == -1
        Cb, finite_alias, tail_alias = RadiiPolynomial._error(exp, a, c₀, (1.5,), ν̄, N_v₀)
        @test iszero(finite_alias)
        @test isfinite(Cb * (finite_alias + tail_alias))
    end

    @testset "InfiniteSequence: the finite/tail split is honest about the plateau" begin
        #= `InfiniteSequence` splits its error at the order of the stored sequence — `project!`
           prices indices inside it by `finite_error`, indices beyond by `tail_error`. Fast
           decay drives the true coefficients below the noise floor well before the codomain
           order, and those unresolved modes belong to the tail budget: stored as zeros of the
           full codomain they would claim the (far smaller) finite budget — here ~1e-49 against
           true band coefficients ~1e-23. The output's order must be the resolved box. =#
        K = 50
        a = InfiniteSequence(Sequence(Taylor(K), interval.([5.0^-k for k ∈ 0:K])), Ell1(GeometricWeight(interval(1.0))))
        r = exp(a)
        N = order(space(sequence(r)))
        @test N < K # the noise floor cuts well before the codomain order
        @test sup(finite_error(r)) < sup(tail_error(r))

        # projecting back onto the codomain prices the band by the tail budget, which really
        # does enclose the true coefficients there
        p = project(r, Taylor(K), Interval{Float64})
        truth = exp(big.(mid.(sequence(a)))) # high-precision reference, error ≈ eps(BigFloat)
        for k ∈ (N + 1, N + 5, K)
            @test in_interval(Float64(truth[k]), p[k])
        end
    end

    @testset "InfiniteSequence: every direction gets room above ν" begin
        #= `_optimize_decay` brackets its search by the *observed* geometric decay of the input
           and of its image. A tensor direction carrying a single mode has no decay to observe,
           so that upper bound can land below ν, the bracket [ν, ν̄_max] inverts, and the search
           returns ν̄ᵢ < νᵢ. The tail factor ∏ ν̄ᵢ/(ν̄ᵢ − νᵢ) then flips sign, and the failure
           surfaces as "errors must be non-negative" from the constructor rather than as
           anything diagnosable. The bracket must leave every direction room above ν. =#
        sT = Fourier(2, 1.0) ⊗ Fourier(2, 1.0)
        sq = Sequence(sT, zeros(Interval{Float64}, dimension(sT)))
        sq[(0, 0)] = interval(1.0)
        sq[(1, 0)] = interval(0.01) # constant along the second direction: no decay to observe
        for ν ∈ (1.3, 1.5)
            X = Ell1((GeometricWeight(interval(ν)), GeometricWeight(interval(ν))))
            a = InfiniteSequence(sq, X)
            @test isfinite(sup(total_error(exp(a))))
        end

        # the same function on a space with enough modes per direction was always fine
        sT4 = Fourier(4, 1.0) ⊗ Fourier(4, 1.0)
        sq4 = Sequence(sT4, zeros(Interval{Float64}, dimension(sT4)))
        sq4[(0, 0)] = interval(1.0)
        sq4[(1, 0)] = interval(0.01)
        X4 = Ell1((GeometricWeight(interval(1.3)), GeometricWeight(interval(1.3))))
        @test isfinite(sup(total_error(exp(InfiniteSequence(sq4, X4)))))
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
        # f(z) = 2 - 3z⁻¹ is positive on |z| = ν but hits (-∞, 0] on |z| = 1/ν, so both
        # radii must be swept
        ν = 1.5
        Xν = Ell1(GeometricWeight(interval(ν)))
        coeffs = Complex{Interval{Float64}}[interval(0)+0im, -interval(3)+0im, interval(2)+0im, interval(0)+0im, interval(0)+0im]
        a = InfiniteSequence(Sequence(Fourier(2, 1.0), coeffs), Xν)
        @test_throws ArgumentError a^0.7
    end

    @testset "_isguaranteed" begin
        Xf = Ell1(GeometricWeight(1.2))
        af = InfiniteSequence(Sequence(Taylor(2), [1.0, 0.0, 0.0]), Xf)
        @test !RadiiPolynomial._isguaranteed(af) # plain Float64 is not an enclosure

        Xi = Ell1(GeometricWeight(interval(1.2)))
        ai = InfiniteSequence(Sequence(Taylor(2), interval.([1.0, 0.0, 0.0])), Xi)
        @test RadiiPolynomial._isguaranteed(ai)
    end
end
