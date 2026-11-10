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

        @testset "the order of the quotient is resolved" begin
            a = Sequence(Taylor(2), [1.0, 0.1, 0.0])
            b = Sequence(Taylor(3), [2.0, 0.1, 0.0, 0.0]) # non-constant: goes through the grid path
            q = a / b
            @test order(space(q)) > 3 # a / b is no polynomial: its order exceeds those of a and b
            @test q(0.3) ≈ a(0.3) / b(0.3)
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
            # the images are resolved past the nine coefficients compared
            a = Sequence(Taylor(8), vcat([0.0, 1.0], zeros(7)))

            expected_exp   = [1 / factorial(k) for k ∈ 0:8]
            expected_cos   = [iseven(k) ? (-1.0)^(k÷2) / factorial(k) : 0.0 for k ∈ 0:8]
            expected_sin   = [isodd(k)  ? (-1.0)^((k-1)÷2) / factorial(k) : 0.0 for k ∈ 0:8]
            expected_cosh  = [iseven(k) ? 1 / factorial(k) : 0.0 for k ∈ 0:8]
            expected_sinh  = [isodd(k)  ? 1 / factorial(k) : 0.0 for k ∈ 0:8]

            @test coefficients(project(exp(a), Taylor(8)))  ≈ expected_exp  atol=1e-10
            @test coefficients(project(cos(a), Taylor(8)))  ≈ expected_cos  atol=1e-10
            @test coefficients(project(sin(a), Taylor(8)))  ≈ expected_sin  atol=1e-10
            @test coefficients(project(cosh(a), Taylor(8))) ≈ expected_cosh atol=1e-10
            @test coefficients(project(sinh(a), Taylor(8))) ≈ expected_sinh atol=1e-10
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
            # a plain sequence carries no error bound: its enclosures hold up to the aliasing, at the rounding level
            a = Sequence(Taylor(16), interval.(vcat([0.0, 1.0], zeros(15))))
            b = exp(a)
            @test all(isguaranteed, coefficients(b))
            @test all(k -> in_interval(1 / factorial(k), b[k]), 0:8)

            bc = cos(a)
            expected_cos = [iseven(k) ? (-1.0)^(k÷2) / factorial(k) : 0.0 for k ∈ 0:8]
            @test all(k -> in_interval(expected_cos[k+1], bc[k]), 0:8)
        end

        @testset "SymmetricSpace: the full lattice by default, the symmetry on request" begin
            # by default the image lives on the full lattice, where exp of an even function is even
            seven = evensym(Fourier(4, 1.0))
            ae = Sequence(seven, zeros(Float64, dimension(seven)))
            ae[0] = 1.0
            ae[1] = 0.1
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
        # Z₂ = ‖ū⁻²‖(2‖ū‖ + R) on the ball of radius R
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
        # the perturbation bound takes the largest of the 2^d corners of the ν̄-polyannulus
        function corner_bounds(seq, ν, ν̄, ε)
            ρ = interval(1) + interval(ε)
            circle = ρ * cispi(interval(-1, 1))
            transfer = prod((ν̄ .+ ν) ./ (ν̄ .- ν))
            W = [sup(RadiiPolynomial._contour(exp, seq + circle, μ) * transfer) for μ ∈ RadiiPolynomial._polyannulus_corners(ν̄)]
            return W, sup(RadiiPolynomial._perturbation_bound(exp, seq, ρ, ν, ν̄))
        end

        # the negative modes dominate: the corner 1/ν̄ carries the largest bound
        W, W_all = corner_bounds(Sequence(Fourier(2, 1.0), interval.([0.05, 0.3, 1.0, 0.0, 0.0])), (interval(1.2),), (interval(2.0),), 1e-2)
        @test W_all == maximum(W)
        @test W_all > W[1] # W[1] is the corner ν̄

        # the mode (-1, 1) dominates: the corner (1/ν̄, ν̄) carries the largest bound
        sT = Fourier(3, 1.0) ⊗ Fourier(3, 1.0)
        sq = Sequence(sT, zeros(Interval{Float64}, dimension(sT)))
        sq[(0,0)] = interval(1.0); sq[(-1,1)] = interval(0.3)
        sq[(1,0)] = interval(0.02); sq[(0,1)] = interval(0.02)
        W, W_all = corner_bounds(sq, (interval(1.3), interval(1.3)), (interval(2.0), interval(2.0)), 1e-6)
        @test W_all == maximum(W)
        @test W_all > 1.5 * W[1]
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
        # floating-point coefficients run the same pipeline, without rigor
        Xf = Ell1(GeometricWeight(1.2))
        af = InfiniteSequence(Sequence(Taylor(3), [1.5, 0.1, 0.01, 0.001]), Xf)
        for f ∈ (exp, x -> x^0.7) # entire, and a branch cut
            r = f(af)
            @test isfinite(total_error(r))
            @test total_error(r) ≥ 0
            @test eltype(r) === Float64
            @test !RadiiPolynomial._isguaranteed(r)
        end

        # with an input error the perturbation circle is sampled: the errors stay floats
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
        # Fourier and Chebyshev need ν > 1, Taylor admits ν = 1, entire functions included
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
        # the Cauchy estimates need a geometric weight; the identity weight is the rate ν = 1
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
        # an input error needs f analytic within 1 + ε of the image: here the branch point is closer
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
        # the radius returned is one the search evaluated
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
        # the radius returned passes the interval analyticity sweep
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
            ν̄ = RadiiPolynomial._optimize_decay(nl, sequence(r), sequence(a), (interval(1.3),), total_error(a), total_error(a))
            @test RadiiPolynomial._check_branch_cut_poles(nl, sequence(a), interval.(ν̄), total_error(a))
        end
    end

    @testset "InfiniteSequence: a search with nothing feasible left says so" begin
        # no feasible radius: the search throws
        X = Ell1(GeometricWeight(interval(1.3)))
        a = InfiniteSequence(Sequence(Taylor(3), interval.([1.5, 0.1, 0.01, 1e-3])), X)
        # a branch cut swallowing the whole plane leaves no radius feasible, at any sweep
        nl = Nonlinearity(x -> x^0.7, Complex{Interval{Float64}}[], interval(-Inf, Inf))
        @test_throws ArgumentError RadiiPolynomial._optimize_decay(nl, sequence(a), sequence(a), (interval(1.3),), total_error(a), total_error(a))
    end

    @testset "the plateau sets the order of the output" begin
        # the box of kept indices stops before the first noisy coefficient; the rest is charged to the tail
        a = Sequence(Taylor(3), [1.0, 0.1, 0.01, 0.001])
        ν̄ = (2.0,)
        C = maximum(μ -> RadiiPolynomial._contour(exp, a, μ), RadiiPolynomial._polyannulus_corners(ν̄))
        # a violator at k = 2 and a quiet, envelope-respecting coefficient at k = 3
        c = Sequence(Taylor(3), [1.0, 0.1, 3C / ν̄[1]^2, 1e-20])
        @test RadiiPolynomial._noise_trim(c, RadiiPolynomial.CauchyCeiling(C, ν̄), RadiiPolynomial.ChargeTail((1.5,) ./ ν̄)) == (1,)

        # a plateau reaching the constant term empties the box; `_alias_and_tail` must still price it
        c₀ = Sequence(Taylor(3), [2C, 0.0, 0.0, 0.0])
        N_v₀ = RadiiPolynomial._noise_trim(c₀, RadiiPolynomial.CauchyCeiling(C, ν̄), RadiiPolynomial.ChargeTail((1.5,) ./ ν̄))
        @test N_v₀ == (-1,)
        finite_alias, tail_alias = RadiiPolynomial._alias_and_tail(RadiiPolynomial._coef_fft_size(space(c₀), space(c₀)), (1.5,), ν̄, N_v₀)
        @test iszero(finite_alias)
        @test isfinite(C * (finite_alias + tail_alias))
    end

    @testset "InfiniteSequence: the finite/tail split is honest about the plateau" begin
        # coefficients below the noise floor belong to the tail: the result has the order of the resolved box
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
        # a direction carrying a single mode still gets a radius above ν
        sT = Fourier(2, 1.0) ⊗ Fourier(2, 1.0)
        sq = Sequence(sT, zeros(Interval{Float64}, dimension(sT)))
        sq[(0, 0)] = interval(1.0)
        sq[(1, 0)] = interval(0.01) # constant along the second direction: no decay to observe
        for ν ∈ (1.3, 1.5)
            X = Ell1((GeometricWeight(interval(ν)), GeometricWeight(interval(ν))))
            a = InfiniteSequence(sq, X)
            @test isfinite(sup(total_error(exp(a))))
        end

        # and with more modes per direction
        sT4 = Fourier(4, 1.0) ⊗ Fourier(4, 1.0)
        sq4 = Sequence(sT4, zeros(Interval{Float64}, dimension(sT4)))
        sq4[(0, 0)] = interval(1.0)
        sq4[(1, 0)] = interval(0.01)
        X4 = Ell1((GeometricWeight(interval(1.3)), GeometricWeight(interval(1.3))))
        @test isfinite(sup(total_error(exp(InfiniteSequence(sq4, X4)))))
    end

    @testset "InfiniteSequence: output resolved to a noise floor" begin
        # exp of the interpolant of log(2 + t/2) is the polynomial 2 + t/2, up to noise
        N = 16
        nodes = [-cospi(j / N) for j ∈ 0:N]
        a = interval(real(RadiiPolynomial.to_coef(reverse(log.(2 .+ nodes ./ 2)), Chebyshev(N))))
        X = Ell1(GeometricWeight(interval(1.2)))
        for codomain ∈ (Chebyshev(N), Chebyshev(2N))
            r = Nonlinearity(exp)(InfiniteSequence(a, X); codomain = codomain)
            @test sup(total_error(r)) < 1e-12
        end
    end

    @testset "InfiniteSequence: roots of b xᵖ = a share one proof" begin
        X = Ell1(GeometricWeight(interval(1.2)))
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
        # the error is only as guaranteed as the input
        ng = InfiniteSequence(Sequence(Chebyshev(1), [interval(2.0) * 1.0, interval(0.25)]), X) # a Float64 factor drops the flag
        for F ∈ (inv, sqrt, cbrt, y -> y / (y + interval(1.0)), y -> x / y)
            @test RadiiPolynomial._isguaranteed(F(x))
            @test !RadiiPolynomial._isguaranteed(F(ng))
            @test !isguaranteed(total_error(F(ng)))
        end
        # a quotient far smoother than 1 / b
        bs = Sequence(Fourier(120, 1.0), [(k == 0) + 0.5 * 0.6^abs(k) for k ∈ -120:120])
        Bs = InfiniteSequence(interval.(bs), Ell1(GeometricWeight(interval(1.05))))
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
        X = Ell1(GeometricWeight(interval(1.001)))
        @test sup(total_error(inv(InfiniteSequence(interval.(a), X)))) < 1e-8
        # a failed proof says so
        @test_throws ArgumentError inv(InfiniteSequence(interval.(a), X); codomain = Chebyshev(8))
    end

    @testset "InfiniteSequence: a symmetric input is read on the full lattice" begin
        ℱ = Fourier(4, 1.0)
        X = Ell1(GeometricWeight(interval(1.2)))
        pow = Nonlinearity(x -> x ^ 1.5, Complex{Interval{Float64}}[], interval(-Inf, 0))
        for (s, coeffs, F) ∈ ((evensym(ℱ), [1.0, 0.1, 0.02, 0.0, 0.0], exp), (oddsym(ℱ), [0.1, 0.02, 0.0, 0.0], sin), (evensym(ℱ), [2.0, 0.1, 0.02, 0.0, 0.0], pow))
            for ε ∈ (0.0, 1e-8)
                a = Sequence(s, interval.(coeffs))
                r = F(InfiniteSequence(a, X; total_error = interval(ε)))
                r_full = F(InfiniteSequence(Projection(ℱ) * a, X; total_error = interval(ε)))
                @test space(sequence(r)) == space(sequence(r_full))
                @test all(isequal_interval.(coefficients(sequence(r)), coefficients(sequence(r_full))))
                @test isequal_interval(total_error(r), total_error(r_full))
            end
        end
        𝒯² = Fourier(1, 1.0) ⊗ Fourier(1, 1.0)
        a = Sequence(d4sym(𝒯²), interval.([1.0, 0.05, 0.02]))
        X² = Ell1((GeometricWeight(interval(1.2)), GeometricWeight(interval(1.2))))
        r = exp(InfiniteSequence(a, X²))
        @test isequal_interval(total_error(r), total_error(exp(InfiniteSequence(Projection(𝒯²) * a, X²))))
    end

    @testset "the plateau of a tensor space is cut in one direction" begin
        # a noisy corner coefficient is left out along one direction only
        s = Chebyshev(16) ⊗ Chebyshev(4)
        c = Sequence(s, [1e-3 * 0.1^(k + l) for (k, l) ∈ indices(s)])
        c[(16, 4)] = 1.0
        N = RadiiPolynomial._noise_trim(c, RadiiPolynomial.CauchyCeiling(1.0, (4.0, 4.0)), RadiiPolynomial.ChargeTail((0.3, 0.3)))
        @test N[1] < 16 || N[2] < 4
        @test N ∈ ((15, 4), (16, 3))
    end

    @testset "InfiniteSequence: a function not real on the real axis keeps its imaginary part" begin
        nl = Nonlinearity(x -> exp(im * x), Complex{Interval{Float64}}[], emptyinterval())
        X = Ell1(GeometricWeight(interval(1.2)))
        r = nl(InfiniteSequence(Sequence(Chebyshev(1), interval.([0.0, 0.25])), X); codomain = Chebyshev(12))
        @test eltype(sequence(r)) == Complex{Interval{Float64}}
        @test abs(mid.(sequence(r))(0.3) - exp(0.15im)) ≤ sup(total_error(r)) + 1e-12 # a(0.3) = 0.15
        # the constant shortcut too
        r₀ = nl(InfiniteSequence(Sequence(Chebyshev(1), interval.([0.5, 0.0])), X); codomain = Chebyshev(1))
        @test mid(sequence(r₀)[0]) ≈ exp(0.5im)
    end

    @testset "InfiniteSequence: tensor space tail and aliasing cover the slabs" begin
        # exp(2z₁ + 0.01z₂) has the coefficients 2^k/k! 0.01^l/l!: its tail lies in the slab k > N, l ≤ N
        s = Taylor(5) ⊗ Taylor(5)
        a = Sequence(s, zeros(Interval{Float64}, dimension(s)))
        a[(1,0)] = interval(2.0)
        a[(0,1)] = interval(0.01)
        X = Ell1((GeometricWeight(interval(1.2)), GeometricWeight(interval(1.2))))
        r = exp(InfiniteSequence(a, X))
        N = order(space(sequence(r)))
        c(k, l) = 2.0^k / factorial(big(k)) * 0.01^l / factorial(big(l)) * 1.2^(k+l)
        true_tail = Float64(sum(c(k, l) for k ∈ 0:60, l ∈ 0:60 if k > N[1] || l > N[2]))
        @test sup(tail_error(r)) ≥ true_tail
        @test sup(total_error(r)) ≥ true_tail
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
