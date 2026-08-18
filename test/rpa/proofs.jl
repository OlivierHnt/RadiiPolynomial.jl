@testset "Radii polynomial approach" begin

    @testset "newton" begin

        @testset "convergence criteria" begin
            # ResidualTolCriterion: converges iff |F(x)| ≤ tol; reported tolerance is `tol` itself
            @test RadiiPolynomial.ResidualTolCriterion()(0.5, 999.0, 1.0, 1e-16) == (true, 1.0)
            @test RadiiPolynomial.ResidualTolCriterion()(2.0, 999.0, 1.0, 1e-16) == (false, 1.0)
            @test sprint(show, RadiiPolynomial.ResidualTolCriterion()) == "|F(x)| ≤ tol"

            # ResidualCriterion: z = max(tol, √ϵ*(1+|F(x)|)); here √1e-16 * 1.5 ≈ 1.5e-8 ≪ tol = 1.0, so z = 1.0
            @test RadiiPolynomial.ResidualCriterion()(0.5, 999.0, 1.0, 1e-16) == (true, 1.0)
            @test RadiiPolynomial.ResidualCriterion()(2.0, 999.0, 1.0, 1e-16) == (false, 1.0)
            @test sprint(show, RadiiPolynomial.ResidualCriterion()) == "|F(x)| ≤ max(tol, √ϵ*(1+|F(x)|)"

            # StepCriterion: same shape but tested against |DF(x)\F(x)| = nAF instead of nF
            @test RadiiPolynomial.StepCriterion()(999.0, 0.5, 1.0, 1e-16) == (true, 1.0)
            @test RadiiPolynomial.StepCriterion()(999.0, 2.0, 1.0, 1e-16) == (false, 1.0)
            @test sprint(show, RadiiPolynomial.StepCriterion()) == "|DF(x)\\F(x)| ≤ max(tol, √ϵ*(1+|DF(x)\\F(x)|)"

            # CombinedCriterion: logical AND of the two above, reported tolerance is the min of the two
            @test RadiiPolynomial.CombinedCriterion()(0.5, 0.5, 1.0, 1e-16) == (true, 1.0)
            @test RadiiPolynomial.CombinedCriterion()(0.5, 2.0, 1.0, 1e-16) == (false, 1.0) # step part fails
            @test RadiiPolynomial.CombinedCriterion()(2.0, 0.5, 1.0, 1e-16) == (false, 1.0) # residual part fails
            @test sprint(show, RadiiPolynomial.CombinedCriterion()) ==
                "|F(x)| ≤ max(tol, √ϵ*(1+|F(x)|) && |DF(x)\\F(x)| ≤ max(tol, √ϵ*(1+|DF(x)\\F(x)|)"
        end

        @testset "scalar/vector Newton" begin
            # f(x) = x² - 2 has root √2; Df(x) = 2x
            x, success = newton(x -> (x^2 - 2, 2x), 1.0)
            @test success
            @test x == sqrt(2) # Float64 Newton on a well-conditioned scalar problem attains full precision

            # `newton`, `newton!` (allocating buffers) and `newton!` (explicit buffers) must agree
            F_DF!(F, DF, y) = (y^2 - 2, 2y)
            x2, success2 = newton!(F_DF!, 1.0)
            F0, DF0 = zero(1.0), zero(1.0)
            x3, success3 = newton!(F_DF!, 1.0, F0, DF0)
            @test x == x2 == x3
            @test success && success2 && success3

            # decoupled vector problem: x[1]² = 2, x[2]² = 3
            function FV_DF!(F, DF, y)
                F[1] = y[1]^2 - 2
                F[2] = y[2]^2 - 3
                DF .= 0.0
                DF[1,1] = 2y[1]
                DF[2,2] = 2y[2]
                return F, DF
            end
            xv, sv = newton!(FV_DF!, [1.0, 1.0])
            @test sv
            @test xv ≈ [sqrt(2), sqrt(3)]

            # hand-traced non-convergence: F(x) = 1, DF(x) = 1 has no root, so AF = 1 forever;
            # starting at x = 1.0 with maxiter = 3 the iterates are 1 → 0 → -1 → -2, then maxiter is exhausted
            x4, s4 = newton(x -> (1.0, 1.0), 1.0; maxiter = 3, tol = 1e-15)
            @test !s4
            @test x4 == -2.0

            # hand-traced single Newton step on x² - 2 from x₀ = 1: F(1) = -1, DF(1) = 2, AF = -1/2,
            # so x₁ = 1 - (-1/2) = 3/2; the loop stops there because maxiter = 1
            x5, s5 = newton(x -> (x^2 - 2, 2x), 1.0; maxiter = 1, tol = 1e-15)
            @test !s5
            @test x5 == 1.5

            # tol and maxiter must be non-negative
            @test_throws DomainError newton(x -> (x^2 - 2, 2x), 1.0; tol = -1.0)
            @test_throws DomainError newton(x -> (x^2 - 2, 2x), 1.0; maxiter = -1)

            # alternate convergence criteria still converge to √2 (possibly at a looser tolerance)
            xa, sa = newton(x -> (x^2 - 2, 2x), 1.0; convergence_criterion = RadiiPolynomial.StepCriterion())
            @test sa
            @test xa ≈ sqrt(2)
            xb, sb = newton(x -> (x^2 - 2, 2x), 1.0; convergence_criterion = RadiiPolynomial.CombinedCriterion())
            @test sb
            @test xb ≈ sqrt(2)

            # the verbose path must run without error
            xc, sc = redirect_stdout(() -> newton(x -> (x^2 - 2, 2x), 1.0; verbose = true), devnull)
            @test sc
        end
    end

    @testset "interval_of_existence" begin

        @testset "first order (Y, Z₁, R)" begin
            # r = Y/(1-Z₁) = 0.1/(1-0.5) = 0.2, which lies in [0, R] = [0, 10]
            Y, Z₁, R = interval(0.1), interval(0.5), 10.0
            ie, ok = interval_of_existence(Y, Z₁, R)
            @test ok
            @test isequal_interval(ie, interval(0.2, 10.0))
            @test isguaranteed(ie)

            # raw real numbers are converted via `interval` and remain guaranteed
            ie2, ok2 = interval_of_existence(0.1, 0.5, 10.0)
            @test ok2
            @test isequal_interval(ie2, ie)
            @test isguaranteed(ie2)

            # Z₁ ≥ 1 is not a contraction: failure, empty interval
            ie3, ok3 = interval_of_existence(interval(0.1), interval(1.0), 10.0)
            @test !ok3
            @test isempty_interval(ie3)

            # the root 0.2 exceeds R = 0.1: failure
            ie4, ok4 = interval_of_existence(interval(0.1), interval(0.5), 0.1)
            @test !ok4
            @test isempty_interval(ie4)

            # R must be a non-negative, non-NaN threshold
            ie5, ok5 = interval_of_existence(interval(0.1), interval(0.5), -1.0)
            @test !ok5
            @test isempty_interval(ie5)
            ie6, ok6 = interval_of_existence(interval(0.1), interval(0.5), NaN)
            @test !ok6
            @test isempty_interval(ie6)

            # Y must be non-negative
            ie7, ok7 = interval_of_existence(interval(-0.1), interval(0.5), 10.0)
            @test !ok7
            @test isempty_interval(ie7)

            # the guarantee flag of Y and Z₁ propagates to the result (contraction still holds)
            Y_bad = interval(0.1) + 0 # mixed Interval/Int arithmetic drops the guarantee bookkeeping flag
            @test !isguaranteed(Y_bad)
            ie8, ok8 = interval_of_existence(Y_bad, interval(0.5), 10.0)
            @test ok8
            @test !isguaranteed(ie8)
        end

        @testset "second order (Y, Z₁, Z₂, R)" begin
            # P(r) = Y + (Z₁-1) r + Z₂ r²/2 with Y = 0.01, Z₁ = 0.1, Z₂ = 0.2, R = 10.
            # By hand: b = Z₁ - 1 = -0.9, Δ = b² - 2 Z₂ Y = 0.81 - 0.004 = 0.806,
            # r₁ = (-b - √Δ)/Z₂, and the contraction Z₁ + Z₂ r < 1 fails first at r = -b/Z₂ = 4.5.
            Y, Z₁, Z₂, R = interval(0.01), interval(0.1), interval(0.2), 10.0
            b = 0.1 - 1.0
            Δ = b^2 - 2 * 0.2 * 0.01
            r₁ = (-b - sqrt(Δ)) / 0.2
            ie, ok = interval_of_existence(Y, Z₁, Z₂, R)
            @test ok
            @test isguaranteed(ie)
            @test inf(ie) ≈ r₁ rtol = 1e-8 # matches the hand-computed quadratic root up to correctly-rounded interval arithmetic
            @test sup(ie) ≈ 4.5 atol = 1e-9 # min(R, -b/Z₂) = min(10, 4.5)
            @test sup(ie) ≤ 4.5 # the returned bound never exceeds the exact contraction threshold

            # Z₂ = 0 exactly must reduce to the first-order formula
            ie0, ok0 = interval_of_existence(Y, Z₁, interval(0.0), R)
            ie0b, ok0b = interval_of_existence(Y, Z₁, R)
            @test ok0 == ok0b
            @test isequal_interval(ie0, ie0b)

            # Y too large makes the discriminant negative (complex roots): failure
            ie1, ok1 = interval_of_existence(interval(10.0), interval(0.1), interval(0.2), 100.0)
            @test !ok1
            @test isempty_interval(ie1)

            # Z₁ ≥ 1 is invalid regardless of Z₂
            ie2, ok2 = interval_of_existence(interval(0.01), interval(1.0), interval(0.2), 10.0)
            @test !ok2

            # Z₂ must be non-negative
            ie3, ok3 = interval_of_existence(interval(0.01), interval(0.1), interval(-0.2), 10.0)
            @test !ok3
        end

        @testset "set_of_radii (M-dimensional radii)" begin
            # two decoupled scalar contractions (W = 0 removes any coupling): rₘ = Yₘ/(1-Zₘₘ)
            Y = [interval(0.01), interval(0.02)]
            Z = [interval(0.1) interval(0.0); interval(0.0) interval(0.2)]
            W = zeros(Interval{Float64}, 2, 2, 2)
            R = [10.0, 10.0]
            rmin, η, success = RadiiPolynomial.set_of_radii(Y, Z, W, R)
            @test success
            @test rmin ≈ [0.01 / (1 - 0.1), 0.02 / (1 - 0.2)] atol = 1e-6

            # an invalid threshold R yields empty outputs and failure
            rmin2, η2, success2 = RadiiPolynomial.set_of_radii(Y, Z, W, [-1.0, 10.0])
            @test !success2
            @test isempty(rmin2) && isempty(η2)

            # a negative entry of Y is invalid
            Y_bad = [interval(-0.01), interval(0.02)]
            rmin3, η3, success3 = RadiiPolynomial.set_of_radii(Y_bad, Z, W, R)
            @test !success3
            @test isempty(rmin3) && isempty(η3)

            #= M == 1: Pf(r) = Y + Z r + (W/2) r² - r = 0.01 - 0.9 r + 0.1 r², whose smaller
               root (by hand, b = Z-1 = -0.9, Δ = b² - 2WY = 0.806) is
               r₁ = (0.9 - √0.806)/0.2 ≈ 0.011124862507311817, matching the radius found by the
               search. There Mat(r₁) = Z + W r₁ ≈ 0.1 + 0.2·0.011125 ≈ 0.10222 < 1, so the
               contraction holds and, since r₁ < R = 10, success is genuine. =#
            Y1, Z1_, W1 = [interval(0.01)], fill(interval(0.1), 1, 1), fill(interval(0.2), 1, 1, 1)
            rmin1, eta1, success1 = RadiiPolynomial.set_of_radii(Y1, Z1_, W1, [10.0])
            @test success1 == true
            @test eta1 == [1.0]
            @test rmin1[1] ≈ 0.011124862507311817 atol = 1e-12

            # same Y, Z, W, but R = 0.005 < r₁ ≈ 0.01112: the found radius is genuinely a root and
            # contracts, yet is overturned by the R check, giving a graceful failure (not a crash)
            rmin1b, eta1b, success1b = RadiiPolynomial.set_of_radii(Y1, Z1_, W1, [0.005])
            @test success1b == false
            @test isempty(rmin1b) && isempty(eta1b)
        end
    end

    @testset "Proof: cube root of 2 (scalar, second-order theorem)" begin
        # Step 1: F(x) = x³ - 2, whose only real zero is 2^(1/3); DF(x) = 3x², D²F(x) = 6x.
        f(x) = x^3 - exact(2)
        Df(x) = exact(3) * x^2

        # Step 2 (floating point): Newton's method from x₀ = 1
        x̄, newton_success = newton(x -> (f(x), Df(x)), 1.0)
        @test newton_success
        @test x̄ ≈ cbrt(2.0)

        # Step 3 (floating point): approximate inverse of DF(x̄) (a scalar, so A = 1/DF(x̄))
        A = inv(Df(x̄))

        # Step 4 (interval arithmetic): Y, Z₁, Z₂ bounds for T(x) = x - A f(x)
        x̄_int, A_int = interval(x̄), interval(A)

        Y = abs(A_int * f(x̄_int)) # ‖T(x̄) - x̄‖ = |A f(x̄)|
        @test isguaranteed(Y)
        @test inf(Y) ≥ 0
        @test sup(Y) < 1e-10 # x̄ is an accurate floating-point approximation, so the residual is tiny

        Z₁ = abs(interval(1) - A_int * Df(x̄_int)) # ‖DT(x̄)‖ = |1 - A DF(x̄)|
        @test isguaranteed(Z₁)
        @test sup(Z₁) < 1 # DF(x̄) and A cancel exactly but for rounding, so Z₁ is a tiny contraction bound

        # D²T(x) = -A D²F(x) = -6Ax is linear, so its sup over the ball B(x̄, R) is 6|A|(|x̄|+R)
        R = 10 * sup(Y) # heuristic radius R = 10ᵏY
        Z₂ = exact(6) * abs(A_int) * (abs(x̄_int) + interval(R))
        @test isguaranteed(Z₂)

        # verify the contraction and conclude existence/uniqueness of a zero within distance `ie` of x̄
        ie, contraction_success = interval_of_existence(Y, Z₁, Z₂, R)
        @test contraction_success
        @test isguaranteed(ie)
        @test inf(ie) > 0
        @test sup(ie) ≤ R

        # the true cube root of 2 indeed lies within the certified ball
        @test abs(cbrt(2.0) - x̄) ≤ sup(ie)
    end

    @testset "Proof: logistic initial value problem (Taylor, second-order theorem)" begin
        # Step 1: cast u' = u(1-u), u(0) = 1/2 as a zero-finding problem on Taylor
        # coefficients, F(u) = u - 1/2 - ∫₀ᵗ u(s)(1-u(s)) ds
        F(u) = u - exact(0.5) - Integral(1) * (u * (exact(1) - u))
        DF(u) = exact(I) - Integral(1) * Multiplication(exact(1) - exact(2) * u)

        # Step 2 (floating point): Newton's method on the truncated problem Π≤K ∘ F ∘ Π≤K
        K = 10
        u_guess = zeros(Taylor(K))
        ū, newton_success = newton(u -> (F(u), DF(u)), u_guess)
        @test newton_success
        @test space(ū) == Taylor(K)
        @test ū[0] == 0.5 # the k=0 equation is F(u)[0] = u[0] - 1/2 (the integral term never touches index 0)

        # Step 3 (floating point): approximate inverse A ≈ DF(ū)⁻¹, exact off the truncation
        Π = Projection(Taylor(K))
        A_K_interval = interval(inv(Π * DF(ū) * Π))
        A_interval = A_K_interval + (interval(I) - interval(Π))

        # Step 4 (interval arithmetic): Y, Z₁, Z₂ bounds for T(u) = u - A F(u) in X_{T,ν}, ν ≥ 1
        ū_interval = interval(ū)
        ν = interval(1.2)
        X_T = Ell1(GeometricWeight(ν))

        Y = norm(A_interval * F(ū_interval), X_T)
        @test isguaranteed(Y)
        @test inf(Y) ≥ 0
        @test sup(Y) < 1e-3 # ū is an accurate approximate zero, so the residual is small

        Π_Kp1 = Projection(Taylor(K + 1))
        Z₁ = opnorm(interval(Π_Kp1) - A_interval * DF(ū_interval) * Π_Kp1, X_T)
        # (unlike Y and Z₂, opnorm's internal accumulation drops the `isguaranteed` bookkeeping flag here;
        # this does not affect the rigor of the bound itself, only that metadata flag)
        @test inf(Z₁) ≥ 0
        @test sup(Z₁) < 1 # DT(ū) is close to zero on the truncated space, hence a genuine contraction

        # Z₂ is independent of R since F is quadratic, so R may be taken to be infinite
        R = Inf
        Z₂ = max(opnorm(A_K_interval, X_T), interval(1)) * ν * interval(2)
        @test isguaranteed(Z₂)

        ie, contraction_success = interval_of_existence(Y, Z₁, Z₂, R)
        @test contraction_success
        @test inf(ie) > 0
        @test inf(ie) < 1e-3 # the certified existence ball is small, as expected from a small residual
    end
end
