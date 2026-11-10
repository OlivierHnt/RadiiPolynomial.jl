@testset "Radii polynomial approach" begin

    @testset "cube root of 2 (scalar, second order)" begin
        f(x) = x^3 - exact(2)
        Df(x) = exact(3) * x^2

        x̄, newton_success = newton(x -> (f(x), Df(x)), 1.0)
        @test newton_success
        @test x̄ ≈ cbrt(2.0)

        # bounds for T(x) = x - A f(x) with A the approximate inverse of Df(x̄)
        x̄_interval, A = interval(x̄), interval(inv(Df(x̄)))
        Y = abs(A * f(x̄_interval))
        Z₁ = abs(interval(1) - A * Df(x̄_interval))
        # D²T(x) = -6 A x is bounded on the ball of radius R around x̄ by 6 |A| (|x̄| + R)
        R = 10 * sup(Y)
        Z₂ = exact(6) * abs(A) * (abs(x̄_interval) + interval(R))
        @test isguaranteed(Y) && isguaranteed(Z₁) && isguaranteed(Z₂)
        @test 0 ≤ inf(Y) && sup(Y) < 1e-10
        @test sup(Z₁) < 1

        ie, contraction_success = interval_of_existence(Y, Z₁, Z₂, R)
        @test contraction_success
        @test isguaranteed(ie)
        @test 0 < inf(ie) && sup(ie) ≤ R
        @test abs(cbrt(2.0) - x̄) ≤ sup(ie)
    end

    @testset "logistic initial value problem (Taylor, second order)" begin
        # u' = u (1 - u), u(0) = 1/2 as the zero of F(u) = u - 1/2 - ∫₀ᵗ u(s) (1 - u(s)) ds
        F(u) = u - exact(0.5) - Integral(1) * (u * (exact(1) - u))
        DF(u) = exact(I) - Integral(1) * Multiplication(exact(1) - exact(2) * u)

        K = 10
        ū, newton_success = newton(u -> (F(u), DF(u)), zeros(Taylor(K)))
        @test newton_success
        @test space(ū) == Taylor(K)
        @test ū[0] == 0.5 # the integral never touches the index 0

        # approximate inverse of DF(ū), equal to the identity off the truncation
        Π = Projection(Taylor(K))
        A_K = interval(inv(Π * DF(ū) * Π))
        A = A_K + (interval(I) - interval(Π))

        # bounds for T(u) = u - A F(u)
        ū_interval = interval(ū)
        ν = interval(1.2)
        X = Ell1(GeometricWeight(ν))
        Π₊ = Projection(Taylor(K + 1))
        Y = norm(A * F(ū_interval), X)
        Z₁ = opnorm(interval(Π₊) - A * DF(ū_interval) * Π₊, X)
        # F is quadratic, so Z₂ does not depend on R, which may be infinite
        Z₂ = max(opnorm(A_K, X), interval(1)) * ν * interval(2)
        @test isguaranteed(Y) && isguaranteed(Z₁) && isguaranteed(Z₂)
        @test 0 ≤ inf(Y) && sup(Y) < 1e-3
        @test 0 ≤ inf(Z₁) && sup(Z₁) < 1

        ie, contraction_success = interval_of_existence(Y, Z₁, Z₂, Inf)
        @test contraction_success
        @test 0 < inf(ie) < 1e-3
    end
end
