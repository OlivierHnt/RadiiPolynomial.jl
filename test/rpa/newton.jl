_newton_stdout(f) = mktemp() do path, io
    redirect_stdout(f, io)
    flush(io)
    return read(path, String)
end

@testset "Newton" begin

    # a criterion maps (|F(x)|, |DF(x)\F(x)|, tol, ϵ) to (converged, tolerance used)
    @testset "convergence criteria" begin
        residual_tol = RadiiPolynomial.ResidualTolCriterion()
        residual = RadiiPolynomial.ResidualCriterion()
        step = RadiiPolynomial.StepCriterion()
        combined = RadiiPolynomial.CombinedCriterion()

        @test residual_tol(0.5, 999.0, 1.0, 1e-16) == (true, 1.0)
        @test residual_tol(2.0, 0.0, 1.0, 1e-16) == (false, 1.0)

        # the tolerance used is max(tol, √ϵ (1 + n)) with n the residual or the step
        z_a = max(1e-12, sqrt(1e-16) * (1 + 0.5))
        z_b = max(1e-12, sqrt(1e-16) * (1 + 1e-10))
        @test residual(0.5, 0.0, 1e-12, 1e-16) == (false, z_a)
        @test residual(1e-10, 0.0, 1e-12, 1e-16) == (true, z_b)
        @test step(0.0, 0.5, 1e-12, 1e-16) == (false, z_a)
        @test step(999.0, 0.5, 1.0, 1e-16) == (true, 1.0)

        # both must hold, and the tolerance used is the smallest of the two
        @test combined(1e-10, 1e-10, 1e-12, 1e-16) == (true, z_b)
        @test combined(0.5, 1e-10, 1e-12, 1e-16) == (false, min(z_a, z_b))
        @test combined(0.5, 2.0, 1.0, 1e-16) == (false, 1.0)
        @test combined(2.0, 0.5, 1.0, 1e-16) == (false, 1.0)

        @test sprint(show, residual_tol) == "|F(x)| ≤ tol"
        @test sprint(show, residual) == "|F(x)| ≤ max(tol, √ϵ*(1+|F(x)|)"
        @test sprint(show, step) == "|DF(x)\\F(x)| ≤ max(tol, √ϵ*(1+|DF(x)\\F(x)|)"
        @test sprint(show, combined) == "|F(x)| ≤ max(tol, √ϵ*(1+|F(x)|) && |DF(x)\\F(x)| ≤ max(tol, √ϵ*(1+|DF(x)\\F(x)|)"
    end

    @testset "scalar x² - 2" begin
        F_DF(x) = (x^2 - 2, 2x)
        F_DF!(F, DF, x) = F_DF(x)

        x, success = newton(F_DF, 1.0)
        @test success
        @test x == sqrt(2)
        @test (x, success) isa Tuple{Float64,Bool}

        # `newton!` with allocated or given buffers agrees with `newton`
        @test newton!(F_DF!, 1.0) == newton!(F_DF!, 1.0, 0.0, 0.0) == (x, true)

        y, success_y = newton(F_DF, 10.0)
        @test success_y
        @test y ≈ sqrt(2) atol = 1e-12

        for criterion ∈ (RadiiPolynomial.StepCriterion(), RadiiPolynomial.CombinedCriterion())
            y, success_y = newton(F_DF, 1.0; convergence_criterion = criterion)
            @test success_y
            @test y ≈ sqrt(2) atol = 1e-10
        end

        @testset "maxiter" begin
            # two hand-unrolled steps xₙ₊₁ = xₙ - f(xₙ) / f'(xₙ) from a start far from the root
            x₀ = 100.0
            x₁ = x₀ - (x₀^2 - 2) / (2x₀)
            x₂ = x₁ - (x₁^2 - 2) / (2x₁)
            @test newton(F_DF, x₀; maxiter = 2) == (x₂, false)
            @test newton(F_DF, x₀; maxiter = 0) == (x₀, false)
            y, success_y = newton(F_DF, x₀; maxiter = 15)
            @test success_y
            @test y ≈ sqrt(2) atol = 1e-12

            # F = 1 and DF = 1 have no root: the iterates are 1, 0, -1, -2
            @test newton(x -> (1.0, 1.0), 1.0; maxiter = 3, tol = 1e-15) == (-2.0, false)
        end

        @test_throws DomainError newton(F_DF, 1.0; tol = -1.0)
        @test_throws DomainError newton(F_DF, 1.0; maxiter = -1)
        @test_throws DomainError newton(F_DF, 1.0; tol = -1.0, maxiter = -1)

        @test isempty(_newton_stdout(() -> newton(F_DF, 1.0; verbose = false)))
        loud = _newton_stdout(() -> newton(F_DF, 1.0; verbose = true))
        @test occursin("Newton's method", loud)
        @test occursin("Iteration", loud)
    end

    # the decoupled equations x₁² = 2 and x₂² = 3
    @testset "newton!: in-place problems" begin
        function F_DF!(F, DF, x)
            F[1] = x[1]^2 - 2
            F[2] = x[2]^2 - 3
            DF .= 0.0
            DF[1,1] = 2x[1]
            DF[2,2] = 2x[2]
            return F, DF
        end

        v, success_v = newton!(F_DF!, [1.0, 1.0])
        @test success_v
        @test v ≈ [sqrt(2), sqrt(3)]

        function F_DF_seq!(F, DF, x)
            F_DF!(coefficients(F), coefficients(DF), coefficients(x))
            return F, DF
        end

        # the initial guess is mutated and returned
        a = Sequence(Taylor(1), [1.0, 1.0])
        x, success = newton!(F_DF_seq!, a)
        @test success
        @test x === a
        @test coefficients(x) ≈ [sqrt(2), sqrt(3)] atol = 1e-12

        # buffers filled with `Inf` to catch an entry left unwritten
        b = Sequence(Taylor(1), [1.0, 1.0])
        x, success = newton!(F_DF_seq!, b, fill(Inf, Taylor(1)), fill(Inf, Taylor(1), Taylor(1)))
        @test success
        @test x === b
        @test coefficients(b) ≈ [sqrt(2), sqrt(3)] atol = 1e-12
    end

    @testset "_similar and _similar_linop" begin
        v = [1.0, 2.0, 3.0]
        a = Sequence(Taylor(1), [1.0, 2.0])
        @test RadiiPolynomial._similar(3.0) === 0.0
        @test RadiiPolynomial._similar(v) isa Vector{Float64} && length(RadiiPolynomial._similar(v)) == 3
        @test RadiiPolynomial._similar_linop(3.0) === 0.0
        @test size(RadiiPolynomial._similar_linop(v)) == (3, 3)
        La = RadiiPolynomial._similar_linop(a)
        @test La isa LinearOperator
        @test domain(La) == codomain(La) == space(a)
    end

    # `F_DF(x)` may return its values on a larger space than the one of `x`, on which they are projected
    @testset "_copy_maybeinplace! and _sub_maybeinplace!" begin
        𝒯₁, 𝒯₂ = Taylor(1), Taylor(2)
        y = Sequence(𝒯₂, [1.0, 2.0, 3.0])
        D = LinearOperator(𝒯₁, 𝒯₂, [1.0 2.0; 3.0 4.0; 5.0 6.0])

        @test RadiiPolynomial._copy_maybeinplace!(0.0, 5.0) == 5.0
        @test RadiiPolynomial._copy_maybeinplace!([Inf, Inf], [1.0, 2.0]) == [1.0, 2.0]
        @test coefficients(RadiiPolynomial._copy_maybeinplace!(fill(Inf, 𝒯₁), y)) == [1.0, 2.0]
        @test coefficients(RadiiPolynomial._copy_maybeinplace!(fill(Inf, 𝒯₁, 𝒯₁), D)) == [1.0 2.0; 3.0 4.0]

        x() = Sequence(𝒯₁, [10.0, 20.0])
        A() = LinearOperator(𝒯₁, 𝒯₁, [10.0 20.0; 30.0 40.0])
        @test RadiiPolynomial._sub_maybeinplace!(5.0, 2.0) == 3.0
        @test RadiiPolynomial._sub_maybeinplace!([10.0, 20.0], [1.0, 2.0]) == [9.0, 18.0]
        @test coefficients(RadiiPolynomial._sub_maybeinplace!(x(), Sequence(𝒯₁, [1.0, 2.0]))) == [9.0, 18.0]
        @test coefficients(RadiiPolynomial._sub_maybeinplace!(x(), y)) == [9.0, 18.0]
        @test coefficients(RadiiPolynomial._sub_maybeinplace!(A(), LinearOperator(𝒯₁, 𝒯₁, [1.0 2.0; 3.0 4.0]))) == [9.0 18.0; 27.0 36.0]
        @test coefficients(RadiiPolynomial._sub_maybeinplace!(A(), D)) == [9.0 18.0; 27.0 36.0]
    end

    #= A map and a Jacobian returned on the plain Fourier space are compressed onto the symmetric
       space of the unknown. The compression keeps the whole orbit of each coefficient, so a
       linear problem converges in a single step. =#
    @testset "map and Jacobian returned on the desymmetrized space" begin
        ℱ = Fourier(2, 1.0)
        es = evensym(ℱ)
        A = project(Multiplication(Sequence(ℱ, [0.0, 0.3, 1.0, 0.3, 0.0])), ℱ, ℱ)
        x_exact = Sequence(es, [1.0, 0.5, 0.2])
        b = A * x_exact
        calls = Ref(0)
        x, success = newton(zeros(es)) do u
            calls[] += 1
            return A * u - b, A
        end
        @test success
        @test x ≈ x_exact atol = 1e-12
        @test calls[] == 2 # one Newton step, then the convergence check
    end
end
