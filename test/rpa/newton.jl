@testset "Newton" begin

    #= Convergence criteria (RadiiPolynomial.ConvergenceCriterion subtypes) are plain
       functors of (nF, nAF, tol, ϵ) ↦ (converged::Bool, tolerance_used). Check the
       hand-derived formulas directly before relying on them inside `newton`/`newton!`. =#
    @testset "ConvergenceCriterion functors" begin
        rt = RadiiPolynomial.ResidualTolCriterion()
        @test rt(0.5, 999.0, 1.0, 1e-16) == (true, 1.0)   # 0.5 ≤ tol=1.0
        @test rt(2.0, 0.0, 1.0, 1e-16) == (false, 1.0)    # 2.0 > tol=1.0

        # z = max(tol, √ϵ*(1+nF)); with ϵ=1e-16, √ϵ = 1e-8 (up to rounding)
        rc = RadiiPolynomial.ResidualCriterion()
        z_a = max(1e-12, sqrt(1e-16) * (1 + 0.5))
        @test rc(0.5, 0.0, 1e-12, 1e-16) == (false, z_a)
        z_b = max(1e-12, sqrt(1e-16) * (1 + 1e-10))
        @test rc(1e-10, 0.0, 1e-12, 1e-16) == (true, z_b) # nF=1e-10 ≤ z_b

        # z = max(tol, √ϵ*(1+nAF)); nAF = 0.5
        sc = RadiiPolynomial.StepCriterion()
        @test sc(0.0, 0.5, 1e-12, 1e-16) == (false, z_a) # same z as z_a (same formula, same value)

        # both ResidualCriterion and StepCriterion must hold; combined value = min of the two z's
        cc = RadiiPolynomial.CombinedCriterion()
        @test cc(1e-10, 1e-10, 1e-12, 1e-16) == (true, min(z_b, z_b))
        @test cc(0.5, 1e-10, 1e-12, 1e-16) == (false, min(z_a, z_b))
    end

    #= newton(F_DF, x0): x² - 2 = 0 converging to √2. F_DF returns fresh (F, DF) each
       call (not in-place); the wrapper copies/projects them into internal buffers. =#
    @testset "newton: scalar x² - 2 → √2" begin
        F_DF(x) = (x^2 - 2, 2x)

        x, success = newton(F_DF, 1.0)
        @test success == true
        @test x ≈ sqrt(2) atol=1e-12
        @test (x, success) isa Tuple{Float64,Bool}

        # different starting guess, same root
        x2, success2 = newton(F_DF, 10.0)
        @test success2 == true
        @test x2 ≈ sqrt(2) atol=1e-12

        # custom convergence criterion threads through end-to-end
        x3, success3 = newton(F_DF, 1.0; convergence_criterion = RadiiPolynomial.StepCriterion())
        @test success3 == true
        @test x3 ≈ sqrt(2) atol=1e-10
    end

    @testset "newton: maxiter / non-convergence" begin
        F_DF(x) = (x^2 - 2, 2x)

        # Hand-unrolled Newton step: xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ), starting far from the root
        # so that 2 iterations do not reach tol = 1e-12.
        x0 = 100.0
        x1 = x0 - (x0^2 - 2) / (2x0)
        x2 = x1 - (x1^2 - 2) / (2x1)
        result, success = newton(F_DF, x0; maxiter = 2)
        @test success == false
        @test result == x2

        # maxiter = 0: only the initial residual is checked, x0 itself is returned unchanged
        result0, success0 = newton(F_DF, x0; maxiter = 0)
        @test success0 == false
        @test result0 == x0

        # maxiter large enough recovers convergence for the same start
        resultbig, successbig = newton(F_DF, x0; maxiter = 15)
        @test successbig == true
        @test resultbig ≈ sqrt(2) atol=1e-12
    end

    @testset "newton: DomainError on invalid tol/maxiter" begin
        F_DF(x) = (x^2 - 2, 2x)
        @test_throws DomainError newton(F_DF, 1.0; tol = -1.0)
        @test_throws DomainError newton(F_DF, 1.0; maxiter = -1)
        @test_throws DomainError newton(F_DF, 1.0; tol = -1.0, maxiter = -1)
    end

    @testset "newton: verbose flag silencing" begin
        F_DF(x) = (x^2 - 2, 2x)

        function capture_stdout(f)
            old = stdout
            rd, wr = redirect_stdout()
            f()
            redirect_stdout(old)
            close(wr)
            return read(rd, String)
        end

        silent = capture_stdout(() -> newton(F_DF, 1.0; verbose = false))
        @test isempty(silent)

        loud = capture_stdout(() -> newton(F_DF, 1.0; verbose = true))
        @test !isempty(loud)
        @test occursin("Newton's method", loud)
        @test occursin("Iteration", loud)
    end

    #= newton!(F_DF!, x0): a small in-place Sequence problem on Taylor(1) (2 coefficients).
       F_DF! solves the two independent equations a₀² = 2 and a₁² = 3 componentwise
       (i.e. two Babylonian sqrt Newton iterations running in parallel), so the exact
       root is (√2, √3). newton! (2-/4-arg, no leading `x0 = copy(...)`) mutates x0
       in place — the returned Sequence is the very same object. =#
    @testset "newton!: in-place Sequence problem" begin
        𝒯 = Taylor(1)
        target = Sequence(𝒯, [2.0, 3.0])

        function F_DF!(F, DF, x)
            cx, ct = coefficients(x), coefficients(target)
            coefficients(F) .= cx .^ 2 .- ct
            M = coefficients(DF)
            fill!(M, 0.0)
            M[1,1] = 2cx[1]
            M[2,2] = 2cx[2]
            return F, DF
        end

        a = Sequence(𝒯, [1.0, 1.0])
        xsol, success = newton!(F_DF!, a)
        @test success == true
        @test xsol === a # mutated in place, same object returned
        @test coefficients(xsol) ≈ [sqrt(2), sqrt(3)] atol=1e-12

        # explicit-buffer form, buffers seeded with Inf so unwritten entries would be caught
        b = Sequence(𝒯, [1.0, 1.0])
        Fbuf = Sequence(𝒯, [Inf, Inf])
        DFbuf = LinearOperator(𝒯, 𝒯, fill(Inf, 2, 2))
        xsol2, success2 = newton!(F_DF!, b, Fbuf, DFbuf)
        @test success2 == true
        @test xsol2 === b
        @test coefficients(b) ≈ [sqrt(2), sqrt(3)] atol=1e-12
    end

    @testset "Internal helpers: _similar / _similar_linop" begin
        @test RadiiPolynomial._similar(3.0) === 0.0 # zero(3.0)
        v = [1.0, 2.0, 3.0]
        sv = RadiiPolynomial._similar(v)
        @test sv isa Vector{Float64} && length(sv) == 3

        @test RadiiPolynomial._similar_linop(3.0) === 0.0
        Mv = RadiiPolynomial._similar_linop(v)
        @test size(Mv) == (3, 3)

        a = Sequence(Taylor(1), [1.0, 2.0])
        La = RadiiPolynomial._similar_linop(a)
        @test La isa LinearOperator
        @test domain(La) == codomain(La) == space(a)
        @test size(coefficients(La)) == (2, 2)
    end

    #= _copy_maybeinplace! is what lets the non-bang `newton` wrapper accept a user
       F_DF(x) that returns a Sequence/LinearOperator of a *different* (typically larger,
       e.g. grown by convolution) space than x: it projects the result down onto the
       buffer's space. Exercised directly here with hand-picked coefficients. =#
    @testset "Internal helpers: _copy_maybeinplace!" begin
        @test RadiiPolynomial._copy_maybeinplace!(0.0, 5.0) == 5.0 # Number: just returns y

        buf = [Inf, Inf]
        RadiiPolynomial._copy_maybeinplace!(buf, [1.0, 2.0])
        @test buf == [1.0, 2.0] # generic fallback: x .= y

        𝒯1, 𝒯2 = Taylor(1), Taylor(2)
        bufS = Sequence(𝒯1, [Inf, Inf])
        y = Sequence(𝒯2, [1.0, 2.0, 3.0]) # 1 + 2t + 3t²
        RadiiPolynomial._copy_maybeinplace!(bufS, y)
        @test coefficients(bufS) == [1.0, 2.0] # Projection(Taylor(1)) drops the t² term

        D1 = LinearOperator(𝒯1, 𝒯1, fill(Inf, 2, 2))
        Draw = LinearOperator(𝒯1, 𝒯2, [1.0 2.0; 3.0 4.0; 5.0 6.0]) # domain 𝒯1, codomain 𝒯2
        RadiiPolynomial._copy_maybeinplace!(D1, Draw)
        @test coefficients(D1) == [1.0 2.0; 3.0 4.0] # Projection(codomain=𝒯1) drops row 3
    end

    @testset "Internal helpers: _sub_maybeinplace!" begin
        @test RadiiPolynomial._sub_maybeinplace!(5.0, 2.0) == 3.0 # Number: x -= y

        buf = [10.0, 20.0]
        RadiiPolynomial._sub_maybeinplace!(buf, [1.0, 2.0])
        @test buf == [9.0, 18.0] # generic fallback: x .-= y

        𝒯1, 𝒯2 = Taylor(1), Taylor(2)
        x = Sequence(𝒯1, [10.0, 20.0])
        y = Sequence(𝒯1, [1.0, 2.0])
        RadiiPolynomial._sub_maybeinplace!(x, y) # same space: direct in-place subtraction
        @test coefficients(x) == [9.0, 18.0]

        x2 = Sequence(𝒯1, [10.0, 20.0])
        y2 = Sequence(𝒯2, [1.0, 2.0, 3.0]) # mismatched (larger) space, projected before subtracting
        RadiiPolynomial._sub_maybeinplace!(x2, y2)
        @test coefficients(x2) == [9.0, 18.0]

        D1 = LinearOperator(𝒯1, 𝒯1, [10.0 20.0; 30.0 40.0])
        D2 = LinearOperator(𝒯1, 𝒯1, [1.0 2.0; 3.0 4.0])
        RadiiPolynomial._sub_maybeinplace!(D1, D2) # same domain/codomain: direct subtraction
        @test coefficients(D1) == [9.0 18.0; 27.0 36.0]

        D3 = LinearOperator(𝒯1, 𝒯1, [10.0 20.0; 30.0 40.0])
        Draw = LinearOperator(𝒯1, 𝒯2, [1.0 2.0; 3.0 4.0; 5.0 6.0]) # mismatched codomain
        RadiiPolynomial._sub_maybeinplace!(D3, Draw)
        @test coefficients(D3) == [9.0 18.0; 27.0 36.0]
    end
end
