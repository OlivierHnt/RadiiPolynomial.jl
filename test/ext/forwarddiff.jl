using ForwardDiff

@testset "ForwardDiff extension" begin

    # central finite differences, to cross-check the dual-number Jacobian
    function _fd_jacobian(f, c; h = 1e-6)
        return hcat([(f(c .+ h .* (eachindex(c) .== i)) .- f(c .- h .* (eachindex(c) .== i))) ./ (2h) for i ∈ eachindex(c)]...)
    end

    @testset "FFT tables lift to dual numbers" begin
        D = ForwardDiff.Dual{Nothing,Float64,2}
        W = RadiiPolynomial._twiddle_table(Complex{D}, 8)
        @test eltype(W) == Complex{D}
        @test ForwardDiff.value.(W) == RadiiPolynomial._twiddle_table(ComplexF64, 8)
        @test all(iszero, ForwardDiff.partials.(real.(W)))
        w, kernel = RadiiPolynomial._chirp_table(Complex{D}, 5)
        @test ForwardDiff.value.(w) == RadiiPolynomial._chirp_table(ComplexF64, 5)[1]
        @test ForwardDiff.value.(kernel) == RadiiPolynomial._chirp_table(ComplexF64, 5)[2]
    end

    @testset "nonlinearities on Taylor / Fourier / Chebyshev" begin
        for s ∈ (Taylor(6), Chebyshev(6)), f ∈ (exp, sin, cos, sqrt, inv)
            g(c) = coefficients(f(Sequence(s, c)))
            c = [2.0 ; 0.1 * (1:dimension(s)-1) ./ dimension(s)]
            @test ForwardDiff.jacobian(g, c) ≈ _fd_jacobian(g, c) atol = 1e-6
        end
        # complex Fourier coefficients: split into real and imaginary parts
        s = Fourier(3, 1.0)
        n = dimension(s)
        g_fourier(c) = (b = exp(Sequence(s, complex.(c[1:n], c[n+1:end]))); [real.(coefficients(b)) ; imag.(coefficients(b))])
        c = 0.1 * (1:2n) ./ 2n
        @test ForwardDiff.jacobian(g_fourier, c) ≈ _fd_jacobian(g_fourier, c) atol = 1e-6
    end

    @testset "Picard map of an initial value problem" begin
        # u' = τ (exp(u) - u²), u(0) = u₀, as a fixed-point problem on Taylor coefficients
        s = Taylor(10)
        u₀ = 0.3
        τ = 0.1
        function F(c)
            u = Sequence(s, c)
            g = project(exp(u) - project(u^2, s), s)
            return coefficients(u - u₀ - τ * project(integrate(g), s))
        end
        c = zeros(dimension(s))
        c[1] = u₀
        @test ForwardDiff.jacobian(F, c) ≈ _fd_jacobian(F, c) atol = 1e-6
        x, converged = newton(c -> (F(c), ForwardDiff.jacobian(F, c)), c; verbose = false)
        @test converged
        u = Sequence(s, x)
        @test u(0.0) ≈ u₀
        @test differentiate(u)(0.5) ≈ τ * (exp(u(0.5)) - u(0.5)^2) atol = 1e-10
    end

end
