@testset "Shift" begin
    @testset "constructors and value accessor" begin
        𝒮 = Shift(1.0)
        @test 𝒮 isa Shift{Float64}
        @test value(𝒮) == 1.0

        𝒮₂ = Shift(1.0, 2.0)
        @test 𝒮₂ == Shift((1.0, 2.0))
        @test value(𝒮₂) == (1.0, 2.0)

        @test_throws ArgumentError Shift(())
        @test_throws ArgumentError Shift()
    end

    @testset "arithmetic on Shift operators" begin
        @test Shift(0.3) * Shift(0.7) == Shift(1.0)
        @test Shift((0.3, 0.5)) * Shift((0.2, 0.5)) == Shift((0.5, 1.0))
        @test Shift(0.3)^2 == Shift(0.6)
        @test Shift((0.3, 0.5))^2 == Shift((0.6, 1.0))
        @test Shift((0.3, 0.5))^(2, 3) == Shift((0.6, 1.5))
    end

    @testset "Fourier: phase rotation" begin
        a = Sequence(Fourier(1, 1.0), [0.5, 0.0, 0.5]) # cos(t)
        τ = π/2
        𝒮 = Shift(τ)
        @test domain(𝒮, Fourier(1, 1.0)) == Fourier(1, 1.0)
        @test codomain(𝒮, Fourier(1, 1.0)) == Fourier(1, 1.0)
        # cₖ = e^{iωτk} aₖ: c₋₁ = e^{-iπ/2}(0.5) = -0.5i, c₀ = 0, c₁ = e^{iπ/2}(0.5) = 0.5i
        expected = Sequence(Fourier(1, 1.0), ComplexF64[-0.5im, 0.0, 0.5im])
        out = Sequence(Fourier(1, 1.0), fill(complex(Inf), 3))
        @test shift(a, τ) ≈ 𝒮(a) ≈ project(𝒮, Fourier(1, 1.0), Fourier(1, 1.0), ComplexF64)(a) ≈
            shift!(copy(out), a, τ) ≈ mul!(out, 𝒮, a) ≈ expected

        # τ = 0: identity
        @test shift(a, 0.0) == a

        # shift! requires the destination space to match codomain(Shift(τ), space(a))
        @test_throws ArgumentError shift!(Sequence(Fourier(2, 1.0), fill(complex(Inf), 5)), a, τ)

        # periodicity: a full period leaves the sequence unchanged
        ω = 2.0
        s = Fourier(3, ω)
        b = Sequence(s, ComplexF64[0.1-0.1im, 0.2, 0.3+0.2im, 0.5, 0.3-0.2im, 0.2, 0.1+0.1im])
        @test shift(b, 2π/ω) ≈ b
    end

    @testset "Taylor: only τ = 0 is supported" begin
        a = Sequence(Taylor(2), [1.0, 2.0, 3.0])
        𝒮 = Shift(0.0)
        @test domain(𝒮, Taylor(2)) == Taylor(2)
        @test codomain(𝒮, Taylor(2)) == Taylor(2)
        out = Sequence(Taylor(2), fill(Inf, 3))
        @test shift(a, 0.0) == 𝒮(a) == project(𝒮, Taylor(2), Taylor(2), Float64)(a) ==
            shift!(copy(out), a, 0.0) == mul!(out, 𝒮, a) == a

        @test_throws DomainError shift(a, 1.0)
    end

    @testset "Chebyshev: only τ = 0 is supported" begin
        a = Sequence(Chebyshev(2), [1.0, 2.0, 3.0])
        𝒮 = Shift(0)
        @test domain(𝒮, Chebyshev(2)) == Chebyshev(2)
        @test codomain(𝒮, Chebyshev(2)) == Chebyshev(2)
        out = Sequence(Chebyshev(2), fill(Inf, 3))
        @test shift(a, 0) == 𝒮(a) == project(𝒮, Chebyshev(2), Chebyshev(2), Float64)(a) ==
            shift!(copy(out), a, 0) == mul!(out, 𝒮, a) == a

        @test_throws DomainError shift(a, 1.0)
    end

    @testset "TensorSpace: tuple form per factor" begin
        # Taylor factor left untouched (τ = 0), Fourier factor rotated
        s = Taylor(0) ⊗ Fourier(1, 1.0)
        a = Sequence(s, ComplexF64[1.0, 2.0, 3.0]) # indices (0,-1),(0,0),(0,1)
        τ = (0.0, π/2)
        c = shift(a, τ)
        # (0,-1): 1*e^{-iπ/2} = -i ; (0,0): 2*1 = 2 ; (0,1): 3*e^{iπ/2} = 3i
        expected = Sequence(s, ComplexF64[-1.0im, 2.0, 3.0im])
        @test c ≈ expected
        @test project(Shift(τ), s, s, ComplexF64)(a) ≈ c

        # both factors rotated
        s2 = Fourier(1, 1.0) ⊗ Fourier(1, 1.0)
        b = Sequence(s2, ComplexF64.(1:9))
        τ2 = (0.1, 0.2)
        cb = shift(b, τ2)
        # c[(k₁,k₂)] = e^{ik₁τ₁} e^{ik₂τ₂} a[(k₁,k₂)]
        expected2 = Sequence(s2, [b[(k1,k2)]*cis(k1*τ2[1])*cis(k2*τ2[2]) for (k1,k2) ∈ indices(s2)])
        @test cb ≈ expected2
        @test project(Shift(τ2), s2, s2, ComplexF64)(b) ≈ cb
    end

    @testset "interval τ" begin
        τ = interval(0.0)
        a = Sequence(Fourier(1, 1.0), Complex{Interval{Float64}}.([0.5, 0.0, 0.5]))
        @test shift(a, τ) == Shift(τ)(a) == a
    end
end
