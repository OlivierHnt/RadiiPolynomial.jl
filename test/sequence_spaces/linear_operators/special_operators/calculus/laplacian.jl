@testset "Laplacian" begin
    Δ = Laplacian()

    @testset "Fourier" begin
        s = Fourier(2, 1.0)
        @test domain(Δ, s) == codomain(Δ, s) == s

        # (Δa)ₖ = -ω²k² aₖ, materialized with real coefficients
        a = Sequence(s, [1.0, 2.0, 3.0, 2.0, 1.0])
        expected = Sequence(s, [-4.0, -2.0, 0.0, -2.0, -4.0])
        test_action(Δ, a, expected; inplace! = laplacian!)
        @test laplacian(a) == differentiate(a, 2)
        @test_throws ArgumentError laplacian!(fill(Inf, Fourier(1, 1.0)), a)

        # the action promotes the coefficient type to complex
        @test eltype(laplacian(a)) == ComplexF64
        @test laplacian(complex.(a)) == expected
        @test eltype(laplacian(complex.(a))) == ComplexF64
        a_c = Sequence(s, ComplexF64[1.0, 2.0 + 3.0im, 3.0, 2.0 - 3.0im, 1.0])
        @test laplacian(a_c) == differentiate(a_c, 2) ==
            Sequence(s, ComplexF64[-4.0-0.0im, -2.0-3.0im, 0.0+0.0im, -2.0+3.0im, -4.0-0.0im])
    end

    @testset "Taylor and Chebyshev" begin
        # on Taylor Δ is the second derivative, which has no domain on Chebyshev
        a = Sequence(Taylor(3), [1.0, 2.0, 3.0, 4.0])
        expected = Sequence(Taylor(1), [6.0, 24.0])
        @test laplacian(a) == differentiate(a, 2) == expected
        @test project(Δ, Taylor(3), codomain(Δ, Taylor(3)), Float64)(a) == expected
        @test laplacian(Sequence(Taylor(1), [1.0, 2.0])) == Sequence(Taylor(0), [0.0])
        @test domain(Δ, Chebyshev(3)) == UndefSpace()
    end

    @testset "Tensor space" begin
        # Δ is the sum of the second derivatives in each direction
        s = Fourier(1, 1.0) ⊗ Fourier(1, 1.0)
        b = Sequence(s, collect(1.0:9.0))
        expected = Sequence(s, ComplexF64[-2.0, -2.0, -6.0, -4.0, 0.0, -6.0, -14.0, -8.0, -18.0])
        @test laplacian(b) == differentiate(b, (2, 0)) + differentiate(b, (0, 2))
        test_action(Δ, b, expected; inplace! = laplacian!)

        bc = Sequence(s, ComplexF64[1+1im, 2, 3-2im, 4, 5, 6+3im, 7, 8, 9-1im])
        @test laplacian(bc) == differentiate(bc, (2, 0)) + differentiate(bc, (0, 2))

        # a Taylor factor of order 1 is annihilated by its second derivative
        s₂ = Taylor(1) ⊗ Fourier(2, 1.0)
        c = Sequence(s₂, collect(1.0:dimension(s₂)))
        @test differentiate(c, (2, 0)) == zeros(ComplexF64, Taylor(0) ⊗ Fourier(2, 1.0))
        @test laplacian(c) == differentiate(c, (0, 2))

        # Δ is diagonal with entries -(ω₁²k₁² + ω₂²k₂²)
        s₃ = Fourier(2, 1.0) ⊗ Fourier(2, 2.0)
        dom, codom = domain(Δ, s₃), codomain(Δ, s₃)
        @test RadiiPolynomial.getcoefficient(Δ, (codom, (1, 1)), (dom, (1, 1)), Float64) == -(1.0^2*1^2 + 2.0^2*1^2)
        @test RadiiPolynomial.getcoefficient(Δ, (codom, (2, 0)), (dom, (2, 0)), Float64) == -(1.0^2*2^2)
        @test RadiiPolynomial.getcoefficient(Δ, (codom, (1, 1)), (dom, (0, 1)), Float64) == 0.0
    end

    @testset "Cartesian space" begin
        s = Fourier(1, 1.0)^2
        a = Sequence(s, collect(1.0:6.0))
        @test laplacian(a) == Sequence(s, [-1.0, 0.0, -3.0, -4.0, 0.0, -6.0])
        @test component(laplacian(a), 1) == laplacian(component(a, 1))
        @test domain(Δ, s) == codomain(Δ, s) == s

        @test laplacian(Sequence(Taylor(1) × Fourier(1, 1.0), collect(1.0:5.0))) ==
            Sequence(Taylor(0) × Fourier(1, 1.0), ComplexF64[0.0, -3.0, 0.0, -5.0])
    end

    @testset "Symmetric space" begin
        sE, sO = evensym(Fourier(2, 1.0)), oddsym(Fourier(2, 1.0))
        @test domain(Δ, sE) == sE
        @test codomain(Δ, sO) == sO

        a = Sequence(sE, ComplexF64[1.0, 2.0, 3.0])
        test_action(Δ, a, Sequence(sE, [0.0, -2.0, -12.0]); inplace! = laplacian!)
        @test eltype(laplacian(a)) == ComplexF64

        aO = Sequence(sO, ComplexF64[2.0im, 6.0im])
        @test laplacian(aO) == differentiate(aO, 2) == Sequence(sO, ComplexF64[-2.0im, -24.0im])
    end

    @testset "Symmetric tensor space" begin
        # Δ keeps the symmetries whose lattice automorphism preserves the symbol -(ω₁²k₁² + ω₂²k₂²)
        t = d4sym(Fourier(2, 2.0) ⊗ Fourier(2, 2.0))
        @test domain(Δ, t) === codomain(Δ, t) === t
        @test domain(Δ, interval(t)) === interval(t)

        a = Sequence(t, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        @test Projection(desymmetrize(t)) * laplacian(a) == laplacian(Projection(desymmetrize(t)) * a)

        # the materialized operator is diagonal on the representatives
        M = project(Δ, t, t, Float64)
        for (i, k) ∈ enumerate(indices(t)), (j, l) ∈ enumerate(indices(t))
            @test coefficients(M)[i,j] == (k == l ? -4.0 * (k[1]^2 + k[2]^2) : 0.0)
        end

        # a swap of two factors with different frequencies breaks the symbol and is dropped
        swap = GroupElement(LatticeAut([0 1 ; 1 0]), Cocycle(0//1, Rational{Int}[0//1, 0//1]))
        sw = SymmetricSpace(Fourier(2, 1.0) ⊗ Fourier(2, 1.0), Group(swap))
        @test domain(Δ, sw) === sw
        swω = SymmetricSpace(Fourier(2, 1.0) ⊗ Fourier(2, 2.0), Group(swap))
        @test symmetry(domain(Δ, swω)) == symmetry(codomain(Δ, swω)) == RadiiPolynomial._trivial_symmetry(Val(2))
        @test desymmetrize(codomain(Δ, swω)) == desymmetrize(swω)
        b = Sequence(swω, [1.0 / (1 + sum(abs, k))^2 for k ∈ indices(swω)])
        @test project(laplacian(b), desymmetrize(swω)) == laplacian(project(b, desymmetrize(swω)))

        # a reflection in the same group preserves the symbol and is kept
        reflection = GroupElement(LatticeAut([-1 0 ; 0 1]), Cocycle(0//1, Rational{Int}[0//1, 0//1]))
        smω = SymmetricSpace(Fourier(2, 1.0) ⊗ Fourier(2, 2.0), Group(swap, reflection))
        @test length(elements(symmetry(codomain(Δ, smω)))) == 4
        @test all(g -> g.lattice_aut.matrix[1,2] == 0, elements(symmetry(codomain(Δ, smω))))
    end

end
