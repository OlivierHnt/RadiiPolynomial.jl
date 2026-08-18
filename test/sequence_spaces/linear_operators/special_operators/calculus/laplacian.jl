@testset "Laplacian" begin

    @testset "Fourier (base space)" begin
        # Δ is defined as Derivative(2) on a single BaseSpace: domain/codomain are unchanged
        s = Fourier(2, 1.0)
        @test domain(Laplacian(), s) == s
        @test codomain(Laplacian(), s) == s

        # a = 1 + 2cos(x) + 3cos(2x) (real, Hermitian-symmetric coefficients stored as Float64)
        a = Sequence(s, [1.0, 2.0, 3.0, 2.0, 1.0]) # j = -2,-1,0,1,2
        Δ = Laplacian()
        out = Sequence(s, fill(Inf, 5))
        expected = Sequence(s, [-4.0, -2.0, 0.0, -2.0, -4.0]) # -ω²j² a_j
        @test Δ(a) == project(Δ, s, s, Float64)(a) == laplacian!(out, a) ==
            mul!(Sequence(s, fill(Inf, 5)), Δ, a) == expected

        @test laplacian(a) == differentiate(a, 2)

        # a Fourier space always promotes the coefficient type to complex
        @test eltype(laplacian(a)) == ComplexF64

        # a complex input whose imaginary part is exactly zero still works
        a_c0 = Sequence(s, ComplexF64[1.0, 2.0, 3.0, 2.0, 1.0])
        @test laplacian(a_c0) == expected
        @test eltype(laplacian(a_c0)) == ComplexF64

        @test_throws ArgumentError laplacian!(Sequence(Fourier(1, 1.0), fill(Inf, 3)), a)

        # a genuinely complex input works too, with no coercion to real
        a_c = Sequence(s, ComplexF64[1.0, 2.0 + 3.0im, 3.0, 2.0 - 3.0im, 1.0]) # Hermitian-symmetric
        @test laplacian(a_c) == differentiate(a_c, 2) ==
            Sequence(s, ComplexF64[-4.0-0.0im, -2.0-3.0im, 0.0+0.0im, -2.0+3.0im, -4.0-0.0im])
    end

    @testset "Tensor space" begin
        # Δ = Σᵢ ∂²/∂xᵢ²: cross-check against the sum of the individual `Derivative(2·eᵢ)`
        s = Fourier(1, 1.0) ⊗ Fourier(1, 1.0)
        b = Sequence(s, collect(1.0:9.0))
        Δ = Laplacian()
        lb = laplacian(b)
        @test lb == differentiate(b, (2, 0)) + differentiate(b, (0, 2))
        @test lb == Sequence(s, ComplexF64[-2.0, -2.0, -6.0, -4.0, 0.0, -6.0, -14.0, -8.0, -18.0])

        out = Sequence(s, fill(complex(Inf), 9))
        @test Δ(b) == project(Δ, s, codomain(Δ, s), ComplexF64)(b) == laplacian!(out, b) ==
            mul!(Sequence(s, fill(complex(Inf), 9)), Δ, b) == lb

        # complex coefficients are preserved on a tensor space too
        bc = Sequence(s, ComplexF64[1+1im, 2, 3-2im, 4, 5, 6+3im, 7, 8, 9-1im])
        @test laplacian(bc) == differentiate(bc, (2, 0)) + differentiate(bc, (0, 2))

        # with a Taylor factor of order 1, ∂²ₓ truncates that factor to zero and only
        # ∂²_y contributes
        s2 = Taylor(1) ⊗ Fourier(2, 1.0)
        c = Sequence(s2, collect(1.0:dimension(s2)))
        @test differentiate(c, (2, 0)) == zeros(ComplexF64, Taylor(0) ⊗ Fourier(2, 1.0))
        @test laplacian(c) == differentiate(c, (0, 2))

        # Δ is diagonal in the Fourier basis, with entry -(ω₁²i₁² + ω₂²i₂²) at (i,i)
        # and 0 off the diagonal
        s3 = Fourier(2, 1.0) ⊗ Fourier(2, 2.0)
        dom3, codom3 = domain(Δ, s3), codomain(Δ, s3)
        @test RadiiPolynomial.getcoefficient(Δ, (codom3, (1, 1)), (dom3, (1, 1)), Float64) == -(1.0^2*1^2 + 2.0^2*1^2)
        @test RadiiPolynomial.getcoefficient(Δ, (codom3, (2, 0)), (dom3, (2, 0)), Float64) == -(1.0^2*2^2)
        @test RadiiPolynomial.getcoefficient(Δ, (codom3, (1, 1)), (dom3, (0, 1)), Float64) == 0.0
    end

    @testset "Cartesian space" begin
        # Laplacian applies component-wise, same as Derivative
        sp = Fourier(1, 1.0)^2
        ap = Sequence(sp, collect(1.0:6.0))
        @test laplacian(ap) == Sequence(sp, [-1.0, 0.0, -3.0, -4.0, 0.0, -6.0])
        @test component(laplacian(ap), 1) == laplacian(component(ap, 1))

        spx = Taylor(1) × Fourier(1, 1.0)
        ax = Sequence(spx, collect(1.0:5.0))
        @test laplacian(ax) == Sequence(Taylor(0) × Fourier(1, 1.0), ComplexF64[0.0, -3.0, 0.0, -5.0])

        @test domain(Laplacian(), sp) == sp
        @test codomain(Laplacian(), sp) == sp
    end

    @testset "Symmetric space" begin
        sE = evensym(Fourier(2, 1.0))
        sO = oddsym(Fourier(2, 1.0))
        Δ = Laplacian()

        @test domain(Δ, sE) == sE # order 2 ⇒ (-1)² = 1, symmetry preserved
        @test codomain(Δ, sO) == sO

        # evensym (cosine-like): real representative values
        a = Sequence(sE, ComplexF64[1.0, 2.0, 3.0])
        out = Sequence(sE, fill(Inf, 3))
        expected = Sequence(sE, [0.0, -2.0, -12.0])
        @test Δ(a) == project(Δ, sE, sE, Float64)(a) == laplacian!(out, a) ==
            mul!(Sequence(sE, fill(Inf, 3)), Δ, a) == expected
        @test eltype(laplacian(a)) == ComplexF64

        # odd (sine-like) representative coefficients are naturally purely imaginary,
        # e.g. sin(x) ↦ i/2
        aO = Sequence(sO, ComplexF64[2.0im, 6.0im])
        @test laplacian(aO) == differentiate(aO, 2) == Sequence(sO, ComplexF64[-2.0im, -24.0im])
    end

    @testset "Symmetric space (tensor)" begin
        # Δ is diagonal with symbol -(ω₁²k₁² + ω₂²k₂²); when every lattice automorphism of the group
        # preserves the symbol, Δ maps the symmetric space into itself with the same group
        t = d4sym(Fourier(2, 2.0) ⊗ Fourier(2, 2.0))
        Δ = Laplacian()
        @test domain(Δ, t) === t # no group transform and no space reconstruction needed
        @test codomain(Δ, t) === t

        # commuting diagram: desymmetrizing then applying Δ equals applying Δ then desymmetrizing
        a = Sequence(t, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        @test Projection(desymmetrize(t)) * laplacian(a) == laplacian(Projection(desymmetrize(t)) * a)

        # the projected matrix is diagonal on the representatives with entry -ω²(k₁² + k₂²)
        M = project(Δ, t, t, Float64)
        for (i, k) ∈ enumerate(indices(t)), (j, l) ∈ enumerate(indices(t))
            @test coefficients(M)[i,j] == (k == l ? -4.0 * (k[1]^2 + k[2]^2) : 0.0)
        end

        # interval frequencies are compared by their bounds in the invariance check
        ti = interval(t)
        @test domain(Δ, ti) === ti

        # a swap action requires the swapped factors to have the same frequency
        swap = Group(GroupElement(LatticeAut([0 1 ; 1 0]), Cocycle(1, Rational{Int}[0//1, 0//1])))
        @test domain(Δ, SymmetricSpace(Fourier(2, 1.0) ⊗ Fourier(2, 1.0), swap)) isa SymmetricSpace
        @test_throws ArgumentError domain(Δ, SymmetricSpace(Fourier(2, 1.0) ⊗ Fourier(2, 2.0), swap))
        @test_throws ArgumentError codomain(Δ, SymmetricSpace(Fourier(2, 1.0) ⊗ Fourier(2, 2.0), swap))
    end

    @testset "unsupported / restrictions" begin
        # on a Chebyshev space a second derivative has no well-defined domain
        @test domain(Laplacian(), Chebyshev(3)) == UndefSpace()

        # on a Taylor space Δ is the second derivative
        a2 = Sequence(Taylor(3), [1.0, 2.0, 3.0, 4.0]) # 1+2x+3x²+4x³ ⇒ Δ = 6+24x
        expected = Sequence(Taylor(1), [6.0, 24.0])
        @test laplacian(a2) == differentiate(a2, 2) == expected
        @test project(Laplacian(), Taylor(3), codomain(Laplacian(), Taylor(3)), Float64)(a2) == expected

        # an order below 2 gives zero
        a1 = Sequence(Taylor(1), [1.0, 2.0])
        @test laplacian(a1) == Sequence(Taylor(0), [0.0])
    end

end
