@testset "Evaluation" begin

    @testset "Constructors and value" begin
        @test Evaluation(1.0) isa Evaluation{Float64}
        @test value(Evaluation(1.0)) == 1.0
        @test Evaluation(nothing) isa Evaluation{Nothing}
        @test value(Evaluation(nothing)) === nothing
        # tuple constructor and varargs constructor coincide
        @test Evaluation(1.0, nothing, 2.0) == Evaluation((1.0, nothing, 2.0))
        @test value(Evaluation(1.0, nothing, 2.0)) == (1.0, nothing, 2.0)
        # empty tuple is explicitly rejected
        @test_throws ArgumentError Evaluation()
    end

    @testset "Taylor" begin
        # p(t) = 1 - 2t + 3t²
        𝒯 = Taylor(2)
        a = Sequence(𝒯, [1.0, -2.0, 3.0])
        ℰ = Evaluation(2.0)

        @test domain(Evaluation(nothing), 𝒯) == 𝒯
        @test domain(ℰ, 𝒯) == UndefSpace()
        @test codomain(Evaluation(nothing), 𝒯) == 𝒯
        @test codomain(ℰ, 𝒯) == Taylor(0)

        # p(2) = 1 - 4 + 12 = 9
        c_expected = Sequence(Taylor(0), [9.0])
        @test project(ℰ, 𝒯, Taylor(0), Float64)(a) ==
            evaluate!(Sequence(Taylor(0), [Inf]), a, 2.0) ==
            mul!(Sequence(Taylor(0), [Inf]), project(ℰ, 𝒯, Taylor(0), Float64), a) ==
            c_expected
        @test a(2.0) == evaluate(a, 2.0) == (ℰ * a) == 9.0 == c_expected[0]
        # in-place vector form
        @test evaluate!([Inf], a, 2.0) == [9.0]

        # p(0) = 1 (hits the `_safe_iszero(x)` short-circuit branch)
        @test a(0.0) == 1.0

        # row operator: [x^0, x^1, x^2] evaluated at x = 2
        @test coefficients(project(ℰ, 𝒯, Taylor(0), Float64)) == [1.0 2.0 4.0]

        # x = nothing is the identity
        @test evaluate(a, nothing) == a

        # mismatched output space
        @test_throws ArgumentError evaluate!(Sequence(Taylor(1), [Inf, Inf]), a, 2.0)

        # complex evaluation point
        z = 1.0 + 2.0im
        @test a(z) ≈ 1.0 - 2.0*z + 3.0*z^2

        # ComplexF64 coefficients
        a_c = Sequence(𝒯, ComplexF64[1.0, -2.0, 3.0])
        @test a_c(2.0) == 9.0 + 0.0im

        # Interval{Float64} coefficients: enclosure of the exact value
        a_i = Sequence(𝒯, interval.([1.0, -2.0, 3.0]))
        @test in_interval(9.0, a_i(interval(2.0)))
        xi = interval(1.9, 2.1) # interval point encloses 2.0
        @test in_interval(9.0, a_i(xi))
    end

    @testset "Fourier" begin
        ω = 1.0
        ℱ = Fourier(1, ω)
        # a₋₁ = 1+i, a₀ = 2, a₁ = 1-i
        a = Sequence(ℱ, ComplexF64[1.0+1.0im, 2.0+0.0im, 1.0-1.0im])
        ℰ0 = Evaluation(0.0)

        @test domain(Evaluation(nothing), ℱ) == ℱ
        @test domain(ℰ0, ℱ) == UndefSpace()
        @test codomain(Evaluation(nothing), ℱ) == ℱ
        @test codomain(ℰ0, ℱ) == Fourier(0, ω)

        # cis(0) = 1 exactly (`_safe_iszero(x)` branch): c₀ = a₋₁ + a₀ + a₁ = (1+i) + 2 + (1-i) = 4
        c_expected = Sequence(Fourier(0, ω), ComplexF64[4.0])
        @test project(ℰ0, ℱ, Fourier(0, ω), ComplexF64)(a) ==
            evaluate!(Sequence(Fourier(0, ω), [complex(Inf)]), a, 0.0) ==
            mul!(Sequence(Fourier(0, ω), [complex(Inf)]), project(ℰ0, ℱ, Fourier(0, ω), ComplexF64), a) ==
            c_expected
        @test a(0.0) == evaluate(a, 0.0) == (ℰ0 * a) == 4.0 + 0.0im == c_expected[0]

        # x = nothing is the identity
        @test evaluate(a, nothing) == a
        out = Sequence(ℱ, fill(complex(Inf), 3))
        @test mul!(out, Evaluation(nothing), a) == a

        # row operator at x = 0 is exactly [1, 1, 1]
        @test coefficients(project(ℰ0, ℱ, Fourier(0, ω), ComplexF64)) == ComplexF64[1 1 1]

        # ωx = π ⟹ cis(ωxj) = -1: c₀ = a₀ - a₁ - a₋₁ = 2 - (1-i) - (1+i) = 0
        @test a(π) ≈ 0.0 atol=1e-12

        # row operator at generic x matches cis(ωxj) by hand
        x = 0.7
        P = project(Evaluation(x), ℱ, Fourier(0, ω), ComplexF64)
        @test coefficients(P) ≈ [cis(-ω*x) cis(0.0*x) cis(ω*x)]
        @test (P * a)[0] ≈ a(x)

        # periodicity: period is 2π/ω
        T = 2π/ω
        @test a(0.37) ≈ a(0.37 + T)
        @test a(0.37) ≈ a(0.37 - 2T)

        # order-0 (constant) Fourier: value is independent of x
        a0 = Sequence(Fourier(0, ω), ComplexF64[3.0])
        @test a0(1.234) == a0(-5.6) == 3.0 + 0.0im

        # real Float64 coefficients still promote to ComplexF64 through `cis`
        a_r = Sequence(ℱ, [0.5, 1.0, 0.5])
        @test a_r(0.3) isa ComplexF64

        # Interval{Float64} coefficients
        a_i = Sequence(ℱ, interval.([1.0, 2.0, 1.0]))
        @test in_interval(4.0 + 0.0im, a_i(interval(0.0)))
    end

    @testset "Chebyshev" begin
        # T₀ = 1, T₁(x) = x, T₂(x) = 2x² - 1 with f(x) = a₀ + 2∑ₖ aₖ Tₖ(x)
        𝒞 = Chebyshev(2)
        a = Sequence(𝒞, [1.0, 2.0, 3.0])
        ℰ = Evaluation(0.4)

        @test domain(Evaluation(nothing), 𝒞) == 𝒞
        @test domain(ℰ, 𝒞) == UndefSpace()
        @test codomain(Evaluation(nothing), 𝒞) == 𝒞
        @test codomain(ℰ, 𝒞) == Chebyshev(0)

        Tk(k, x) = cos(k*acos(x))
        expected_0p4 = a[0] + 2*a[1]*Tk(1, 0.4) + 2*a[2]*Tk(2, 0.4)
        @test project(ℰ, 𝒞, Chebyshev(0), Float64)(a) ≈
            evaluate!(Sequence(Chebyshev(0), [Inf]), a, 0.4) ≈
            mul!(Sequence(Chebyshev(0), [Inf]), project(ℰ, 𝒞, Chebyshev(0), Float64), a) ≈
            Sequence(Chebyshev(0), [expected_0p4])
        @test a(0.4) ≈ evaluate(a, 0.4) ≈ (ℰ * a) ≈ expected_0p4

        # T_k(±1) = ±1 shortcuts: exact integer arithmetic
        @test a(1.0) == a[0] + 2*a[1] + 2*a[2] == 11.0                      # Tₖ(1) = 1
        @test a(-1.0) == a[0] - 2*a[1] + 2*a[2] == 3.0                      # Tₖ(-1) = (-1)^k
        @test a(0.0) == a[0] - 2*a[2] == -5.0                               # T₁(0)=0, T₂(0)=-1

        # x = nothing is the identity
        @test evaluate(a, nothing) == a
        outb = Sequence(𝒞, fill(Inf, 3))
        @test mul!(outb, Evaluation(nothing), a) == a

        # row operators are exact at x = 0, ±1
        @test coefficients(project(Evaluation(1.0), 𝒞, Chebyshev(0), Float64)) == [1.0 2.0 2.0]
        @test coefficients(project(Evaluation(-1.0), 𝒞, Chebyshev(0), Float64)) == [1.0 -2.0 2.0]
        @test coefficients(project(Evaluation(0.0), 𝒞, Chebyshev(0), Float64)) == [1.0 0.0 -2.0]

        # order-0 and order-1 special-cased Clenshaw paths
        a0 = Sequence(Chebyshev(0), [5.0])
        @test a0(0.3) == a0(-0.9) == 5.0
        a1 = Sequence(Chebyshev(1), [1.0, 2.0])
        @test a1(0.3) ≈ 1.0 + 2*2*0.3

        # complex evaluation point (still governed by the same Clenshaw recursion)
        z = 0.3 + 0.2im
        @test a(z) ≈ a[0] + 2*a[1]*Tk(1, z) + 2*a[2]*Tk(2, z)

        # Interval{Float64} coefficients: enclosure of the exact value at x = 1
        a_i = Sequence(𝒞, interval.([1.0, 2.0, 3.0]))
        @test in_interval(11.0, a_i(interval(1.0)))
    end

    @testset "TensorSpace (partial evaluation)" begin
        # Taylor(1) ⊗ Fourier(1, 1.0): coefficients laid out with Taylor index fastest
        s = Taylor(1) ⊗ Fourier(1, 1.0)
        a = Sequence(s, collect(1.0:6.0))
        # a(i,j): a(0,-1)=1, a(1,-1)=2, a(0,0)=3, a(1,0)=4, a(0,1)=5, a(1,1)=6

        # fix Taylor at x=2, keep Fourier free: cⱼ = a(0,j) + 2*a(1,j)
        c1 = evaluate(a, (2.0, nothing))
        @test space(c1) == Taylor(0) ⊗ Fourier(1, 1.0)
        @test c1 == Sequence(Taylor(0) ⊗ Fourier(1, 1.0), [5.0, 11.0, 17.0])
        @test a(2.0, nothing) == c1

        # keep Taylor free, evaluate Fourier at x=0: cᵢ = a(i,-1)+a(i,0)+a(i,1)
        c2 = evaluate(a, (nothing, 0.0))
        @test space(c2) == Taylor(1) ⊗ Fourier(0, 1.0)
        @test c2 == Sequence(Taylor(1) ⊗ Fourier(0, 1.0), ComplexF64[9.0, 12.0])

        # fully specified tuple collapses to a raw scalar (like the BaseSpace case)
        @test evaluate(a, (2.0, 0.0)) == 33.0 + 0.0im

        # (nothing, nothing) is the identity
        @test evaluate(a, (nothing, nothing)) == a

        # partial evaluation exercising the Chebyshev array branch
        s2 = Taylor(1) ⊗ Chebyshev(2)
        b = Sequence(s2, collect(1.0:6.0))
        # T₁(0.5)=0.5, T₂(0.5)=-0.5; vᵢ = b(i,0) + 2*0.5*b(i,1) - 2*0.5*b(i,2)
        c3 = evaluate(b, (nothing, 0.5))
        @test space(c3) == Taylor(1) ⊗ Chebyshev(0)
        @test c3 == Sequence(Taylor(1) ⊗ Chebyshev(0), [-1.0, 0.0])

        # Fourier as the *leading* tensor factor evaluated at x = 0 (exercises the
        # array-based `_apply!` zero-shortcut branch for the first factor)
        s5 = Fourier(1, 1.0) ⊗ Taylor(1)
        d = Sequence(s5, collect(1.0:6.0))
        # d(k,m): d(-1,0)=1, d(0,0)=2, d(1,0)=3, d(-1,1)=4, d(0,1)=5, d(1,1)=6
        # cis(0)=1 exactly ⟹ cₘ = d(0,m) + d(1,m) + d(-1,m)
        c5 = evaluate(d, (0.0, nothing))
        @test space(c5) == Fourier(0, 1.0) ⊗ Taylor(1)
        @test c5 == Sequence(Fourier(0, 1.0) ⊗ Taylor(1), ComplexF64[6.0, 15.0])

        # Chebyshev as a *non-leading* tensor factor: each closed-form shortcut
        # (x=0, x=-1, x=1) and each low-order special case (ord=0, ord=1) of the
        # Clenshaw recursion, all reached through the `Val`-dimension array branch
        e0 = Sequence(Taylor(1) ⊗ Chebyshev(2), collect(1.0:6.0))
        # x=0, ord=2: only the i=2 term contributes, 2%4≠0 ⟹ coefficient -2
        # vᵢ = e0(i,0) - 2*e0(i,2): v₀ = 1-2*5 = -9, v₁ = 2-2*6 = -10
        @test evaluate(e0, (nothing, 0.0)) == Sequence(Taylor(1) ⊗ Chebyshev(0), [-9.0, -10.0])

        e1 = Sequence(Taylor(1) ⊗ Chebyshev(1), collect(1.0:4.0))
        # x=-1, ord=1: i=1 is odd ⟹ coefficient -2; vᵢ = e1(i,0) - 2*e1(i,1)
        @test evaluate(e1, (nothing, -1.0)) == Sequence(Taylor(1) ⊗ Chebyshev(0), [-5.0, -6.0])
        # x=1, ord=1: coefficient +2; vᵢ = e1(i,0) + 2*e1(i,1)
        @test evaluate(e1, (nothing, 1.0)) == Sequence(Taylor(1) ⊗ Chebyshev(0), [7.0, 10.0])
        # generic x, ord=1 special-cased Clenshaw path: vᵢ = e1(i,0) + 2x*e1(i,1)
        @test evaluate(e1, (nothing, 0.3)) ≈ Sequence(Taylor(1) ⊗ Chebyshev(0), [1.0 + 2*0.3*3.0, 2.0 + 2*0.3*4.0])

        e2 = Sequence(Taylor(1) ⊗ Chebyshev(0), [1.0, 2.0])
        # ord=0: constant term only, independent of x
        @test evaluate(e2, (nothing, 0.3)) == Sequence(Taylor(1) ⊗ Chebyshev(0), [1.0, 2.0])
    end

    @testset "CartesianSpace" begin
        s = Taylor(1)^2 × Fourier(1, 1.0)
        a = Sequence(s, collect(1.0:7.0))
        # component(a,1) = [1,2,3,4] on Taylor(1)², component(a,2) = [5,6,7] on Fourier(1,1.0)

        v = evaluate(a, 0.5)
        @test v isa Vector{ComplexF64}
        # first two entries come from the Taylor(1)² block: [1+2*0.5, 3+4*0.5] = [2, 5]
        @test v[1] == 2.0
        @test v[2] == 5.0
        @test v[3] ≈ component(a, 2)(0.5)
        @test v == vcat(component(a, 1)(0.5), component(a, 2)(0.5))

        # in-place vector form, seeded with Inf
        outvec = fill(complex(Inf), 3)
        @test evaluate!(outvec, a, 0.5) == v

        # x = nothing is the identity
        @test evaluate(a, nothing) == a
    end

    @testset "SymmetricSpace" begin
        # evensym(Taylor): p(t) = 1 + 2t² + 3t⁴ (odd-order coefficients forced to 0)
        s = evensym(Taylor(4))
        @test indices(s) == 0:2:4
        a = Sequence(s, [1.0, 2.0, 3.0])
        full = Projection(desymmetrize(s)) * a
        @test full == Sequence(Taylor(4), [1.0, 0.0, 2.0, 0.0, 3.0])
        # codomain is a `SymmetricSpace` wrapping `Taylor(0)` (dimension 1, index 0)
        cs = codomain(Evaluation(0.3), s)
        @test desymmetrize(cs) == Taylor(0)
        @test indices(cs) == 0:0
        @test a(0.3) == full(0.3) == 1.0 + 2*0.3^2 + 3*0.3^4

        # oddsym(Fourier): only positive frequencies are stored, a₋ⱼ = -aⱼ
        s2 = oddsym(Fourier(2, 1.0))
        a2 = Sequence(s2, ComplexF64[1.0, 2.0])
        full2 = Projection(desymmetrize(s2)) * a2
        @test a2(0.4) ≈ full2(0.4)

        # oddsym(Chebyshev): only odd-order coefficients are stored
        s3 = oddsym(Chebyshev(3))
        a3 = Sequence(s3, [1.0, 2.0])
        full3 = Projection(desymmetrize(s3)) * a3
        @test full3 == Sequence(Chebyshev(3), [0.0, 1.0, 0.0, 2.0])
        @test a3(0.5) == full3(0.5) == -3.0

        # x = nothing is the identity on a symmetric space
        @test evaluate(a, nothing) == a

        # d4sym on a Fourier⊗Fourier tensor: full-tuple evaluation matches desymmetrization
        s4 = d4sym(Fourier(1, 1.0) ⊗ Fourier(1, 1.0))
        a4 = Sequence(s4, ComplexF64[1.0, 2.0, 3.0])
        full4 = Projection(desymmetrize(s4)) * a4
        x = (0.3, 0.4)
        @test a4(x...) ≈ full4(x...)
        # codomain is a `SymmetricSpace` wrapping `Fourier(0,1.0) ⊗ Fourier(0,1.0)` (dimension 1)
        cs4 = codomain(Evaluation(x), s4)
        @test desymmetrize(cs4) == Fourier(0, 1.0) ⊗ Fourier(0, 1.0)
        @test indices(cs4) == [(0, 0)]

        # a mixed tuple (some `nothing`, some `Number`) has an empty domain on a
        # symmetric tensor space: only the all-`nothing` tuple is handled
        @test domain(Evaluation((0.3, nothing)), s4) == UndefSpace()
        @test domain(Evaluation((nothing, nothing)), s4) == s4
    end

    @testset "InfiniteSequence / domain checks" begin
        # Taylor: closed disk of radius ν
        X = Ell1(GeometricWeight(2.0))
        a = InfiniteSequence(Sequence(Taylor(2), [1.0, 1.0, 1.0]), 0.0, 0.0, 0.0, X)
        @test in_interval(1.0 + 1.5 + 1.5^2, evaluate(a, 1.5)) # inside disk of radius 2
        @test_throws DomainError evaluate(a, 3.0)              # outside disk of radius 2

        X1 = Ell1() # IdentityWeight ⟺ disk of radius 1
        b = InfiniteSequence(Sequence(Taylor(2), [1.0, 1.0, 1.0]), X1)
        @test in_interval(3.0, evaluate(b, 1.0))
        @test_throws DomainError evaluate(b, 1.5)

        # `nothing` always passes, regardless of the sequence space / Banach space
        @test space(evaluate(b, nothing)) == Taylor(2)

        # Fourier: closed strip |ω·Im(x)| ≤ log(ν)
        Xf = Ell1(GeometricWeight(2.0))
        f = InfiniteSequence(Sequence(Fourier(1, 1.0), ComplexF64[0.5, 1.0, 0.5]), 0.0, 0.0, 0.0, Xf)
        @test evaluate(f, 0.3 + 0.5im) isa Complex
        @test_throws DomainError evaluate(f, 0.3 + 1.0im) # log(2) ≈ 0.693 < 1.0

        Xf1 = Ell1() # IdentityWeight ⟺ strip degenerates to the real axis
        f1 = InfiniteSequence(Sequence(Fourier(1, 1.0), ComplexF64[0.5, 1.0, 0.5]), Xf1)
        @test evaluate(f1, 0.3) isa Complex
        @test_throws DomainError evaluate(f1, 0.3 + 0.1im)

        # Chebyshev: closed Bernstein ellipse of parameter ν
        Xc = Ell1(GeometricWeight(2.0))
        c = InfiniteSequence(Sequence(Chebyshev(2), [1.0, 1.0, 1.0]), 0.0, 0.0, 0.0, Xc)
        # a₀ + 2a₁T₁(0.5) + 2a₂T₂(0.5) = 1 + 2(0.5) + 2(-0.5) = 1
        @test in_interval(1.0, evaluate(c, 0.5))
        @test_throws DomainError evaluate(c, 2.0)   # outside the ellipse
        @test_throws DomainError evaluate(c, 2.0im) # outside the ellipse (semi-minor axis b = 0.75)

        Xc1 = Ell1() # ν = 1: the ellipse degenerates to the segment [-1, 1]
        c1 = InfiniteSequence(Sequence(Chebyshev(2), [1.0, 1.0, 1.0]), Xc1)
        @test in_interval(5.0, evaluate(c1, 1.0))
        @test_throws DomainError evaluate(c1, 0.5im)
        @test_throws DomainError evaluate(c1, 1.5)

        # SymmetricSpace delegates the domain check to `desymmetrize`
        Xs = Ell1(GeometricWeight(2.0))
        s = InfiniteSequence(Sequence(evensym(Taylor(4)), [1.0, 2.0, 3.0]), 0.0, 0.0, 0.0, Xs)
        @test in_interval(1.0 + 2*1.5^2 + 3*1.5^4, evaluate(s, 1.5))
        @test_throws DomainError evaluate(s, 3.0)

        # tensor space: per-factor check, `nothing` passes that factor unconditionally
        X2 = Ell1((GeometricWeight(2.0), GeometricWeight(2.0)))
        t = InfiniteSequence(Sequence(Taylor(1) ⊗ Taylor(1), [1.0, 1.0, 1.0, 1.0]), 0.0, 0.0, 0.0, X2)
        @test space(evaluate(t, (0.5, nothing))) == Taylor(0) ⊗ Taylor(1)
        @test_throws DomainError evaluate(t, (3.0, 0.5))

        # unimplemented Banach space / sequence space combination
        Xg = Ell2()
        g = InfiniteSequence(Sequence(Taylor(2), [1.0, 1.0, 1.0]), Xg)
        @test_throws DomainError evaluate(g, 0.5)
        @test space(evaluate(g, nothing)) == Taylor(2) # `nothing` bypasses the check entirely
    end

    # `Evaluation` on a `ScalarSpace` is commented out in
    # src/sequence_spaces/linear_operators/special_operators/evaluation.jl, so the
    # tests below are commented out too; restore them together with the feature
    #
    # @testset "ScalarSpace component" begin
    #     # a scalar component carries no x-dependence: evaluating passes it through
    #     # unchanged, so `x = nothing` and any `Number` both act as the identity
    #     @test domain(Evaluation(nothing), ScalarSpace()) == ScalarSpace()
    #     @test codomain(Evaluation(nothing), ScalarSpace()) == ScalarSpace()
    #     @test codomain(Evaluation(1.0), ScalarSpace()) == ScalarSpace()
    #
    #     # `domain` under a `Number` evaluation point is `UndefSpace()`: unlike
    #     # `nothing`, this is not meant to be composed with as a plain projection
    #     @test domain(Evaluation(1.0), ScalarSpace()) == UndefSpace()
    #
    #     b = Sequence(ScalarSpace(), [5.0])
    #     @test evaluate(b, 2.0) == b
    #     @test b(2.0) == b
    #     @test evaluate(b, nothing) == b
    #
    #     # `ScalarSpace() × Taylor(...)`: the standard continuation layout pairing a
    #     # scalar (e.g. a parameter) with a sequence-space component
    #     s = ScalarSpace() × Taylor(2)
    #     a = Sequence(s, [10.0, 1.0, -2.0, 3.0]) # p(t) = 1 - 2t + 3t², parameter = 10
    #
    #     @test codomain(Evaluation(2.0), s) == ScalarSpace() × Taylor(0)
    #
    #     # p(2) = 1 - 4 + 12 = 9; the scalar component (10.0) is untouched
    #     c_expected = [10.0, 9.0]
    #     @test evaluate(a, 2.0) == a(2.0) == c_expected
    #     @test coefficients(evaluate!(Sequence(codomain(Evaluation(2.0), s), [Inf, Inf]), a, 2.0)) == c_expected
    #     @test evaluate!([Inf, Inf], a, 2.0) == c_expected
    #     @test coefficients(project(Evaluation(2.0), s, codomain(Evaluation(2.0), s), Float64) * a) == c_expected
    #
    #     # row operator: scalar row is the identity row [1, 0, 0], Taylor row is
    #     # [x^0, x^1, x^2] evaluated at x = 2
    #     @test coefficients(project(Evaluation(2.0), s, codomain(Evaluation(2.0), s), Float64)) ==
    #         [1.0 0.0 0.0 0.0
    #          0.0 1.0 2.0 4.0]
    #
    #     # x = nothing is the identity on the whole cartesian sequence
    #     @test evaluate(a, nothing) == a
    # end

end
