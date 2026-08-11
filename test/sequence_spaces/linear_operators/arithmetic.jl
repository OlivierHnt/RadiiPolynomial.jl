@testset "LinearOperator arithmetic" begin

    @testset "unary +, -, and lazy Negate" begin
        𝒯 = Taylor(1)
        A = LinearOperator(𝒯, 𝒯, [1.0 2.0; 3.0 4.0])

        @test +A == A
        @test -A == LinearOperator(𝒯, 𝒯, [-1.0 -2.0; -3.0 -4.0])

        # `-` on a generic (non-`LinearOperator`) `AbstractLinearOperator` stays lazy: it
        # wraps its argument in a `Negate`, it does not materialize anything.
        B = LinearOperator(𝒯, 𝒯, [5.0 6.0; 7.0 8.0])
        S = A ∘ B # lazy `ComposedOperator`
        @test S isa ComposedOperator
        negS = -S
        @test negS isa Negate
        @test -negS === S # double negation exactly cancels, `-Negate(S) === S`

        # materializing `-S` must equal minus the materialized product A*B
        AB_expected = [1.0 2.0; 3.0 4.0] * [5.0 6.0; 7.0 8.0]
        @test project(negS, 𝒯, 𝒯) == -(A * B) == LinearOperator(𝒯, 𝒯, -AB_expected)
    end

    @testset "+ and - between LinearOperators: domain/codomain adaptation" begin
        @testset "Taylor" begin
            𝒯₁ = Taylor(1) # indices 0,1   (dim 2)
            𝒯₂ = Taylor(2) # indices 0,1,2 (dim 3)
            A = LinearOperator(𝒯₁, 𝒯₁, [1.0 2.0; 3.0 4.0])
            B = LinearOperator(𝒯₂, 𝒯₁, [1.0 2.0 3.0; 4.0 5.0 6.0]) # bigger domain

            # out-of-place `+`/`-` widen to the union of the domains/codomains: B (2×3) with
            # A folded into its first two (overlapping) columns; column index 2 is untouched
            expected_sum = [2.0 4.0 3.0; 7.0 9.0 6.0]
            @test domain(A + B) == 𝒯₂
            @test codomain(A + B) == 𝒯₁
            @test coefficients(A + B) == expected_sum
            @test A + B == A - (-B) == ladd!(copy(A), copy(B)) == lsub!(copy(A), -B)

            # `radd!`/`rsub!` instead keep A's own (smaller) shape: only the top-left 2×2
            # overlap of B is folded in, column index 2 of B is discarded
            expected_radd = [1.0 2.0; 3.0 4.0] .+ [1.0 2.0; 4.0 5.0] # top-left 2×2 block of B
            @test coefficients(radd!(copy(A), B)) == expected_radd
            @test radd!(copy(A), B) == rsub!(copy(A), -B)

            # mismatched codomain instead of domain
            D = LinearOperator(𝒯₁, 𝒯₂, [1.0 2.0; 3.0 4.0; 5.0 6.0]) # bigger codomain
            expected_AD = [2.0 4.0; 6.0 8.0; 5.0 6.0] # D with A folded into its first two rows
            @test coefficients(A + D) == expected_AD
            @test A + D == A - (-D)
        end

        @testset "Fourier" begin
            ℱ₁ = Fourier(1, 1.0) # indices -1,0,1     (dim 3)
            ℱ₂ = Fourier(2, 1.0) # indices -2,...,2   (dim 5)
            A = LinearOperator(ℱ₁, ℱ₁, Float64[1 2 3; 4 5 6; 7 8 9])
            B = LinearOperator(ℱ₂, ℱ₁, Float64[1 2 3 4 5; 6 7 8 9 10; 11 12 13 14 15])

            # A's frequencies -1,0,1 land in B's columns 2,3,4 (1-indexed): B's domain is
            # ordered -2,-1,0,1,2, so column 1 (freq. -2) and column 5 (freq. 2) are untouched
            expected = Float64[1 2 3 4 5; 6 7 8 9 10; 11 12 13 14 15]
            expected[:, 2:4] .+= Float64[1 2 3; 4 5 6; 7 8 9]
            @test coefficients(A + B) == expected
            @test A + B == A - (-B)
        end

        @testset "Chebyshev" begin
            𝒞₁ = Chebyshev(1)
            𝒞₂ = Chebyshev(2)
            A = LinearOperator(𝒞₁, 𝒞₁, [1.0 2.0; 3.0 4.0])
            B = LinearOperator(𝒞₂, 𝒞₁, [1.0 2.0 3.0; 4.0 5.0 6.0])
            @test coefficients(A + B) == [2.0 4.0 3.0; 7.0 9.0 6.0] # identical mechanics to Taylor
            @test A + B == A - (-B)
        end

        @testset "TensorSpace" begin
            𝒮 = Taylor(1) ⊗ Chebyshev(1) # dim 4
            A = LinearOperator(𝒮, 𝒮, Float64[1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16])
            B = LinearOperator(𝒮, 𝒮, Matrix{Float64}(I, 4, 4))
            @test coefficients(A + B) == coefficients(A) .+ coefficients(B)
            @test A + B == A - (-B)
        end

        @testset "SymmetricSpace" begin
            𝓈₂ = evensym(Taylor(2)) # only even powers survive: indices 0,2 (dim 2)
            𝓈₁ = evensym(Taylor(1)) # index 0 only (dim 1)
            A = LinearOperator(𝓈₂, 𝓈₂, [1.0 2.0; 3.0 4.0])
            B = LinearOperator(𝓈₂, 𝓈₂, [1.0 0.0; 0.0 1.0])
            @test coefficients(A + B) == [2.0 2.0; 3.0 5.0]

            # mismatched order within the same symmetry: only the shared index (0) is folded in
            C = LinearOperator(𝓈₁, 𝓈₁, reshape([5.0], 1, 1))
            @test coefficients(C + A) == [6.0 2.0; 3.0 4.0]
        end
    end

    @testset "scalar *, /, and their in-place forms" begin
        𝒯 = Taylor(1)
        A = LinearOperator(𝒯, 𝒯, [1.0 2.0; 3.0 4.0])

        expected_mul = LinearOperator(𝒯, 𝒯, [3.0 6.0; 9.0 12.0])
        @test A * 3.0 == 3.0 * A == rmul!(copy(A), 3.0) == lmul!(3.0, copy(A)) == expected_mul

        expected_div = LinearOperator(𝒯, 𝒯, [0.5 1.0; 1.5 2.0])
        @test A / 2.0 == 2.0 \ A == rdiv!(copy(A), 2.0) == ldiv!(2.0, copy(A)) == expected_div

        # Interval{Float64} coefficients
        Ai = LinearOperator(𝒯, 𝒯, interval.([1.0 2.0; 3.0 4.0]))
        r = 2.0 * Ai
        @test all(isequal_interval.(coefficients(r), interval.([2.0 4.0; 6.0 8.0])))
        @test all(isequal_interval.(coefficients(rmul!(copy(Ai), 2.0)), coefficients(r)))

        # ComplexF64 coefficients
        Ac = LinearOperator(𝒯, 𝒯, ComplexF64[1.0+1.0im 2.0; 3.0 4.0-2.0im])
        @test 2.0 * Ac == Ac * 2.0 == rmul!(copy(Ac), 2.0) == lmul!(2.0, copy(Ac)) ==
            LinearOperator(𝒯, 𝒯, ComplexF64[2.0+2.0im 4.0; 6.0 8.0-4.0im])
    end

    @testset "* between LinearOperators: materialized product vs ∘ (lazy composition)" begin
        𝒯₁ = Taylor(1) # dim 2
        𝒯₂ = Taylor(2) # dim 3
        A = LinearOperator(𝒯₁, 𝒯₁, [1.0 2.0; 3.0 4.0])
        E = LinearOperator(𝒯₂, 𝒯₂, [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0])

        # `A * E`: domain(A) ≠ codomain(E) (order 1 vs 2), so the contracted dimension is
        # truncated to their intersection (order 1): only E's first two rows enter the product
        expected_AE = [1.0 2.0; 3.0 4.0] * [1.0 2.0 3.0; 4.0 5.0 6.0]
        @test coefficients(A * E) == expected_AE
        @test domain(A * E) == 𝒯₂ && codomain(A * E) == 𝒯₁
        @test A * E == mul!(similar(LinearOperator(𝒯₂, 𝒯₁, Matrix{Float64}(undef, 2, 3))), A, E, true, false)

        # `E * A`: contracted dimension is domain(E) ∩ codomain(A) = order 1; E's columns
        # 3 (order 2) are dropped, all 3 rows of E are kept
        expected_EA = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0][:, 1:2] * [1.0 2.0; 3.0 4.0]
        @test coefficients(E * A) == expected_EA

        # a `LinearOperator * LinearOperator` product materializes directly; the same result
        # is obtained by explicitly composing (`∘`, a lazy `ComposedOperator`) then projecting
        C = A ∘ E
        @test C isa ComposedOperator
        @test project(C, domain(E), codomain(A)) == A * E

        # `mul!` into a smaller preallocated `C` truncates the natural product further:
        # only the first 2 columns of the natural 2×3 product survive
        Csmall = LinearOperator(𝒯₁, 𝒯₁, fill(Inf, 2, 2))
        mul!(Csmall, A, E, 1.0, 0.0)
        @test coefficients(Csmall) == expected_AE[:, 1:2]

        # general β (≠ 0, 1) on that same truncated product: C is rescaled, then the
        # (truncated) product is added on top
        Csmall2 = LinearOperator(𝒯₁, 𝒯₁, [50.0 50.0; 50.0 50.0])
        mul!(Csmall2, A, E, 1.0, 2.0)
        @test coefficients(Csmall2) == 2.0 .* [50.0 50.0; 50.0 50.0] .+ expected_AE[:, 1:2]

        # `domain(A) == codomain(B)` but `C`'s domain/codomain don't match `B`/`A`: the
        # contracted product is again truncated to the overlap, β = 0 then general β
        Bop = LinearOperator(𝒯₂, 𝒯₁, [1.0 2.0 3.0; 4.0 5.0 6.0]) # domain 𝒯₂, codomain 𝒯₁ = domain(A)
        expected_ABop = [1.0 2.0; 3.0 4.0] * [1.0 2.0 3.0; 4.0 5.0 6.0][:, 1:2] # drop Bop's order-2 column
        Cmix = LinearOperator(𝒯₁, 𝒯₁, fill(Inf, 2, 2))
        mul!(Cmix, A, Bop, 1.0, 0.0)
        @test coefficients(Cmix) == expected_ABop

        Cmix2 = LinearOperator(𝒯₁, 𝒯₁, [100.0 100.0; 100.0 100.0])
        mul!(Cmix2, A, Bop, 2.0, 3.0)
        @test coefficients(Cmix2) == 3.0 .* [100.0 100.0; 100.0 100.0] .+ 2.0 .* expected_ABop
    end

    @testset "^, inv, /, and \\" begin
        𝒯 = Taylor(1)
        Amat = [1.0 2.0; 3.0 4.0]
        A = LinearOperator(𝒯, 𝒯, Amat)
        B = LinearOperator(𝒯, 𝒯, [5.0 6.0; 7.0 8.0])

        @test A^0 == LinearOperator(𝒯, 𝒯, [1.0 0.0; 0.0 1.0]) == one(A)
        @test A^1 == A
        @test A^2 == A * A == LinearOperator(𝒯, 𝒯, Amat * Amat)
        @test A^3 == A * A * A == LinearOperator(𝒯, 𝒯, Amat * Amat * Amat)

        Ainv = inv(A)
        @test coefficients(Ainv) ≈ inv(Amat) # A is not exactly invertible in Float64 arithmetic
        @test A^(-1) == Ainv
        @test coefficients(A^(-2)) ≈ inv(Amat) * inv(Amat)

        # literal exponents (`A^(-2)`, `A^4`, ...) are rewritten by Julia into `Base.literal_pow`,
        # which bypasses our custom `^` entirely (it only uses `inv`/`*`); a `let`-bound
        # (non-literal) exponent variable is needed to actually exercise `n < 0` and the
        # power-by-squaring loop below
        let n = -2
            @test coefficients(A^n) ≈ inv(Amat) * inv(Amat)
        end
        let n = 4 # trailing_zeros(4)+1 = 3, so the first squaring `while` loop runs twice
            @test coefficients(A^n) ≈ Amat * Amat * Amat * Amat
        end

        # A / B solves X*B = A ⟺ X = A*inv(B); A \ B solves A*X = B ⟺ X = inv(A)*B
        @test coefficients(A / B) ≈ Amat / [5.0 6.0; 7.0 8.0]
        @test coefficients(A \ B) ≈ Amat \ [5.0 6.0; 7.0 8.0]
    end

    @testset "A + I, A - I, and UniformScaling(Operator) combinations" begin
        𝒯₁ = Taylor(1) # dim 2
        𝒯₂ = Taylor(2) # dim 3
        A = LinearOperator(𝒯₁, 𝒯₁, [1.0 2.0; 3.0 4.0])
        Amat = [1.0 2.0; 3.0 4.0]
        identity2 = [1.0 0.0; 0.0 1.0]

        # `A + J` for `J::UniformScaling` stays lazy (a generic `Add`); materializing it
        # (via `project`) must equal the eager in-place forms
        @test project(A + I, 𝒯₁, 𝒯₁) == project(I + A, 𝒯₁, 𝒯₁) ==
            radd!(copy(A), I) == ladd!(I, copy(A)) == LinearOperator(𝒯₁, 𝒯₁, Amat + identity2)
        @test project(A - I, 𝒯₁, 𝒯₁) == rsub!(copy(A), I) == LinearOperator(𝒯₁, 𝒯₁, Amat - identity2)
        @test project(I - A, 𝒯₁, 𝒯₁) == lsub!(I, copy(A)) == LinearOperator(𝒯₁, 𝒯₁, identity2 - Amat)

        # same, but going through `UniformScalingOperator` explicitly
        @test radd!(copy(A), UniformScalingOperator(1.0)) == radd!(copy(A), I)
        @test radd!(copy(A), UniformScalingOperator(2.0)) == LinearOperator(𝒯₁, 𝒯₁, Amat + 2.0 * identity2)

        # `ScalarSpace`: `A[1,1] += J.λ`
        As = LinearOperator(ScalarSpace(), ScalarSpace(), reshape([3.0], 1, 1))
        @test radd!(copy(As), I) == LinearOperator(ScalarSpace(), ScalarSpace(), reshape([4.0], 1, 1))

        # domain(A) ≠ codomain(A) (different orders, still both `SequenceSpace`): `I` is
        # added only on the shared indices (0 and 1), the extra row (index 2) is untouched
        D = LinearOperator(𝒯₁, 𝒯₂, [1.0 2.0; 3.0 4.0; 5.0 6.0])
        expected_D = [2.0 2.0; 3.0 5.0; 5.0 6.0]
        @test coefficients(radd!(copy(D), I)) == expected_D

        # incompatible domain/codomain *types* (Taylor vs Chebyshev): `_iscompatible` fails
        Dmix = LinearOperator(𝒯₁, Chebyshev(1), [1.0 2.0; 3.0 4.0])
        @test_throws ArgumentError radd!(copy(Dmix), I)
        @test_throws ArgumentError lsub!(I, copy(Dmix))
    end

    @testset "A + I and A - I materialize on spaces built solely out of ScalarSpace" begin
        Amat = [1.0 2.0; 3.0 4.0]
        identity2 = [1.0 0.0; 0.0 1.0]

        # no truncated tail, so the identity is exactly representable and `+`/`-` return a
        # `LinearOperator` rather than the lazy `Add` used on sequence spaces
        @testset "$s" for s ∈ (ScalarSpace()^2, ScalarSpace() × ScalarSpace())
            A = LinearOperator(s, s, copy(Amat))
            @test A + I == I + A == LinearOperator(s, s, Amat + identity2)
            @test A - I == LinearOperator(s, s, Amat - identity2)
            @test I - A == LinearOperator(s, s, identity2 - Amat)
            @test A + 2I == LinearOperator(s, s, Amat + 2.0 * identity2)
        end

        # `ScalarSpace` itself
        As = LinearOperator(ScalarSpace(), ScalarSpace(), reshape([3.0], 1, 1))
        @test As + I == LinearOperator(ScalarSpace(), ScalarSpace(), reshape([4.0], 1, 1))
        @test I - As == LinearOperator(ScalarSpace(), ScalarSpace(), reshape([-2.0], 1, 1))

        # nested cartesian spaces recurse block by block
        s = ScalarSpace()^2 × ScalarSpace() # dim 3
        Bmat = Float64[1 2 3; 4 5 6; 7 8 9]
        identity3 = Matrix{Float64}(I, 3, 3)
        B = LinearOperator(s, s, copy(Bmat))
        @test B + I == LinearOperator(s, s, Bmat + identity3)
        @test I - B == LinearOperator(s, s, identity3 - Bmat)

        # a `SequenceSpace` factor puts the tail back, so the sum stays lazy
        smix = ScalarSpace() × Taylor(1) # dim 3
        M = LinearOperator(smix, smix, copy(Bmat))
        @test M + I isa Add
        @test project(M + I, smix, smix) == LinearOperator(smix, smix, Bmat + identity3)
        @test project(I - M, smix, smix) == LinearOperator(smix, smix, identity3 - Bmat)

        # the identity needs compatible domain and codomain
        @test_throws ArgumentError LinearOperator(ScalarSpace()^2, ScalarSpace()^3, ones(3, 2)) + I

        # the coefficients are promoted against the scaling
        Aint = LinearOperator(ScalarSpace()^2, ScalarSpace()^2, [1 2 ; 3 4])
        @test eltype(Aint + I) == Int
        @test eltype(Aint - 2.5I) == Float64
    end

    @testset "in-place add!/sub!/radd!/rsub!/ladd!/lsub! with AbstractLinearOperator operands" begin
        𝒯 = Taylor(1)
        B = LinearOperator(𝒯, 𝒯, [1.0 2.0; 3.0 4.0])
        expected = [1.0 2.0; 3.0 4.0] + 5.0 * [1.0 0.0; 0.0 1.0] # B + 5I

        # `add!(C, S₁, S₂)` where one or both of S₁, S₂ is a generic `AbstractLinearOperator`
        # (here `UniformScalingOperator`) projects it onto C's domain/codomain first
        C1 = LinearOperator(𝒯, 𝒯, fill(Inf, 2, 2))
        add!(C1, UniformScalingOperator(5.0), B)
        @test coefficients(C1) == expected

        C2 = LinearOperator(𝒯, 𝒯, fill(Inf, 2, 2))
        add!(C2, B, UniformScalingOperator(5.0))
        @test coefficients(C2) == expected

        # `radd!`/`ladd!` with a generic `AbstractLinearOperator` right-hand side
        @test coefficients(radd!(copy(B), UniformScalingOperator(5.0))) == expected
        @test coefficients(ladd!(UniformScalingOperator(5.0), copy(B))) == expected
    end

    @testset "add!/sub!/lsub!(C, A, B): every domain/codomain-matching branch" begin
        𝒯₀ = Taylor(0) # dim 1
        𝒯₁ = Taylor(1) # dim 2
        𝒯₂ = Taylor(2) # dim 3

        @testset "domain(A) == domain(C) && codomain(A) == codomain(C) (B differs)" begin
            A = LinearOperator(𝒯₁, 𝒯₁, [1.0 2.0; 3.0 4.0])
            B = LinearOperator(𝒯₂, 𝒯₁, [1.0 2.0 3.0; 4.0 5.0 6.0]) # bigger domain than A/C

            # `add!`/`sub!` copy A into C, then fold in only B's overlapping (order 0,1) columns
            Cadd = LinearOperator(𝒯₁, 𝒯₁, fill(Inf, 2, 2))
            add!(Cadd, A, B)
            @test coefficients(Cadd) == [1.0 2.0; 3.0 4.0] .+ [1.0 2.0; 4.0 5.0]

            Csub = LinearOperator(𝒯₁, 𝒯₁, fill(Inf, 2, 2))
            sub!(Csub, A, B)
            @test coefficients(Csub) == [1.0 2.0; 3.0 4.0] .- [1.0 2.0; 4.0 5.0]
        end

        @testset "fully generic: none of A, B, C share both domain and codomain" begin
            A = LinearOperator(𝒯₀, 𝒯₀, reshape([10.0], 1, 1))
            B = LinearOperator(𝒯₁, 𝒯₀, [1.0 2.0])
            # entry β=0: A[0]+B[0] = 10+1 = 11; β=1: A(out of domain)=0, B[1] = 2; β=2: both out
            # of range = 0
            Cadd = LinearOperator(𝒯₂, 𝒯₀, fill(Inf, 1, 3))
            add!(Cadd, A, B)
            @test coefficients(Cadd) == [11.0 2.0 0.0]

            Csub = LinearOperator(𝒯₂, 𝒯₀, fill(Inf, 1, 3))
            sub!(Csub, A, B)
            @test coefficients(Csub) == [9.0 -2.0 0.0]
        end

        @testset "lsub!(A, B) fast path: domain(A) == domain(B) && codomain(A) == codomain(B)" begin
            𝒯₁ = Taylor(1)
            A = LinearOperator(𝒯₁, 𝒯₁, [1.0 2.0; 3.0 4.0])
            B = LinearOperator(𝒯₁, 𝒯₁, [5.0 6.0; 7.0 8.0])
            # matching shapes: `lsub!(A, B)` overwrites B with A - B and returns it
            r = lsub!(copy(A), copy(B))
            @test coefficients(r) == [1.0 2.0; 3.0 4.0] .- [5.0 6.0; 7.0 8.0]
        end
    end

    @testset "mul! (5-argument α, β form)" begin
        𝒯 = Taylor(1)
        A = LinearOperator(𝒯, 𝒯, [1.0 2.0; 3.0 4.0])
        B = LinearOperator(𝒯, 𝒯, [5.0 6.0; 7.0 8.0])
        AB = [1.0 2.0; 3.0 4.0] * [5.0 6.0; 7.0 8.0]

        # α=1, β=0: pure overwrite; seed with Inf to catch unwritten entries
        C = LinearOperator(𝒯, 𝒯, fill(Inf, 2, 2))
        @test mul!(C, A, B, true, false) == A * B == LinearOperator(𝒯, 𝒯, AB)

        # α, β accumulate: C_new = α*(A*B) + β*C_old
        Cacc = LinearOperator(𝒯, 𝒯, [100.0 100.0; 100.0 100.0])
        mul!(Cacc, A, B, 2.0, 3.0)
        @test coefficients(Cacc) == 2.0 .* AB .+ 3.0 .* [100.0 100.0; 100.0 100.0]

        # generic `AbstractLinearOperator` operands (here two `UniformScalingOperator`s):
        # both get projected onto C's domain/codomain before the product
        Cs = LinearOperator(𝒯, 𝒯, fill(Inf, 2, 2))
        mul!(Cs, UniformScalingOperator(2.0), UniformScalingOperator(3.0), 1.0, 0.0)
        @test coefficients(Cs) == [6.0 0.0; 0.0 6.0]

        Cs2 = LinearOperator(𝒯, 𝒯, fill(Inf, 2, 2))
        mul!(Cs2, UniformScalingOperator(3.0), B, 1.0, 0.0)
        @test coefficients(Cs2) == 3.0 .* [5.0 6.0; 7.0 8.0]
    end

    @testset "cartesian-block operator arithmetic" begin
        𝒯₁ = Taylor(1) # dim 2

        @testset "CartesianPower: +, -, *, mul! on a matching block structure" begin
            CP = CartesianPower(𝒯₁, 2) # dim 4, two blocks of Taylor(1)

            Amat = [1.0 2.0 5.0 6.0; 3.0 4.0 7.0 8.0; 9.0 10.0 13.0 14.0; 11.0 12.0 15.0 16.0]
            Bmat = Matrix{Float64}(I, 4, 4)
            A = LinearOperator(CP, CP, Amat)
            B = LinearOperator(CP, CP, Bmat)

            @test coefficients(A + B) == Amat .+ Bmat
            @test A + B == A - (-B) == ladd!(copy(A), copy(B))
            @test coefficients(A * B) == Amat * Bmat == Amat # B is the identity
            out = LinearOperator(CP, CP, fill(Inf, 4, 4))
            @test mul!(out, A, B, true, false) == A * B

            # each 2×2 diagonal block can be pulled out with `component`
            @test coefficients(component(A, 1, 1)) == Amat[1:2, 1:2]
            @test coefficients(component(A, 2, 2)) == Amat[3:4, 3:4]
        end

        @testset "CartesianPower: mismatched suborders adapt per block" begin
            𝒯₂ = Taylor(2) # dim 3
            CPa = CartesianPower(𝒯₁, 2) # dim 4 (two blocks of Taylor(1))
            CPb = CartesianPower(𝒯₂, 2) # dim 6 (two blocks of Taylor(2))

            # the two diagonal blocks reuse the exact Taylor(1)/Taylor(2) example verified
            # above (a Taylor(1)→Taylor(1) block plus a Taylor(2)→Taylor(1) block); the
            # off-diagonal blocks are zero, so the sum is again block-diagonal
            Ablk = [1.0 2.0; 3.0 4.0]
            Bblk = [1.0 2.0 3.0; 4.0 5.0 6.0]
            Z22, Z23 = zeros(2, 2), zeros(2, 3)
            A = LinearOperator(CPa, CPa, [Ablk Z22; Z22 Ablk])
            B = LinearOperator(CPb, CPa, [Bblk Z23; Z23 Bblk])

            expected_blk = [2.0 4.0 3.0; 7.0 9.0 6.0] # the Taylor(1)+Taylor(2) sum, verified above
            @test coefficients(A + B) == [expected_blk Z23; Z23 expected_blk]
            @test coefficients(component(A + B, 1, 1)) == expected_blk
            @test all(iszero, coefficients(component(A + B, 1, 2)))

            # natural product A*B (domain(A) matches codomain(B) exactly): standard block
            # matrix product, each diagonal block is Ablk*Bblk
            expected_prod_blk = Ablk * Bblk
            @test coefficients(A * B) == [expected_prod_blk Z23; Z23 expected_prod_blk]

            # `mul!` into a smaller preallocated C (CPa → CPa instead of the natural CPb
            # domain) truncates each block's contracted columns to the first 2 (order 1)
            Csmall = LinearOperator(CPa, CPa, fill(Inf, 4, 4))
            mul!(Csmall, A, B, 1.0, 0.0)
            trunc_blk = Ablk * Bblk[:, 1:2]
            Z22b = zeros(2, 2)
            @test coefficients(Csmall) == [trunc_blk Z22b; Z22b trunc_blk]

            # general β (≠ 0, 1): the WHOLE preallocated C is rescaled first, then each
            # diagonal block adds α*(truncated block product); off-diagonal blocks only
            # ever see the rescaling (their block product is zero)
            Cgen = LinearOperator(CPa, CPa, fill(10.0, 4, 4))
            mul!(Cgen, A, B, 3.0, 2.0)
            expected_gen_blk = 2.0 .* fill(10.0, 2, 2) .+ 3.0 .* trunc_blk
            expected_gen_offblk = 2.0 .* fill(10.0, 2, 2)
            @test coefficients(Cgen) == [expected_gen_blk expected_gen_offblk; expected_gen_offblk expected_gen_blk]
        end

        @testset "mixed VectorSpace/CartesianSpace multiplication branches" begin
            # a single family of 4 operators exercises every mixed `_mul!` dispatch:
            # Bfull: CP → CP (pure cartesian), G: CP → 𝕂 (row functional),
            # K: 𝕂 → CP (column/sequence), H: 𝕂 → 𝕂 (plain scalar)
            CP = CartesianPower(𝒯₁, 2) # dim 4
            S = ScalarSpace()

            Bmat = Float64[1 0 2 0; 0 1 0 2; 3 0 1 0; 0 3 0 1]
            Gmat = Float64[1 2 3 4]
            Kmat = reshape(Float64[2, 3, 5, 7], 4, 1)
            Hmat = reshape(Float64[3.0], 1, 1)
            Bfull = LinearOperator(CP, CP, Bmat)
            G = LinearOperator(CP, S, Gmat)
            K = LinearOperator(S, CP, Kmat)
            H = LinearOperator(S, S, Hmat)

            @test coefficients(H * G) == Hmat * Gmat # 𝕂 ← CP
            @test coefficients(G * K) == Gmat * Kmat # 𝕂 ← 𝕂 (both operands cartesian-adjacent)
            @test coefficients(K * H) == Kmat * Hmat # CP ← 𝕂
            @test coefficients(G * Bfull) == Gmat * Bmat # 𝕂 ← CP, contracted over CP
            @test coefficients(Bfull * K) == Bmat * Kmat # CP ← 𝕂, contracted over CP
            @test coefficients(K * G) == Kmat * Gmat # CP ← CP, rank-1 outer product
            @test coefficients(Bfull * Bfull) == Bmat * Bmat # pure cartesian product

            # in-place α,β forms match the out-of-place products, seeded with Inf
            outHG = LinearOperator(CP, S, fill(Inf, 1, 4))
            @test mul!(outHG, H, G, true, false) == H * G
            outKG = LinearOperator(CP, CP, fill(Inf, 4, 4))
            @test mul!(outKG, K, G, true, false) == K * G
        end

        @testset "mixed VectorSpace/CartesianSpace _mul!: mismatched-shape (else) branches" begin
            # each sub-testset forces one specific mixed `_mul!` method away from its
            # domain/codomain-matching fast path (`mul!(coefficients(C), ...)`) into its
            # per-block (`component`) else branch, by giving C (or a sub-space) a different
            # order than the operand it would otherwise match exactly.

            @testset "codomain(A)/domain(B) both cartesian, C cartesian with mismatched codomain" begin
                S = Taylor(0)
                CP = CartesianPower(Taylor(1), 2) # dim 4
                CPc = CartesianPower(Taylor(0), 2) # dim 2, mismatched codomain(C) vs codomain(A)
                A = LinearOperator(S, CP, reshape([1.0, 2.0, 3.0, 4.0], 4, 1))
                B = LinearOperator(CP, S, reshape([10.0, 20.0, 30.0, 40.0], 1, 4))
                C = LinearOperator(CP, CPc, fill(Inf, 2, 4))
                mul!(C, A, B, 1.0, 0.0)
                # block (i,j): α * A[i-th block, row0] * B[:, j-th block] (β=0, no prior term)
                @test coefficients(C) == [10.0 20.0 30.0 40.0; 30.0 60.0 90.0 120.0]
            end

            @testset "domain(C) cartesian mismatched, codomain(A)/domain(B) cartesian match" begin
                CP_A = CartesianPower(Taylor(1), 2) # dim 4, domain(A) == codomain(B)
                S0 = Taylor(0)
                CP_C = CartesianPower(Taylor(0), 2) # dim 2, mismatched vs domain(B)
                A = LinearOperator(CP_A, S0, [1.0 2.0 3.0 4.0])
                B = LinearOperator(CP_A, CP_A, Matrix{Float64}(I, 4, 4))

                C = LinearOperator(CP_C, S0, fill(Inf, 1, 2))
                mul!(C, A, B, 1.0, 0.0)
                @test coefficients(C) == [1.0 3.0] # A's block-0 entries (B is the identity)

                Cacc = LinearOperator(CP_C, S0, [5.0 7.0])
                mul!(Cacc, A, B, 1.0, 2.0)
                @test coefficients(Cacc) == 2.0 .* [5.0 7.0] .+ [1.0 3.0]
            end

            @testset "domain(A) plain/domain(C) cartesian mismatched, B pure plain" begin
                S = Taylor(0)
                A = LinearOperator(S, S, reshape([5.0], 1, 1))
                CP = CartesianPower(Taylor(1), 2) # dim 4, domain(B) == codomain(A)... actually
                B = LinearOperator(CP, S, [10.0 20.0 30.0 40.0])
                CPc = CartesianPower(Taylor(0), 2) # dim 2, mismatched vs domain(B)
                C = LinearOperator(CPc, S, fill(Inf, 1, 2))
                mul!(C, A, B, 1.0, 0.0)
                # block j: α * A * B[j-th block, index 0] = 5 * B[block j, 0]
                @test coefficients(C) == [50.0 150.0]
            end

            @testset "domain(A)/codomain(B) cartesian match, domain(C) plain mismatched" begin
                CP_dom = CartesianPower(Taylor(1), 2) # dim 4, domain(A) == codomain(B)
                CP_codomA = CartesianPower(Taylor(0), 2) # dim 2, codomain(A) == codomain(C)
                S0 = Taylor(0) # domain(B)
                S1 = Taylor(1) # domain(C), mismatched vs domain(B)
                A = LinearOperator(CP_dom, CP_codomA, [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0])
                B = LinearOperator(S0, CP_dom, reshape([10.0, 20.0, 30.0, 40.0], 4, 1))

                C = LinearOperator(S1, CP_codomA, fill(Inf, 2, 2))
                mul!(C, A, B, 1.0, 0.0)
                # only domain-index 0 of C (column 1) is ever written by the block recursion
                # (since domain(B) = Taylor(0) only has index 0); column 2 stays at β*0 = 0
                @test coefficients(C) == [300.0 0.0; 700.0 0.0]

                Cacc = LinearOperator(S1, CP_codomA, [1.0 2.0; 3.0 4.0])
                mul!(Cacc, A, B, 1.0, 2.0)
                @test coefficients(Cacc) == [302.0 4.0; 706.0 8.0]
            end

            @testset "codomain(A)/codomain(C) cartesian match, domain(A) plain mismatched vs domain(B)" begin
                S = Taylor(0)
                CP = CartesianPower(Taylor(1), 2) # dim 4, codomain(A) == codomain(C)
                A = LinearOperator(S, CP, reshape([1.0, 2.0, 3.0, 4.0], 4, 1))
                B = LinearOperator(S, S, reshape([10.0], 1, 1))
                C = LinearOperator(Taylor(1), CP, fill(Inf, 4, 2)) # domain(C) ≠ domain(B)
                mul!(C, A, B, 1.0, 0.0)
                # per block i: only domain-index 0 (column 1) of C_i is ever written
                @test coefficients(C) == [10.0 0.0; 20.0 0.0; 30.0 0.0; 40.0 0.0]
            end

            @testset "domain(A) == codomain(B) (both cartesian), C plain mismatched" begin
                CP = CartesianPower(Taylor(1), 2) # dim 4, domain(A) == codomain(B)
                S = Taylor(0)
                Ta1 = Taylor(1) # domain(C), mismatched vs domain(B)
                A = LinearOperator(CP, S, [1.0 2.0 3.0 4.0])
                B = LinearOperator(S, CP, reshape([10.0, 20.0, 30.0, 40.0], 4, 1))

                C = LinearOperator(Ta1, S, fill(Inf, 1, 2))
                mul!(C, A, B, 1.0, 0.0)
                dotAB = 1.0 * 10.0 + 2.0 * 20.0 + 3.0 * 30.0 + 4.0 * 40.0 # = A ⋅ B = 300
                # only domain-index 0 (column 1) of C is written; column 2 stays at β*0 = 0
                @test coefficients(C) == [dotAB 0.0]

                Cacc = LinearOperator(Ta1, S, [5.0 7.0])
                mul!(Cacc, A, B, 1.0, 2.0)
                @test coefficients(Cacc) == [2.0 * 5.0 + dotAB 2.0 * 7.0]
            end

            @testset "domain(A) ≠ codomain(B) (both cartesian, mismatched order)" begin
                S = Taylor(0)
                domA = CartesianPower(Taylor(0), 2) # dim 2
                codomB = CartesianPower(Taylor(1), 2) # dim 4, different suborder than domA
                A = LinearOperator(domA, S, [2.0 5.0])
                B = LinearOperator(S, codomB, reshape([10.0, 20.0, 30.0, 40.0], 4, 1))

                C = LinearOperator(S, S, fill(Inf, 1, 1))
                mul!(C, A, B, 1.0, 0.0)
                # per block k: α * A[k] * (row-0 of B's k-th block); accumulated over k=1,2
                @test coefficients(C) == reshape([2.0 * 10.0 + 5.0 * 30.0], 1, 1)

                Cacc = LinearOperator(S, S, reshape([3.0], 1, 1))
                mul!(Cacc, A, B, 1.0, 2.0)
                @test coefficients(Cacc) == reshape([2.0 * 3.0 + (2.0 * 10.0 + 5.0 * 30.0)], 1, 1)
            end
        end

        @testset "radd!/rsub!/ladd!/lsub! between two CartesianSpace LinearOperators" begin
            CP = CartesianPower(𝒯₁, 2) # dim 4
            Amat = [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0; 9.0 10.0 11.0 12.0; 13.0 14.0 15.0 16.0]
            Bmat = Matrix{Float64}(I, 4, 4)
            A = LinearOperator(CP, CP, Amat)
            B = LinearOperator(CP, CP, Bmat)

            # exact domain/codomain match: `radd!`/`rsub!` take the elementwise fast path
            @test coefficients(radd!(copy(A), B)) == Amat .+ Bmat
            @test coefficients(rsub!(copy(A), B)) == Amat .- Bmat

            # mismatched suborders (reusing the CPa/CPb block example above): `ladd!`/`lsub!`
            # fall back to the per-block loop, each block resolved by the plain (non-cartesian)
            # `_ladd!`/`_lsub!`
            𝒯₂ = Taylor(2)
            CPa = CartesianPower(𝒯₁, 2)
            CPb = CartesianPower(𝒯₂, 2)
            Ablk = [1.0 2.0; 3.0 4.0]
            Bblk = [1.0 2.0 3.0; 4.0 5.0 6.0]
            Z22, Z23 = zeros(2, 2), zeros(2, 3)
            Acart = LinearOperator(CPa, CPa, [Ablk Z22; Z22 Ablk])
            Bcart = LinearOperator(CPb, CPa, [Bblk Z23; Z23 Bblk])

            expected_ladd_blk = [2.0 4.0 3.0; 7.0 9.0 6.0] # the Taylor(1)+Taylor(2) sum, verified above
            @test coefficients(ladd!(copy(Acart), copy(Bcart))) == [expected_ladd_blk Z23; Z23 expected_ladd_blk]

            # `lsub!` on a mismatched block: `-Bblk` then Ablk is added back on the shared columns
            expected_lsub_blk = (-Bblk) .+ [Ablk zeros(2, 1)]
            @test coefficients(lsub!(copy(Acart), copy(Bcart))) == [expected_lsub_blk Z23; Z23 expected_lsub_blk]
        end

        @testset "Diagonal{UniformScaling} block identity" begin
            CP = CartesianPower(𝒯₁, 2) # dim 4
            Amat = Float64[1 0 2 0; 0 1 0 2; 3 0 1 0; 0 3 0 1]
            A = LinearOperator(CP, CP, Amat)

            # each diagonal block gets its own scaling: block 1 gets +2I, block 2 gets +3I
            Acopy = copy(A)
            radd!(Acopy, RadiiPolynomial.LinearAlgebra.Diagonal([2.0 * I, 3.0 * I]))
            expected = copy(Amat)
            expected[1:2, 1:2] .+= 2.0 .* [1.0 0.0; 0.0 1.0]
            expected[3:4, 3:4] .+= 3.0 .* [1.0 0.0; 0.0 1.0]
            @test coefficients(Acopy) == expected
        end

        @testset "Diagonal{UniformScalingOperator} with a nested CartesianSpace block" begin
            # domain(A) has 2 top-level blocks: the first is itself a CartesianSpace
            # (CartesianPower(Taylor(1), 2), dim 4), the second is plain (Taylor(0), dim 1);
            # `_deep_nspaces` of the whole domain is 2 + 1 = 3, matching `length(J.diag)`
            dom = CartesianPower(𝒯₁, 2) × Taylor(0) # dim 5
            A = LinearOperator(dom, dom, Matrix{Float64}(I, 5, 5))
            J = RadiiPolynomial.LinearAlgebra.Diagonal([UniformScalingOperator(2.0), UniformScalingOperator(3.0), UniformScalingOperator(5.0)])
            r = radd!(copy(A), J)

            # component 1 (the nested CartesianSpace) recurses: its own 2 sub-blocks get
            # +2I and +3I respectively (indices 1,2 and 3,4); component 2 (Taylor(0)) gets +5I
            expected = Matrix{Float64}(RadiiPolynomial.LinearAlgebra.Diagonal([3.0, 3.0, 4.0, 4.0, 6.0]))
            @test coefficients(r) == expected
        end
    end

    @testset "interval and complex coefficient promotion" begin
        𝒯 = Taylor(1)
        Amat = [1.0 2.0; 3.0 4.0]
        Bmat = [5.0 6.0; 7.0 8.0]
        A = LinearOperator(𝒯, 𝒯, Amat)
        Ai = interval(A) # `interval(::LinearOperator)` promotes domain, codomain and coefficients
        Bi = LinearOperator(𝒯, 𝒯, interval.(Bmat))

        Si = Ai + Bi
        @test eltype(Si) == Interval{Float64}
        @test all(isequal_interval.(coefficients(Si), interval.(Amat .+ Bmat)))

        Pi = Ai * Bi
        expected_prod = Amat * Bmat
        @test all(in_interval.(expected_prod, coefficients(Pi)))

        # `mul!` into a sentinel-seeded interval output still gets fully overwritten
        Ci = LinearOperator(𝒯, 𝒯, fill(interval(1e10), 2, 2))
        mul!(Ci, Ai, Bi, true, false)
        @test all(in_interval.(expected_prod, coefficients(Ci)))

        # ComplexF64 coefficients
        Ac = LinearOperator(𝒯, 𝒯, ComplexF64[1.0+1.0im 2.0; 3.0 4.0-2.0im])
        Bc = LinearOperator(𝒯, 𝒯, ComplexF64[1.0 0.0; 0.0 1.0])
        @test coefficients(Ac + Bc) == ComplexF64[2.0+1.0im 2.0; 3.0 5.0-2.0im]
        @test coefficients(Ac * Bc) == coefficients(Ac) # Bc is the identity

        # Complex{Interval{Float64}} promotion
        Aci = LinearOperator(𝒯, 𝒯, complex.(interval.(Amat)))
        Sci = Aci + Ai
        @test eltype(Sci) == Complex{Interval{Float64}}
        @test all(isequal_interval.(real.(coefficients(Sci)), interval.(2.0 .* Amat)))
        @test all(isequal_interval.(imag.(coefficients(Sci)), interval(0.0)))
    end

    @testset "error paths: incompatible spaces" begin
        𝒯 = Taylor(1)
        𝒞 = Chebyshev(1)
        A = LinearOperator(𝒯, 𝒯, [1.0 2.0; 3.0 4.0])
        Bc = LinearOperator(𝒞, 𝒞, [1.0 2.0; 3.0 4.0])

        @test_throws ArgumentError A * Bc # domain(A)=Taylor vs codomain(Bc)=Chebyshev
        @test_throws ArgumentError A / Bc # domain(A) vs domain(Bc)
        @test_throws ArgumentError A \ Bc # codomain(A) vs codomain(Bc)

        C = zeros(𝒞, 𝒞)
        @test_throws ArgumentError add!(C, A, A)
        @test_throws ArgumentError sub!(C, A, A)

        # `CartesianPower`s with a different number of blocks cannot combine under `+`/`-`
        CP2 = CartesianPower(𝒯, 2)
        CP3 = CartesianPower(𝒯, 3)
        A2 = LinearOperator(CP2, CP2, Matrix{Float64}(I, 4, 4))
        A3 = LinearOperator(CP3, CP3, Matrix{Float64}(I, 6, 6))
        @test_throws ArgumentError A2 + A3
    end
end
