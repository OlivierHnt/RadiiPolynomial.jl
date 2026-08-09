@testset "LinearOperator" begin

    @testset "Construction, accessors, and dimension checks" begin
        𝒯 = Taylor(1) # dim 2
        𝒞 = Chebyshev(2) # dim 3
        M = [1.0 2.0 3.0 ; 4.0 5.0 6.0] # 2×3: codomain dim 2, domain dim 3
        A = LinearOperator(𝒞, 𝒯, M)

        @test domain(A) == 𝒞
        @test codomain(A) == 𝒯
        @test coefficients(A) == M
        @test eltype(A) == Float64
        @test eltype(typeof(A)) == Float64
        @test size(A) == (2, 3)
        @test size(A, 1) == 2
        @test size(A, 2) == 3
        @test length(A) == 6
        @test firstindex(A, 1) == 0 && lastindex(A, 1) == 1 # indices(𝒯)
        @test firstindex(A, 2) == 0 && lastindex(A, 2) == 2 # indices(𝒞)
        @test firstindex(A, 3) == 1 && lastindex(A, 3) == 1 # fallback for i ∉ {1,2}
        @test collect(A) == vec(M) # `iterate` forwards to `coefficients`

        # codomain and domain dimensions must match the coefficient matrix size
        @test_throws DimensionMismatch LinearOperator(𝒞, 𝒯, zeros(3, 3))
        @test_throws DimensionMismatch LinearOperator(𝒞, 𝒯, zeros(2, 2))

        # `order`/`frequency` forward to domain/codomain when they are cartesian
        # (a plain `BaseSpace` such as `Taylor` has no 2-argument `order`/`frequency`)
        ℱ = Fourier(1, 1.5)
        B = LinearOperator(ℱ^2, ℱ^3, zeros(9, 6))
        @test order(B) == (order(ℱ^2), order(ℱ^3)) == ([1, 1], [1, 1, 1])
        @test order(B, 1, 2) == (order(ℱ^2, 2), order(ℱ^3, 1)) == (1, 1)
        @test frequency(B) == (frequency(ℱ^2), frequency(ℱ^3)) == ([1.5, 1.5], [1.5, 1.5, 1.5])
        @test frequency(B, 1, 2) == (1.5, 1.5)
    end

    @testset "Alternate constructors" begin
        # `LinearOperator(coefficient::Number)`
        s = LinearOperator(5)
        @test domain(s) == codomain(s) == ScalarSpace()
        @test coefficients(s) == [5;;]

        # `LinearOperator(coefficients::AbstractMatrix)` ≡ ScalarSpace()^n → ScalarSpace()^m
        m = LinearOperator([1 2 3 ; 4 5 6])
        @test domain(m) == ScalarSpace()^3
        @test codomain(m) == ScalarSpace()^2
        @test coefficients(m) == [1 2 3 ; 4 5 6]

        # `LinearOperator(a::Sequence)`: column operator ScalarSpace() → space(a)
        𝒯 = Taylor(1)
        a = Sequence(𝒯, [1.0, 2.0])
        Aa = LinearOperator(a)
        @test domain(Aa) == ScalarSpace()
        @test codomain(Aa) == 𝒯
        @test coefficients(Aa) == reshape([1.0, 2.0], 2, 1)

        # round-trip `Sequence(::LinearOperator)`
        @test Sequence(Aa) == a
    end

    @testset "domain/codomain adaptation and compatibility checks" begin
        𝒯₁ = Taylor(1)
        ℱ₁ = Fourier(1, 1.0)
        A = LinearOperator(𝒯₁, ℱ₁, zeros(3, 2))

        # `domain(A, s)`/`codomain(A, s)` return the fixed domain/codomain of `A` once
        # compatibility of `s` with the other side has been checked
        @test domain(A, ℱ₁) == 𝒯₁ # any Fourier is compatible with codomain(A) = ℱ₁
        @test domain(A, ScalarSpace()) == 𝒯₁ # ScalarSpace is compatible with any SequenceSpace
        @test codomain(A, 𝒯₁) == ℱ₁
        @test codomain(A, ScalarSpace()) == ℱ₁

        # incompatible frequency / incompatible space families throw `ArgumentError`
        @test_throws ArgumentError domain(A, Fourier(1, 2.0))
        @test_throws ArgumentError codomain(A, Fourier(1, 3.0))
    end

    @testset "getindex / setindex! in numeric index coordinates" begin
        𝒯₂ = Taylor(2) # indices 0,1,2
        𝒯₁ = Taylor(1) # indices 0,1
        M = Float64[1 2 3 ; 4 5 6] # rows = codomain (𝒯₁), cols = domain (𝒯₂)
        A = LinearOperator(𝒯₂, 𝒯₁, copy(M))

        @test A[0, 0] == 1.0
        @test A[0, 2] == 3.0
        @test A[1, 1] == 5.0
        @test A[0:1, 0] == [1.0, 4.0] # range × integer
        @test A[:, 0] == [1.0, 4.0] # colon over the full codomain
        @test A[0, :] == [1.0, 2.0, 3.0]

        @test_throws BoundsError A[5, 0] # 5 ∉ indices(domain(A))
        @test_throws BoundsError A[0, 9] # 9 ∉ indices(codomain(A))

        B = copy(A)
        B[1, 2] = 999.0
        @test B[1, 2] == 999.0
        @test A[1, 2] == 6.0 # `copy` does not alias

        # `.=` broadcasting uses `dotview`/`view` under the hood
        C = copy(A)
        C[:, 0] .= [10.0, 20.0]
        @test coefficients(C)[:, 1] == [10.0, 20.0]
    end

    @testset "space-based getindex / view (sub-block selection)" begin
        𝒯₃ = Taylor(3) # indices 0,1,2,3
        B = LinearOperator(𝒯₃, 𝒯₃, Float64[10i+j for i ∈ 0:3, j ∈ 0:3]) # B[i,j] = 10i+j

        sub = B[Taylor(1), Taylor(1)] # restrict to indices 0,1 on both sides
        @test domain(sub) == codomain(sub) == Taylor(1)
        @test coefficients(sub) == [0.0 1.0 ; 10.0 11.0] # B[i,j], i,j ∈ {0,1}

        # `getindex` with one space and one plain index is explicitly unsupported
        @test_throws ErrorException B[Taylor(1), 0]
        @test_throws ErrorException B[0, Taylor(1)]

        # `view` gives the same values but stays glued to the parent's coefficients
        vsub = view(B, Taylor(1), Taylor(1))
        @test coefficients(vsub) == coefficients(sub)
        vsub[0, 0] = -1.0
        @test B[0, 0] == -1.0 # mutated through the view
        @test coefficients(sub)[1, 1] == 0.0 # `getindex` copy is unaffected

        @test_throws ErrorException view(B, Taylor(1), 0)
        @test_throws ErrorException view(B, 0, Taylor(1))
    end

    @testset "component, eachcomponent, unpack: cartesian domain and codomain" begin
        dom = Taylor(1) × Taylor(0) # component dims 2,1 (total 3)
        codom = Taylor(0)^2 # component dims 1,1 (total 2)
        M = Float64[1 2 3 ; 4 5 6]
        A = LinearOperator(dom, codom, copy(M))

        c11 = component(A, 1, 1)
        @test domain(c11) == Taylor(1) && codomain(c11) == Taylor(0)
        @test coefficients(c11) == [1.0 2.0]

        c12 = component(A, 1, 2)
        @test domain(c12) == Taylor(0) && codomain(c12) == Taylor(0)
        @test coefficients(c12) == reshape([3.0], 1, 1)

        c21 = component(A, 2, 1)
        @test coefficients(c21) == [4.0 5.0]

        U = unpack(A)
        @test size(U) == (2, 2)
        @test U == collect(eachcomponent(A))
        @test all(coefficients(U[i,j]) == coefficients(component(A,i,j)) for i ∈ 1:2, j ∈ 1:2)

        # mutating a component (or an unpacked entry) mutates the parent: view semantics
        c11[0, 0] = 999.0
        @test A[1, 1] == 999.0 # (1,1) is a plain `CartesianSpace` position, not a `Taylor` index
        U[2, 2][0, 0] = 555.0
        @test A[2, 3] == 555.0

        # ranges and `:` also select components
        cr = component(A, 1:2, 1)
        @test domain(cr) == Taylor(1) && codomain(cr) == CartesianProduct(Taylor(0), Taylor(0))
        cc = component(A, :, 1)
        @test coefficients(cc) == coefficients(cr)
    end

    @testset "component, eachcomponent, unpack: one-sided cartesian space" begin
        # cartesian domain, plain codomain
        A = LinearOperator(Taylor(1) × Taylor(0), Taylor(2), Float64[1 2 3 ; 4 5 6 ; 7 8 9])
        a1 = component(A, 1)
        a2 = component(A, 2)
        @test domain(a1) == Taylor(1) && codomain(a1) == Taylor(2)
        @test coefficients(a1) == [1.0 2.0 ; 4.0 5.0 ; 7.0 8.0]
        @test domain(a2) == Taylor(0) && codomain(a2) == Taylor(2)
        @test coefficients(a2) == reshape([3.0, 6.0, 9.0], 3, 1)
        Ua = unpack(A)
        @test size(Ua) == (1, 2)
        @test Ua == collect(eachcomponent(A))

        # plain domain, cartesian codomain
        B = LinearOperator(Taylor(2), Taylor(1) × Taylor(0), Float64[1 2 3 ; 4 5 6 ; 7 8 9])
        b1 = component(B, 1)
        b2 = component(B, 2)
        @test domain(b1) == Taylor(2) && codomain(b1) == Taylor(1)
        @test coefficients(b1) == [1.0 2.0 3.0 ; 4.0 5.0 6.0]
        @test domain(b2) == Taylor(2) && codomain(b2) == Taylor(0)
        @test coefficients(b2) == [7.0 8.0 9.0]
        Ub = unpack(B)
        @test size(Ub) == (2, 1)
        @test Ub == collect(eachcomponent(B))
    end

    @testset "copy, zero, one, similar, fill, fill!" begin
        𝒯₂ = Taylor(2) # dim 3
        𝒯₃ = Taylor(3) # dim 4
        A = LinearOperator(𝒯₂, 𝒯₃, Float64[1 2 3 ; 4 5 6 ; 7 8 9 ; 10 11 12])

        Ac = copy(A)
        @test Ac == A
        Ac[0, 0] = -1.0
        @test A[0, 0] == 1.0 # `copy` does not alias

        Az = zero(A)
        @test domain(Az) == 𝒯₂ && codomain(Az) == 𝒯₃
        @test iszero(Az)
        @test !iszero(A)

        Ao = one(A)
        # `one` puts the identity on `domain(A) ∩ codomain(A) = Taylor(2)`, the rest is zero
        @test coefficients(Ao) == [1.0 0.0 0.0 ; 0.0 1.0 0.0 ; 0.0 0.0 1.0 ; 0.0 0.0 0.0]

        As = similar(A)
        @test domain(As) == domain(A) && codomain(As) == codomain(A)
        @test eltype(As) == Float64
        Asi = similar(A, Int)
        @test eltype(Asi) == Int

        F = fill(7.0, Taylor(1), Taylor(0))
        @test domain(F) == Taylor(1) && codomain(F) == Taylor(0)
        @test coefficients(F) == [7.0 7.0]
        fill!(F, 3.0)
        @test coefficients(F) == [3.0 3.0]

        # type-level `zero`/`one`/`complex`
        T = LinearOperator{Taylor,Taylor,Matrix{Float64}}
        z = zero(T)
        @test domain(z) == codomain(z) == Taylor(0) && coefficients(z) == [0.0;;]
        o = one(T)
        @test coefficients(o) == [1.0;;]
        @test complex(T) == LinearOperator{Taylor,Taylor,Matrix{ComplexF64}}
    end

    @testset "float, complex, real, imag, conj" begin
        ℱ = Fourier(1, 1) # Int frequency
        A = LinearOperator(ℱ, ℱ, Float64[1 2 3 ; 4 5 6 ; 7 8 9])
        Af = float(A)
        @test eltype(Af) == Float64
        @test domain(Af) == codomain(Af) == Fourier(1, 1.0) # frequency promoted to Float64
        @test coefficients(Af) == Float64[1 2 3 ; 4 5 6 ; 7 8 9]

        Ac = complex(A)
        @test eltype(Ac) == ComplexF64
        @test domain(Ac) == domain(A) # `complex` does not touch the spaces

        B = LinearOperator(Taylor(1), Taylor(1), ComplexF64[1+2im 3-1im ; 0+1im 2+0im])
        @test real(B) == LinearOperator(Taylor(1), Taylor(1), Float64[1 3 ; 0 2])
        @test imag(B) == LinearOperator(Taylor(1), Taylor(1), Float64[2 -1 ; 1 0])
        @test conj(B) == LinearOperator(Taylor(1), Taylor(1), ComplexF64[1-2im 3+1im ; 0-1im 2-0im])
        Bconj = copy(B)
        conj!(Bconj)
        @test Bconj == conj(B)

        # binary `complex(A, B)`: combines the coefficients elementwise (same domain/codomain)
        D = LinearOperator(Taylor(1), Taylor(0), [10.0 20.0])
        E = LinearOperator(Taylor(1), Taylor(0), [1.0 2.0])
        @test coefficients(complex(D, E)) == ComplexF64[10+1im 20+2im]
    end

    @testset "transpose, adjoint" begin
        A = LinearOperator(Taylor(1), Taylor(0), [1.0 2.0]) # 1×2
        At = transpose(A)
        @test domain(At) == Taylor(0) && codomain(At) == Taylor(1)
        @test coefficients(At) == reshape([1.0, 2.0], 2, 1)

        B = LinearOperator(Taylor(1), Taylor(0), ComplexF64[1+2im 3-1im])
        Ba = adjoint(B)
        @test coefficients(Ba) == reshape(ComplexF64[1-2im, 3+1im], 2, 1)

        # `transpose`/`adjoint` on a `Sequence` go through `LinearOperator(a)` first
        a = Sequence(Taylor(1), [1.0, 2.0])
        at = transpose(a)
        @test domain(at) == Taylor(1) && codomain(at) == ScalarSpace()
        @test coefficients(at) == [1.0 2.0]
    end

    @testset "equality, isapprox, iszero" begin
        A = LinearOperator(Taylor(1), Taylor(1), [1.0 2.0 ; 3.0 4.0])
        B = LinearOperator(Taylor(1), Taylor(1), [1.0 2.0 ; 3.0 4.0])
        C = LinearOperator(Taylor(1), Taylor(1), [1.0 2.0 ; 3.0 4.1])
        @test A == B
        @test A != C
        @test isapprox(A, B)
        @test !isapprox(A, C)
        @test isapprox(A, LinearOperator(Taylor(1), Taylor(1), [1.0 2.0 ; 3.0 4.0 + 1e-10]))
        @test !isapprox(A, LinearOperator(Taylor(1), Taylor(1), [1.0 2.0 ; 3.0 4.0 + 1e-10]); atol = 1e-12)
        @test !iszero(A)
        @test iszero(zeros(Taylor(1), Taylor(1)))
    end

    @testset "interval and complex numeric-type conversions" begin
        A = LinearOperator(Taylor(1), Taylor(1), [1.0 2.0 ; 3.0 4.0])
        Ai = interval(A)
        @test domain(Ai) == domain(A) && codomain(Ai) == codomain(A)
        @test all(isequal_interval.(coefficients(Ai), interval.(coefficients(A))))

        a = Sequence(Taylor(1), [10.0, 20.0])
        ai = interval(a)
        ci = Ai * ai
        # 1*10+2*20 = 50, 3*10+4*20 = 110, both exactly representable
        @test issubset_interval(interval(50.0), ci[0])
        @test issubset_interval(interval(110.0), ci[1])

        Ac = complex(A)
        ac = complex(a)
        @test Ac * ac == Sequence(Taylor(1), ComplexF64[50.0, 110.0])
    end

    @testset "getcoefficient with SymmetricSpace index coordinates" begin
        # `evensym(Fourier(1,1.0))`: Z₂ group {id, k ↦ -k} acting with unit amplitude and
        # zero phase (a₋ₖ = aₖ), representatives are the orbit maxima: indices 0,1
        ℱ = Fourier(1, 1.0)
        es = evensym(ℱ)
        @test indices(es) == 0:1

        @testset "plain domain/codomain queried in symmetric coordinates" begin
            # A[k,l] = 10(k+2) + (l+2), k,l ∈ {-1,0,1} → M[i,j] = 10i+j, i,j ∈ {1,2,3}
            M = Float64[1 2 3 ; 4 5 6 ; 7 8 9]
            A = LinearOperator(ℱ, ℱ, M)

            # A_sym[1,1] = (1/2)Σ_{k,l∈{-1,1}} A[k,l] = (A[1,1]+A[1,-1]+A[-1,1]+A[-1,-1])/2
            #            = (9+7+3+1)/2 = 10
            @test RadiiPolynomial.getcoefficient(A, (es, 1), (es, 1)) == 10.0 + 0.0im
            # A_sym[0,0] = A[0,0] (orbit of 0 is a singleton) = 5
            @test RadiiPolynomial.getcoefficient(A, (es, 0), (es, 0)) == 5.0 + 0.0im
            # A_sym[1,0] = (A[1,0]+A[-1,0])/2 = (8+2)/2 = 5
            @test RadiiPolynomial.getcoefficient(A, (es, 1), (es, 0)) == 5.0 + 0.0im
            # A_sym[0,1] = (A[0,1]+A[0,-1]) / |orbit(0)| = (6+4)/1 = 10 (the divisor is always
            # the size of the *codomain* orbit, here a singleton, unlike the α=1 case above)
            @test RadiiPolynomial.getcoefficient(A, (es, 0), (es, 1)) == 10.0 + 0.0im
        end

        @testset "symmetric domain/codomain queried in full (desymmetrized) coordinates" begin
            # B[(es i),(es j)] = N[i+1,j+1], N = [1 2 ; 3 4] (es indices 0,1)
            B = LinearOperator(es, es, Float64[1 2 ; 3 4])

            # desym(α=-1,β=-1): representative of -1 is 1 (both sides), factor 1;
            # value = B[1,1] / |orbit(1)| = 4/2 = 2
            @test RadiiPolynomial.getcoefficient(B, (ℱ, -1), (ℱ, -1)) == 2.0 - 0.0im
            # α=1 is already its own representative: same value 2
            @test RadiiPolynomial.getcoefficient(B, (ℱ, -1), (ℱ, 1)) == 2.0 - 0.0im
            # α=0,β=0: representative of 0 is 0, orbit is a singleton: B[0,0]/1 = 1
            @test RadiiPolynomial.getcoefficient(B, (ℱ, 0), (ℱ, 0)) == 1.0 - 0.0im
        end

        @testset "one side symmetric, the other plain (domain symmetric, codomain plain)" begin
            # domain symmetric (es), codomain plain (ℱ): only β needs desymmetrizing
            P = LinearOperator(es, ℱ, Float64[1 2 ; 3 4 ; 5 6]) # P[(ℱ i),(es j)] = N[i+1,j+1]
            # β=-1 desymmetrizes to representative 1 (factor 1, orbit size 2):
            # value = P[0,1]/2 = 4/2 = 2
            @test RadiiPolynomial.getcoefficient(P, (ℱ, 0), (ℱ, -1)) == 2.0 - 0.0im
            @test RadiiPolynomial.getcoefficient(P, (ℱ, 0), (ℱ, 1)) == 2.0 - 0.0im # β=1 is its own rep
        end

        @testset "one side symmetric, the other plain (codomain symmetric, domain plain)" begin
            # domain plain (ℱ), codomain symmetric (es): only α needs desymmetrizing
            Q = LinearOperator(ℱ, es, Float64[1 2 3 ; 4 5 6]) # Q[(es i),(ℱ j)] = N[i+1,j+1]
            # α=1 is already a representative (∈ indices(es)): Q[1,0] = 5
            @test RadiiPolynomial.getcoefficient(Q, (ℱ, 1), (ℱ, 0)) == 5.0 + 0.0im
            # α=-1 has representative 1 (factor 1) but -1 ∉ indices(es) = 0:1: factor_α * Q[1,0] = 5
            @test RadiiPolynomial.getcoefficient(Q, (ℱ, -1), (ℱ, 0)) == 5.0 + 0.0im
        end
    end

    @testset "lazy wrappers: domain/codomain adaptation" begin
        𝒯₁ = Taylor(1)
        𝒯₂ = Taylor(2)
        𝒯₃ = Taylor(3)
        A = LinearOperator(𝒯₂, 𝒯₃, zeros(4, 3)) # domain 𝒯₂, codomain 𝒯₃
        B = LinearOperator(𝒯₂, 𝒯₁, zeros(2, 3)) # domain 𝒯₂, codomain 𝒯₁

        @testset "UniformScalingOperator" begin
            J = UniformScalingOperator(2.0)
            @test domain(J, 𝒯₂) == 𝒯₂ && codomain(J, 𝒯₂) == 𝒯₂ # acts as the identity space-wise
            @test domain(J, EmptySpace()) == EmptySpace() # ambiguity-resolving method
            @test codomain(J, EmptySpace()) == EmptySpace()
            @test eltype(J) == Float64
            @test zero(J) == UniformScalingOperator(0.0)
            @test one(J) == UniformScalingOperator(1.0)
            @test UniformScalingOperator(I) == UniformScalingOperator(true) # `I` is re-exported
            Ji = interval(J)
            @test isequal_interval(Ji.λ, interval(2.0))
        end

        @testset "Negate" begin
            N = Negate(A)
            @test domain(N, Taylor(5)) == domain(A, Taylor(5)) == 𝒯₂
            @test codomain(N, 𝒯₂) == codomain(A, 𝒯₂) == 𝒯₃
        end

        @testset "Add" begin
            S = Add(A, B)
            # both A and B need a domain compatible with codomain 𝒯₅ ⊇ 𝒯₃,𝒯₁, giving 𝒯₂ ∪ 𝒯₂ = 𝒯₂
            @test domain(S, Taylor(5)) == 𝒯₂
            # given input 𝒯₂, A maps into 𝒯₃ and B into 𝒯₁: the union is 𝒯₃
            @test codomain(S, 𝒯₂) == 𝒯₃
        end

        @testset "ComposedOperator" begin
            C = LinearOperator(𝒯₁, 𝒯₂, zeros(3, 2)) # domain 𝒯₁, codomain 𝒯₂
            Comp = ComposedOperator(A, C) # outer = A (𝒯₂ → 𝒯₃), inner = C (𝒯₁ → 𝒯₂)
            @test Comp isa ComposedOperator
            # to land in codomain 𝒯₅: outer needs domain 𝒯₂ (⊇ requirement on 𝒯₅ trivial),
            # inner then needs domain 𝒯₁ so that its codomain (𝒯₂) matches outer's domain
            @test domain(Comp, Taylor(5)) == 𝒯₁
            # given input domain 𝒯₁: inner maps into 𝒯₂, outer then maps 𝒯₂ into 𝒯₃
            @test codomain(Comp, 𝒯₁) == 𝒯₃
        end
    end

    @testset "space+index tuple-coordinate getindex on LinearOperator" begin
        # `A[(codom,i),(dom,j)]` forwards to `getcoefficient`; a specific method on `LinearOperator`
        # resolves what would otherwise be an ambiguity with the plain `getindex(A::LinearOperator, α, β)`
        # method (untyped α, β also match 2-tuples). `RadiiPolynomial.getcoefficient(A, ..., ...)`
        # (used internally, e.g. by `project!`) is exercised directly in the symmetric-coordinates
        # tests above.
        A = LinearOperator(Taylor(2), Taylor(2), Float64[1 2 3 ; 4 5 6 ; 7 8 9])
        @test A[(Taylor(2), 0), (Taylor(2), 0)] == 1.0
        @test A[(Taylor(2), 1), (Taylor(2), 2)] == 6.0

        # both index-coordinate forms must agree on a `TensorSpace` domain/codomain, where a plain
        # index α is itself a tuple of integers (e.g. `(0,0)`) and hence does NOT match the
        # `Tuple{VectorSpace,Any}` signature (its first element is an `Int`, not a `VectorSpace`):
        # the two forms dispatch to different methods but must return the same coefficient.
        𝒮 = Taylor(1) ⊗ Chebyshev(1) # dim 4
        M = Matrix{Float64}(reshape(1:16, 4, 4)) # column-major
        B = LinearOperator(𝒮, 𝒮, M)
        @test B[(0, 0), (1, 1)] == 13.0 # plain tuple-of-integers coordinates
        @test B[(𝒮, (0, 0)), (𝒮, (1, 1))] == 13.0 # space+index tuple coordinates
        @test B[(0, 0), (1, 1)] == B[(𝒮, (0, 0)), (𝒮, (1, 1))]
    end

    @testset "grids of LinearOperators (parameter families, cf. sequences/fft.jl)" begin
        @testset "Chebyshev parameter: to_grid/to_coef round trip" begin
            s_par = Chebyshev(2)
            dom, codom = Taylor(1), Fourier(1, 1.0)
            new_dom, new_codom = RadiiPolynomial._zero_space(s_par) ⊗ dom, s_par ⊗ codom
            A = LinearOperator(new_dom, new_codom, reshape(collect(1.0:dimension(new_codom)*dimension(new_dom)), :, dimension(new_dom)))
            A_grid = to_grid(A, grid_size(s_par))
            @test A_grid isa Vector
            @test size(A_grid) == grid_size(s_par) == (3,)
            @test all(X -> (domain(X) == dom) & (codomain(X) == codom), A_grid)
            B = to_coef(A_grid, s_par)
            @test (domain(B) == new_dom) & (codomain(B) == new_codom)
            @test real.(coefficients(B)) ≈ coefficients(A) atol=1e-9
        end

        @testset "grid elements agree with sequence-wise partial evaluation" begin
            # applying the interpolated operator to a fixed sequence and evaluating
            # at a node equals applying the grid operator at that node
            s_par = Chebyshev(2)
            dom, codom = Taylor(1), Fourier(1, 1.0)
            new_dom, new_codom = RadiiPolynomial._zero_space(s_par) ⊗ dom, s_par ⊗ codom
            A = LinearOperator(new_dom, new_codom, reshape(collect(1.0:dimension(new_codom)*dimension(new_dom)), :, dimension(new_dom)))
            A_grid = to_grid(A, grid_size(s_par))
            v = Sequence(dom, [1.0, -2.0])
            v₀ = Sequence(new_dom, coefficients(v)) # the same `v`, seen as constant in the parameter
            m = only(grid_size(s_par))
            for k ∈ 1:m
                x_k = cospi((k-1)/(m-1))
                @test real.(coefficients(A_grid[k] * v)) ≈ coefficients(Evaluation(x_k, nothing) * (A * v₀)) atol=1e-9
            end
        end

        @testset "two parameter factors give a Matrix grid" begin
            s_par = Chebyshev(1) ⊗ Chebyshev(2)
            dom, codom = Taylor(1), Taylor(1)
            new_dom, new_codom = RadiiPolynomial._zero_space(s_par) ⊗ dom, s_par ⊗ codom
            A = LinearOperator(new_dom, new_codom, reshape(collect(1.0:dimension(new_codom)*dimension(new_dom)), :, dimension(new_dom)))
            A_grid = to_grid(A, grid_size(s_par))
            @test A_grid isa Matrix
            @test size(A_grid) == grid_size(s_par) == (2, 3)
            B = to_coef(A_grid, s_par)
            @test real.(coefficients(B)) ≈ coefficients(A) atol=1e-9
        end

        @testset "interval enclosure round trip" begin
            s_par = Chebyshev(1)
            dom, codom = Taylor(1), Chebyshev(1)
            new_dom, new_codom = RadiiPolynomial._zero_space(s_par) ⊗ dom, s_par ⊗ codom
            A = LinearOperator(new_dom, new_codom, interval.(reshape(collect(1.0:dimension(new_codom)*dimension(new_dom)), :, dimension(new_dom))))
            B = to_coef(to_grid(A, grid_size(s_par)), s_par)
            for (i, j) ∈ Iterators.product(1:size(coefficients(A), 1), 1:size(coefficients(A), 2))
                @test issubset_interval(coefficients(A)[i, j], real(coefficients(B)[i, j]))
            end
        end

        @testset "resampling an operator family on a finer grid" begin
            s_par = Chebyshev(2)
            dom, codom = Taylor(1), Fourier(1, 1.0)
            new_dom, new_codom = RadiiPolynomial._zero_space(s_par) ⊗ dom, s_par ⊗ codom
            A = LinearOperator(new_dom, new_codom, reshape(collect(1.0:dimension(new_codom)*dimension(new_dom)), :, dimension(new_dom)))
            A_fine = to_grid(A, (9,)) # 9 nodes instead of 3
            @test size(A_fine) == (9,)
            B = to_coef(A_fine, s_par)
            @test real.(coefficients(B)) ≈ coefficients(A) atol=1e-9
        end

        @testset "symmetric codomain: reduced rows round trip" begin
            s_par = Chebyshev(2)
            dom = Taylor(1)
            codom_sym = evensym(Fourier(2, 1.0))
            prod_sym = s_par ⊗ codom_sym
            new_dom = RadiiPolynomial._zero_space(s_par) ⊗ dom
            A = project(LinearOperator(new_dom, s_par ⊗ Fourier(2, 1.0), reshape(collect(1.0:30), 15, 2)), new_dom, prod_sym)
            A_grid = to_grid(A, grid_size(s_par))
            # grid elements carry the restricted symmetry group on the codomain
            @test all(X -> (domain(X) == dom) & (codomain(X) == codom_sym), A_grid)
            B = to_coef(A_grid, s_par)
            @test (domain(B) == new_dom) & (codomain(B) == prod_sym)
            @test real.(coefficients(B)) ≈ coefficients(A) atol=1e-9
        end

        @testset "every codomain factor discretized: the inner space collapses to `ScalarSpace`" begin
            # nothing is left as coefficients, so each node holds a functional
            s_par, dom = Chebyshev(2), Taylor(1)
            new_dom = RadiiPolynomial._zero_space(s_par) ⊗ dom
            A = LinearOperator(new_dom, s_par, reshape(collect(1.0:dimension(s_par)*dimension(new_dom)), :, dimension(new_dom)))
            A_grid = to_grid(A, grid_size(s_par))
            @test size(A_grid) == grid_size(s_par) == (3,)
            @test all(Z -> (domain(Z) == dom) & (codomain(Z) == ScalarSpace()), A_grid)

            v = Sequence(dom, [1.0, -2.0])
            v₀ = Sequence(new_dom, coefficients(v))
            Av = A * v₀ # the scalar family t -> A(t)v
            m = only(grid_size(s_par))
            for k ∈ 1:m
                x_k = cospi((k-1)/(m-1))
                @test real(only(coefficients(A_grid[k] * v))) ≈ Evaluation(x_k) * Av atol=1e-9
            end
        end

        @testset "empty inner domain: reduces to the sequence case" begin
            # nothing is left of the domain, so the family is a family of columns,
            # that is a family of sequences: no trivial `ScalarSpace` wrapper is kept
            s_par, codom = Chebyshev(2), Fourier(1, 1.0)
            X₀ = RadiiPolynomial._zero_space(s_par)
            new_codom = s_par ⊗ codom
            coeffs = reshape(collect(1.0:dimension(new_codom)), :, 1)
            A_grid = to_grid(LinearOperator(X₀, new_codom, coeffs), grid_size(s_par))
            @test eltype(A_grid) <: Sequence
            @test all(x -> space(x) == codom, A_grid)
            @test A_grid == to_grid(Sequence(new_codom, vec(coeffs)), grid_size(s_par))

            # both sides collapse: plain numbers, as for a `Sequence`
            cB = reshape(collect(1.0:dimension(s_par)), :, 1)
            B_grid = to_grid(LinearOperator(X₀, s_par, cB), grid_size(s_par))
            @test B_grid isa Vector{ComplexF64}
            @test B_grid == to_grid(Sequence(s_par, vec(cB)), grid_size(s_par))
        end

        @testset "error paths" begin
            s_par = Chebyshev(1)
            # codomain does not start with the parameter factor
            A_bad = LinearOperator(Taylor(1), Fourier(1, 1.0), ones(3, 2))
            @test_throws ArgumentError to_grid(A_bad, (2,))
            # the leading factors of the domain must be the zero space of the parameter
            A_bad_dom = LinearOperator(Chebyshev(1) ⊗ Taylor(1), Chebyshev(2) ⊗ Fourier(1, 1.0), ones(9, 4))
            @test_throws ArgumentError to_grid(A_bad_dom, (3,))
            # mismatched domains/codomains in the grid
            X₁ = LinearOperator(Taylor(1), Taylor(1), ones(2, 2))
            X₂ = LinearOperator(Taylor(2), Taylor(1), ones(2, 3))
            @test_throws ArgumentError to_coef([X₁, X₂], s_par)
            # grid dimension must match the number of factors of `s`
            @test_throws ArgumentError to_coef(fill(X₁, 2, 2), s_par)
        end
    end

end
