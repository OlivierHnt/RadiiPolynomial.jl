# operator family over the parameter space `s_par`, with entries 1, 2, ... in column-major order
function _lo_family(s_par, dom, codom)
    new_dom, new_codom = RadiiPolynomial._zero_space(s_par) ⊗ dom, s_par ⊗ codom
    return LinearOperator(new_dom, new_codom, reshape(collect(1.0:dimension(new_codom)*dimension(new_dom)), :, dimension(new_dom)))
end

function _lo_test_roundtrip(A, s_par, A_grid = to_grid(A, grid_size(s_par)))
    B = to_coef(A_grid, s_par)
    @testset let s_par = s_par, dom = domain(A), codom = codomain(A)
        @test (domain(B) == dom) & (codomain(B) == codom)
        @test real.(coefficients(B)) ≈ coefficients(A) atol=1e-9
    end
end

# the grid operator at each Chebyshev node acts as the family evaluated at that node,
# `v` being read as constant in the parameter
function _lo_test_nodes(A, A_grid, v)
    v₀ = Sequence(domain(A), coefficients(v))
    m = length(A_grid)
    for k ∈ 1:m
        @test real.(coefficients(A_grid[k] * v)) ≈ coefficients(Evaluation(cospi((k-1)/(m-1)), nothing) * (A * v₀)) atol=1e-9
    end
end

_lo_test_enclosure(A, B) = @test all(issubset_interval.(coefficients(A), real.(coefficients(B))))

@testset "LinearOperator" begin

    @testset "Construction, accessors, and dimension checks" begin
        𝒯, 𝒞 = Taylor(1), Chebyshev(2)
        M = [1.0 2.0 3.0 ; 4.0 5.0 6.0]
        A = LinearOperator(𝒞, 𝒯, M)

        @test domain(A) == 𝒞
        @test codomain(A) == 𝒯
        @test coefficients(A) == M
        @test eltype(A) == eltype(typeof(A)) == Float64
        @test size(A) == (size(A, 1), size(A, 2)) == (2, 3)
        @test length(A) == 6
        @test (firstindex(A, 1), lastindex(A, 1)) == (0, 1)
        @test (firstindex(A, 2), lastindex(A, 2)) == (0, 2)
        @test (firstindex(A, 3), lastindex(A, 3)) == (1, 1) # any further dimension is trivial
        @test collect(A) == vec(M)

        @test_throws DimensionMismatch LinearOperator(𝒞, 𝒯, zeros(3, 3))
        @test_throws DimensionMismatch LinearOperator(𝒞, 𝒯, zeros(2, 2))

        # `order` and `frequency` forward to the domain and codomain
        ℱ = Fourier(1, 1.5)
        B = LinearOperator(ℱ^2, ℱ^3, zeros(9, 6))
        @test order(B) == (order(ℱ^2), order(ℱ^3)) == ([1, 1], [1, 1, 1])
        @test order(B, 1, 2) == (order(ℱ^2, 2), order(ℱ^3, 1)) == (1, 1)
        @test frequency(B) == (frequency(ℱ^2), frequency(ℱ^3)) == ([1.5, 1.5], [1.5, 1.5, 1.5])
        @test frequency(B, 1, 2) == (1.5, 1.5)
    end

    @testset "Alternate constructors" begin
        s = LinearOperator(5)
        @test domain(s) == codomain(s) == ScalarSpace()
        @test coefficients(s) == [5;;]

        m = LinearOperator([1 2 3 ; 4 5 6])
        @test domain(m) == ScalarSpace()^3
        @test codomain(m) == ScalarSpace()^2
        @test coefficients(m) == [1 2 3 ; 4 5 6]

        a = Sequence(Taylor(1), [1.0, 2.0])
        Aa = LinearOperator(a)
        @test domain(Aa) == ScalarSpace()
        @test codomain(Aa) == Taylor(1)
        @test coefficients(Aa) == reshape([1.0, 2.0], 2, 1)
        @test Sequence(Aa) == a
    end

    @testset "domain/codomain adaptation and compatibility checks" begin
        𝒯₁, ℱ₁ = Taylor(1), Fourier(1, 1.0)
        A = LinearOperator(𝒯₁, ℱ₁, zeros(3, 2))

        # the domain and codomain of `A` are fixed, the argument is only checked for compatibility
        @test domain(A, ℱ₁) == domain(A, ScalarSpace()) == 𝒯₁
        @test codomain(A, 𝒯₁) == codomain(A, ScalarSpace()) == ℱ₁
        @test_throws ArgumentError domain(A, Fourier(1, 2.0))
        @test_throws ArgumentError codomain(A, Fourier(1, 3.0))
    end

    @testset "getindex / setindex! in numeric index coordinates" begin
        # rows are indexed by the codomain, columns by the domain
        A = LinearOperator(Taylor(2), Taylor(1), Float64[1 2 3 ; 4 5 6])

        @test A[0, 0] == 1.0
        @test A[0, 2] == 3.0
        @test A[1, 1] == 5.0
        @test A[0:1, 0] == A[:, 0] == [1.0, 4.0]
        @test A[0, :] == [1.0, 2.0, 3.0]

        @test_throws BoundsError A[5, 0]
        @test_throws BoundsError A[0, 9]

        B = copy(A)
        B[1, 2] = 999.0
        @test B[1, 2] == 999.0
        @test A[1, 2] == 6.0 # `copy` does not alias

        C = copy(A)
        C[:, 0] .= [10.0, 20.0]
        @test coefficients(C)[:, 1] == [10.0, 20.0]
    end

    @testset "space-based getindex / view (sub-block selection)" begin
        B = LinearOperator(Taylor(3), Taylor(3), Float64[10i+j for i ∈ 0:3, j ∈ 0:3])

        sub = B[Taylor(1), Taylor(1)]
        @test domain(sub) == codomain(sub) == Taylor(1)
        @test coefficients(sub) == [0.0 1.0 ; 10.0 11.0]

        # a `view` shares the coefficients of its parent, `getindex` copies them
        vsub = view(B, Taylor(1), Taylor(1))
        @test coefficients(vsub) == coefficients(sub)
        vsub[0, 0] = -1.0
        @test B[0, 0] == -1.0
        @test coefficients(sub)[1, 1] == 0.0

        # mixing a space with a plain index is unsupported
        @test_throws ErrorException B[Taylor(1), 0]
        @test_throws ErrorException B[0, Taylor(1)]
        @test_throws ErrorException view(B, Taylor(1), 0)
        @test_throws ErrorException view(B, 0, Taylor(1))
    end

    @testset "component, eachcomponent, unpack: cartesian domain and codomain" begin
        dom = Taylor(1) × Taylor(0)
        codom = Taylor(0)^2
        A = LinearOperator(dom, codom, Float64[1 2 3 ; 4 5 6])

        c11 = component(A, 1, 1)
        @test domain(c11) == Taylor(1) && codomain(c11) == Taylor(0)
        @test coefficients(c11) == [1.0 2.0]

        c12 = component(A, 1, 2)
        @test domain(c12) == Taylor(0) && codomain(c12) == Taylor(0)
        @test coefficients(c12) == reshape([3.0], 1, 1)

        @test coefficients(component(A, 2, 1)) == [4.0 5.0]

        U = unpack(A)
        @test size(U) == (2, 2)
        @test U == collect(eachcomponent(A))
        @test all(coefficients(U[i,j]) == coefficients(component(A,i,j)) for i ∈ 1:2, j ∈ 1:2)

        # a component and an unpacked entry are views of the parent, whose own indices are
        # the plain positions of the cartesian space
        c11[0, 0] = 999.0
        @test A[1, 1] == 999.0
        U[2, 2][0, 0] = 555.0
        @test A[2, 3] == 555.0

        cr = component(A, 1:2, 1)
        @test domain(cr) == Taylor(1) && codomain(cr) == CartesianProduct(Taylor(0), Taylor(0))
        @test coefficients(component(A, :, 1)) == coefficients(cr)
    end

    @testset "component, eachcomponent, unpack: one-sided cartesian space" begin
        M = Float64[1 2 3 ; 4 5 6 ; 7 8 9]

        A = LinearOperator(Taylor(1) × Taylor(0), Taylor(2), M)
        a1, a2 = component(A, 1), component(A, 2)
        @test domain(a1) == Taylor(1) && codomain(a1) == Taylor(2)
        @test coefficients(a1) == M[:, 1:2]
        @test domain(a2) == Taylor(0) && codomain(a2) == Taylor(2)
        @test coefficients(a2) == M[:, 3:3]
        @test size(unpack(A)) == (1, 2)
        @test unpack(A) == collect(eachcomponent(A))

        B = LinearOperator(Taylor(2), Taylor(1) × Taylor(0), M)
        b1, b2 = component(B, 1), component(B, 2)
        @test domain(b1) == Taylor(2) && codomain(b1) == Taylor(1)
        @test coefficients(b1) == M[1:2, :]
        @test domain(b2) == Taylor(2) && codomain(b2) == Taylor(0)
        @test coefficients(b2) == M[3:3, :]
        @test size(unpack(B)) == (2, 1)
        @test unpack(B) == collect(eachcomponent(B))
    end

    @testset "copy, zero, one, similar, fill, fill!" begin
        𝒯₂, 𝒯₃ = Taylor(2), Taylor(3)
        A = LinearOperator(𝒯₂, 𝒯₃, Float64[1 2 3 ; 4 5 6 ; 7 8 9 ; 10 11 12])

        Ac = copy(A)
        @test Ac == A
        Ac[0, 0] = -1.0
        @test A[0, 0] == 1.0 # `copy` does not alias

        Az = zero(A)
        @test domain(Az) == 𝒯₂ && codomain(Az) == 𝒯₃
        @test iszero(Az)
        @test !iszero(A)

        # `one` is the identity on `domain(A) ∩ codomain(A)` and zero elsewhere
        @test coefficients(one(A)) == [1.0 0.0 0.0 ; 0.0 1.0 0.0 ; 0.0 0.0 1.0 ; 0.0 0.0 0.0]

        As = similar(A)
        @test domain(As) == domain(A) && codomain(As) == codomain(A)
        @test eltype(As) == Float64
        @test eltype(similar(A, Int)) == Int

        F = fill(7.0, Taylor(1), Taylor(0))
        @test domain(F) == Taylor(1) && codomain(F) == Taylor(0)
        @test coefficients(F) == [7.0 7.0]
        fill!(F, 3.0)
        @test coefficients(F) == [3.0 3.0]

        T = LinearOperator{Taylor,Taylor,Matrix{Float64}}
        z = zero(T)
        @test domain(z) == codomain(z) == Taylor(0) && coefficients(z) == [0.0;;]
        @test coefficients(one(T)) == [1.0;;]
        @test complex(T) == LinearOperator{Taylor,Taylor,Matrix{ComplexF64}}
    end

    @testset "float, complex, real, imag, conj" begin
        ℱ = Fourier(1, 1) # Int frequency
        A = LinearOperator(ℱ, ℱ, Float64[1 2 3 ; 4 5 6 ; 7 8 9])
        Af = float(A)
        @test eltype(Af) == Float64
        @test domain(Af) == codomain(Af) == Fourier(1, 1.0) # the frequency is promoted too
        @test coefficients(Af) == Float64[1 2 3 ; 4 5 6 ; 7 8 9]

        Ac = complex(A)
        @test eltype(Ac) == ComplexF64
        @test domain(Ac) == domain(A) # `complex` leaves the spaces untouched

        𝒯 = Taylor(1)
        B = LinearOperator(𝒯, 𝒯, ComplexF64[1+2im 3-1im ; 0+1im 2+0im])
        @test real(B) == LinearOperator(𝒯, 𝒯, Float64[1 3 ; 0 2])
        @test imag(B) == LinearOperator(𝒯, 𝒯, Float64[2 -1 ; 1 0])
        @test conj(B) == conj!(copy(B)) == LinearOperator(𝒯, 𝒯, ComplexF64[1-2im 3+1im ; 0-1im 2-0im])

        D = LinearOperator(𝒯, Taylor(0), [10.0 20.0])
        E = LinearOperator(𝒯, Taylor(0), [1.0 2.0])
        @test coefficients(complex(D, E)) == ComplexF64[10+1im 20+2im]
        # the parts are paired on the sums of the spaces, not position by position
        E₂ = LinearOperator(Taylor(2), Taylor(0), [1.0 2.0 3.0])
        @test coefficients(complex(D, E₂)) == ComplexF64[10+1im 20+2im 3im]
    end

    @testset "transpose, adjoint" begin
        At = transpose(LinearOperator(Taylor(1), Taylor(0), [1.0 2.0]))
        @test domain(At) == Taylor(0) && codomain(At) == Taylor(1)
        @test coefficients(At) == reshape([1.0, 2.0], 2, 1)

        B = LinearOperator(Taylor(1), Taylor(0), ComplexF64[1+2im 3-1im])
        @test coefficients(adjoint(B)) == reshape(ComplexF64[1-2im, 3+1im], 2, 1)

        at = transpose(Sequence(Taylor(1), [1.0, 2.0]))
        @test domain(at) == Taylor(1) && codomain(at) == ScalarSpace()
        @test coefficients(at) == [1.0 2.0]
    end

    @testset "equality, isapprox, iszero" begin
        op(x) = LinearOperator(Taylor(1), Taylor(1), [1.0 2.0 ; 3.0 x])
        A = op(4.0)
        @test A == op(4.0)
        @test A != op(4.1)
        @test isapprox(A, op(4.0))
        @test !isapprox(A, op(4.1))
        @test isapprox(A, op(4.0 + 1e-10))
        @test !isapprox(A, op(4.0 + 1e-10); atol = 1e-12)
        @test !iszero(A)
        @test iszero(zeros(Taylor(1), Taylor(1)))
    end

    @testset "interval and complex numeric-type conversions" begin
        A = LinearOperator(Taylor(1), Taylor(1), [1.0 2.0 ; 3.0 4.0])
        Ai = interval(A)
        @test domain(Ai) == domain(A) && codomain(Ai) == codomain(A)
        @test all(isequal_interval.(coefficients(Ai), interval.(coefficients(A))))

        # 1*10 + 2*20 = 50 and 3*10 + 4*20 = 110
        a = Sequence(Taylor(1), [10.0, 20.0])
        ci = Ai * interval(a)
        @test issubset_interval(interval(50.0), ci[0])
        @test issubset_interval(interval(110.0), ci[1])
        @test complex(A) * complex(a) == Sequence(Taylor(1), ComplexF64[50.0, 110.0])
    end

    @testset "getcoefficient with SymmetricSpace index coordinates" begin
        # evensym: a₋ₖ = aₖ, with representatives 0 and 1. A column of a representative sums its
        # orbit and a row averages its orbit.
        ℱ = Fourier(1, 1.0)
        es = evensym(ℱ)
        es2 = evensym(Fourier(2, 1.0))
        @test indices(es) == 0:1

        A = LinearOperator(ℱ, ℱ, Float64[1 2 3 ; 4 5 6 ; 7 8 9]) # A[k,l] = 3(k+1) + l+2
        B = LinearOperator(es, es, Float64[1 2 ; 3 4])
        P = LinearOperator(es, ℱ, Float64[1 2 ; 3 4 ; 5 6])
        Q = LinearOperator(ℱ, es, Float64[1 2 3 ; 4 5 6])
        cases = (
            # plain operator read in symmetric coordinates
            (A, (es, 1), (es, 1), (9 + 7 + 3 + 1) / 2),
            (A, (es, 0), (es, 0), 5.0),
            (A, (es, 1), (es, 0), (8 + 2) / 2),
            (A, (es, 0), (es, 1), 6.0 + 4.0),
            (A, (ℱ, 0), (es, 1), 6.0 + 4.0),
            (A, (ℱ, 0), (es, 0), 5.0),
            (A, (es, 1), (ℱ, 0), (8 + 2) / 2),
            (A, (es, 0), (ℱ, 0), 5.0),
            # the coefficients vanish outside the truncation of the operator
            (A, (es2, 2), (es2, 2), 0.0),
            (A, (es2, 1), (es2, 1), 10.0),
            # symmetric operator read in full coordinates: the column of -1 is the one of its
            # representative 1 divided by the size of the orbit
            (B, (ℱ, -1), (ℱ, -1), 4 / 2),
            (B, (ℱ, -1), (ℱ, 1), 4 / 2),
            (B, (ℱ, 0), (ℱ, 0), 1.0),
            (P, (ℱ, 0), (ℱ, -1), 4 / 2),
            (P, (ℱ, 0), (ℱ, 1), 4 / 2),
            (Q, (ℱ, 1), (ℱ, 0), 5.0),
            (Q, (ℱ, -1), (ℱ, 0), 5.0),
        )
        for (L, α, β, v) ∈ cases
            @test RadiiPolynomial.getcoefficient(L, α, β) == v
        end
    end

    @testset "lazy wrappers: domain/codomain adaptation" begin
        𝒯₁, 𝒯₂, 𝒯₃ = Taylor(1), Taylor(2), Taylor(3)
        A = LinearOperator(𝒯₂, 𝒯₃, zeros(4, 3))
        B = LinearOperator(𝒯₂, 𝒯₁, zeros(2, 3))
        undefined(L) = domain(L, UndefSpace()) == codomain(L, UndefSpace()) == UndefSpace()
        @test undefined(A)

        @testset "UniformScalingOperator" begin
            J = UniformScalingOperator(2.0)
            @test domain(J, 𝒯₂) == 𝒯₂ && codomain(J, 𝒯₂) == 𝒯₂
            @test undefined(J)
            @test eltype(J) == Float64
            @test zero(J) == UniformScalingOperator(0.0)
            @test one(J) == UniformScalingOperator(1.0)
            @test UniformScalingOperator(I) == UniformScalingOperator(true)
            @test isequal_interval(interval(J).λ, interval(2.0))
        end

        @testset "Negate" begin
            N = Negate(A)
            @test domain(N, Taylor(5)) == domain(A, Taylor(5)) == 𝒯₂
            @test codomain(N, 𝒯₂) == codomain(A, 𝒯₂) == 𝒯₃
            @test undefined(N)
        end

        @testset "Add" begin
            S = Add(A, B)
            @test domain(S, Taylor(5)) == 𝒯₂ # union of the domains
            @test codomain(S, 𝒯₂) == 𝒯₃ # union of the codomains 𝒯₃ and 𝒯₁
            @test undefined(S)
            # an undefined summand makes the whole sum undefined
            @test codomain(Add(A, Evaluation(0.5)), 𝒯₂) == 𝒯₃
            @test domain(Add(A, Evaluation(0.5)), 𝒯₃) == UndefSpace()
        end

        @testset "ComposedOperator" begin
            C = LinearOperator(𝒯₁, 𝒯₂, zeros(3, 2))
            Comp = ComposedOperator(A, C) # 𝒯₁ → 𝒯₂ → 𝒯₃
            @test Comp isa ComposedOperator
            @test domain(Comp, Taylor(5)) == 𝒯₁
            @test codomain(Comp, 𝒯₁) == 𝒯₃
            @test undefined(Comp)
        end
    end

    @testset "space+index tuple-coordinate getindex on LinearOperator" begin
        A = LinearOperator(Taylor(2), Taylor(2), Float64[1 2 3 ; 4 5 6 ; 7 8 9])
        @test A[(Taylor(2), 0), (Taylor(2), 0)] == 1.0
        @test A[(Taylor(2), 1), (Taylor(2), 2)] == 6.0

        # on a tensor product a plain index is itself a tuple of integers, which must not be
        # confused with a space+index tuple
        𝒮 = Taylor(1) ⊗ Chebyshev(1)
        B = LinearOperator(𝒮, 𝒮, Matrix{Float64}(reshape(1:16, 4, 4)))
        @test B[(0, 0), (1, 1)] == B[(𝒮, (0, 0)), (𝒮, (1, 1))] == 13.0
    end

    @testset "grids of LinearOperators (parameter families)" begin
        s_par = Chebyshev(2)
        dom, codom = Taylor(1), Fourier(1, 1.0)
        A = _lo_family(s_par, dom, codom)

        @testset "Chebyshev parameter: to_grid/to_coef round trip" begin
            A_grid = to_grid(A, grid_size(s_par))
            @test A_grid isa Vector
            @test size(A_grid) == grid_size(s_par) == (3,)
            @test all(X -> (domain(X) == dom) & (codomain(X) == codom), A_grid)
            _lo_test_roundtrip(A, s_par, A_grid)
            _lo_test_nodes(A, A_grid, Sequence(dom, [1.0, -2.0]))
        end

        @testset "two parameter factors give a Matrix grid" begin
            s₂ = Chebyshev(1) ⊗ Chebyshev(2)
            A₂ = _lo_family(s₂, Taylor(1), Taylor(1))
            A_grid = to_grid(A₂, grid_size(s₂))
            @test A_grid isa Matrix
            @test size(A_grid) == grid_size(s₂) == (2, 3)
            _lo_test_roundtrip(A₂, s₂, A_grid)
        end

        @testset "interval enclosure round trip" begin
            s₁ = Chebyshev(1)
            Ai = interval(_lo_family(s₁, Taylor(1), Chebyshev(1)))
            _lo_test_enclosure(Ai, to_coef(to_grid(Ai, grid_size(s₁)), s₁))
        end

        @testset "resampling an operator family on a finer grid" begin
            A_fine = to_grid(A, (9,))
            @test size(A_fine) == (9,)
            _lo_test_roundtrip(A, s_par, A_fine)
        end

        @testset "symmetric codomain: reduced rows round trip" begin
            codom_sym = evensym(Fourier(2, 1.0))
            As = project(_lo_family(s_par, dom, Fourier(2, 1.0)), domain(A), s_par ⊗ codom_sym)
            A_grid = to_grid(As, grid_size(s_par))
            @test all(X -> (domain(X) == dom) & (codomain(X) == codom_sym), A_grid)
            _lo_test_roundtrip(As, s_par, A_grid)
        end

        @testset "every codomain factor discretized: each node holds a functional" begin
            Af = LinearOperator(domain(A), s_par, reshape(collect(1.0:6.0), :, 2))
            A_grid = to_grid(Af, grid_size(s_par))
            @test size(A_grid) == (3,)
            @test all(Z -> (domain(Z) == dom) & (codomain(Z) == ScalarSpace()), A_grid)

            v = Sequence(dom, [1.0, -2.0])
            Av = Af * Sequence(domain(A), coefficients(v)) # the scalar family t -> A(t)v
            for k ∈ 1:3
                @test real(only(coefficients(A_grid[k] * v))) ≈ Evaluation(cospi((k-1)/2)) * Av atol=1e-9
            end
        end

        @testset "empty inner domain: reduces to the sequence case" begin
            # a family of columns is a family of sequences, without a `ScalarSpace` wrapper
            X₀ = RadiiPolynomial._zero_space(s_par)
            for (s, T) ∈ ((s_par ⊗ codom, Sequence), (s_par, ComplexF64))
                coeffs = reshape(collect(1.0:dimension(s)), :, 1)
                A_grid = to_grid(LinearOperator(X₀, s, coeffs), grid_size(s_par))
                @test eltype(A_grid) <: T
                @test A_grid == to_grid(Sequence(s, vec(coeffs)), grid_size(s_par))
                T == Sequence && @test all(x -> space(x) == codom, A_grid)
            end
        end

        @testset "error paths" begin
            # the codomain must start with the parameter factor
            @test_throws ArgumentError to_grid(LinearOperator(Taylor(1), Fourier(1, 1.0), ones(3, 2)), (2,))
            # the leading factors of the domain must be the zero space of the parameter
            @test_throws ArgumentError to_grid(LinearOperator(Chebyshev(1) ⊗ Taylor(1), Chebyshev(2) ⊗ Fourier(1, 1.0), ones(9, 4)), (3,))
            # the grid elements must share their domain and codomain
            X₁ = LinearOperator(Taylor(1), Taylor(1), ones(2, 2))
            X₂ = LinearOperator(Taylor(2), Taylor(1), ones(2, 3))
            @test_throws ArgumentError to_coef([X₁, X₂], Chebyshev(1))
            # the grid has one dimension per factor of the parameter space
            @test_throws ArgumentError to_coef(fill(X₁, 2, 2), Chebyshev(1))
        end
    end

    @testset "grids of LinearOperators on a CartesianSpace" begin
        s_par = Chebyshev(2)
        X₀ = RadiiPolynomial._zero_space(s_par)
        inner_dom, inner_codom = Taylor(1) × Fourier(1, 1.0), Fourier(1, 1.0) × Taylor(1)
        dom, codom = (X₀ ⊗ Taylor(1)) × (X₀ ⊗ Fourier(1, 1.0)), (s_par ⊗ Fourier(1, 1.0)) × (s_par ⊗ Taylor(1))
        family(dom, codom) = LinearOperator(dom, codom, reshape(collect(1.0:dimension(codom)*dimension(dom)), :, dimension(dom)))
        A = family(dom, codom)
        A_grid = to_grid(A, grid_size(s_par))

        @testset "both sides cartesian: the grid holds the blocks at each node" begin
            @test size(A_grid) == grid_size(s_par) == (3,)
            @test all(X -> (domain(X) == inner_dom) & (codomain(X) == inner_codom), A_grid)
            # each block is sampled as it would be on its own
            for i ∈ 1:2, j ∈ 1:2
                ref = to_grid(component(A, i, j), grid_size(s_par))
                @test all(k -> coefficients(component(A_grid[k], i, j)) == coefficients(ref[k]), eachindex(A_grid))
            end
            _lo_test_roundtrip(A, s_par, A_grid)
            _lo_test_nodes(A, A_grid, Sequence(inner_dom, [1.0, -2.0, 0.5, 1.5, -1.0]))
        end

        @testset "one side cartesian, or every codomain factor discretized" begin
            cases = (
                (family(X₀ ⊗ Taylor(1), codom), Taylor(1), inner_codom),
                (family(dom, s_par ⊗ Fourier(1, 1.0)), inner_dom, Fourier(1, 1.0)),
                (family(dom, s_par^2), inner_dom, ScalarSpace()^2),
            )
            for (B, d, c) ∈ cases
                G = to_grid(B, grid_size(s_par))
                @test all(X -> (domain(X) == d) & (codomain(X) == c), G)
                _lo_test_roundtrip(B, s_par, G)
            end
        end

        @testset "in-place entry points" begin
            G = to_grid(A, grid_size(s_par))
            foreach(X -> fill!(coefficients(X), complex(Inf, Inf)), G)
            @test to_grid!(G, A) === G
            @test all(k -> coefficients(G[k]) == coefficients(A_grid[k]), eachindex(G))

            B = zeros(ComplexF64, dom, codom)
            @test to_coef!(B, A_grid) === B
            @test real.(coefficients(B)) ≈ coefficients(A) atol=1e-9
        end

        @testset "interval enclosure round trip" begin
            Ai = interval(A)
            _lo_test_enclosure(Ai, to_coef(to_grid(Ai, grid_size(s_par)), s_par))
        end

        @testset "error paths" begin
            # the components must be sampled on the same nodes
            codom_bad = (s_par ⊗ Fourier(1, 1.0)) × (Fourier(1, 1.0) ⊗ Taylor(1))
            @test_throws MethodError to_grid(LinearOperator(dom, codom_bad, ones(dimension(codom_bad), dimension(dom))), (3,))
            # the leading factors of the domain must be the zero space of the parameter
            dom_bad = (s_par ⊗ Taylor(1)) × (X₀ ⊗ Fourier(1, 1.0))
            @test_throws ArgumentError to_grid(LinearOperator(dom_bad, codom, ones(dimension(codom), dimension(dom_bad))), (3,))
            # the grid elements must match the domain and codomain of the destination
            @test_throws ArgumentError to_coef!(zeros(ComplexF64, dom, s_par ⊗ Fourier(1, 1.0)), A_grid)
        end
    end

end
