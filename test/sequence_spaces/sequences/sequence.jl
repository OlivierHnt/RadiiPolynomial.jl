struct _OffsetVector{T} <: AbstractVector{T}
    v :: Vector{T}
end
Base.size(a::_OffsetVector) = size(a.v)
Base.axes(a::_OffsetVector) = (2:length(a.v)+1,)
Base.IndexStyle(::Type{<:_OffsetVector}) = IndexLinear()
Base.getindex(a::_OffsetVector, i::Int) = a.v[i-1]

@testset "Sequence" begin

    𝒯 = Taylor(1)
    ℱ = Fourier(1, 1.0)
    𝒞 = Chebyshev(2)

    @testset "AbstractSequence utilities" begin
        𝑇 = 𝒯 ⊗ ℱ ⊗ 𝒞
        𝕊 = 𝑇^1
        coeffs = [inv(prod((2, 2, 2) .^ abs.(α))) for α ∈ indices(𝑇)]
        a = Sequence(𝕊, coeffs)
        @test space(a) == 𝕊
        @test coefficients(a) == coeffs
        @test order(a) == order(𝕊)
        @test order(a, 1) == order(𝕊, 1)
        @test_throws MethodError frequency(a) # 𝕊 mixes Taylor/Chebyshev with Fourier
        @test frequency(component(a, 1), 2) == frequency(𝑇, 2)
        @test firstindex(a) == RadiiPolynomial._firstindex(𝕊) == 1
        @test lastindex(a) == RadiiPolynomial._lastindex(𝕊) == length(coeffs)
        @test length(a) == length(coeffs)
        @test size(a) == size(coeffs)
        @test iterate(a) == iterate(coeffs)
        @test iterate(a, 2) == iterate(coeffs, 2)
        @test eltype(a) == eltype(typeof(a)) == eltype(coeffs)
    end

    @testset "construction" begin
        a = Sequence(𝒯, [1, 2])
        @test space(a) == 𝒯
        @test coefficients(a) == [1, 2]

        # dimension mismatch: Taylor(1) has dimension 2
        @test_throws DimensionMismatch Sequence(𝒯, [1])
        @test_throws DimensionMismatch Sequence(𝒯, [1, 2, 3])

        # offset vectors (eachindex != Base.OneTo) are rejected
        @test_throws ArgumentError Sequence(𝒯, _OffsetVector([1, 2]))

        # scalar / plain vector convenience constructors
        @test Sequence(3) == Sequence(ScalarSpace(), [3])
        @test Sequence([1, 2, 3]) == Sequence(ScalarSpace()^3, [1, 2, 3])
    end

    @testset "getindex, view, setindex!" begin
        # Taylor: indices 0:order
        a = Sequence(Taylor(2), [1, 2, 3])
        @test a[0] == 1 && a[1] == 2 && a[2] == 3
        @test_throws BoundsError a[3]
        @test_throws BoundsError a[0:3]
        @test a[0:1] == [1, 2]
        @test view(a, 0:1) == [1, 2]
        b = copy(a)
        b[0] = 10
        @test coefficients(b) == [10, 2, 3]
        b[[0, 2]] = [100, 300]
        @test coefficients(b) == [100, 2, 300]

        # Fourier: negative indices, indices -order:order
        f = Sequence(Fourier(2, 1.0), [1, 2, 3, 4, 5])
        @test f[-2] == 1 && f[-1] == 2 && f[0] == 3 && f[1] == 4 && f[2] == 5
        @test_throws BoundsError f[-3]
        @test f[-2:0] == [1, 2, 3]
        g = copy(f)
        g[-1] = 20
        @test coefficients(g) == [1, 20, 3, 4, 5]

        # Chebyshev: indices 0:order, same index convention as Taylor
        c = Sequence(Chebyshev(2), [1, 2, 3])
        @test c[0] == 1 && c[1] == 2 && c[2] == 3

        # TensorSpace: multi-index tuples.
        # For Taylor(1) ⊗ Fourier(1, 1.0), coefficient position(i, j) = (i + 1) + 2*(j + 1)
        # (Taylor varies fastest), so coefficients 1:6 sit at
        # (0,-1)->1 (1,-1)->2 (0,0)->3 (1,0)->4 (0,1)->5 (1,1)->6.
        t = Sequence(𝒯 ⊗ ℱ, collect(1:6))
        @test t[(0, -1)] == 1 && t[(1, -1)] == 2
        @test t[(0, 0)] == 3 && t[(1, 0)] == 4
        @test t[(0, 1)] == 5 && t[(1, 1)] == 6
        @test_throws BoundsError t[(0, -2)]

        # colon / range forms on tensor indices
        @test t[(:, 0)] == [3, 4]
        @test t[(0, :)] == [1, 3, 5]
        @test t[TensorIndices((0:1, -1:0))] == [1, 2, 3, 4]

        t2 = copy(t)
        t2[(:, 0)] = [999, 888]
        @test coefficients(t2) == [1, 2, 999, 888, 5, 6]
        t3 = copy(t)
        t3[TensorIndices((0:1, -1:0))] = [11, 22, 33, 44]
        @test coefficients(t3) == [11, 22, 33, 44, 5, 6]

        # a[:] returns a copy of the whole coefficient vector
        @test a[:] == coefficients(a)
        @test a[:] !== coefficients(a)

        # view(a, α) aliases the parent, getindex(a, α) does not.
        # view returns a plain array view with its own 1-based indexing,
        # so v[2] corresponds to Taylor index 1, i.e. coefficients(a)[2].
        v = view(a, 0:1)
        v[2] = -7
        @test coefficients(a)[2] == -7

        # getindex/view with a VectorSpace subspace
        a2 = Sequence(Taylor(3), [1.0, 2.0, 3.0, 4.0])
        p = a2[Taylor(1)]
        @test p == Sequence(Taylor(1), [1.0, 2.0])
        p[0] = 999.0
        @test coefficients(a2) == [1.0, 2.0, 3.0, 4.0] # getindex copies, does not alias
        vv = view(a2, Taylor(1))
        vv[0] = 999.0
        @test coefficients(a2) == [999.0, 2.0, 3.0, 4.0] # view aliases
        @test_throws BoundsError a2[Taylor(5)]
    end

    @testset "coefficients" begin
        coeffs = [1, 2, 3]
        a = Sequence(Taylor(2), coeffs)
        @test coefficients(a) === coeffs
    end

    @testset "copy, similar, zero, one, fill, iszero" begin
        a = Sequence(Taylor(2), [1, 2, 3])
        @test copy(a) == a
        @test copy(a) !== a
        @test coefficients(copy(a)) !== coefficients(a)

        s = similar(a)
        @test space(s) == space(a) && eltype(s) == eltype(a) && length(s) == length(a)
        s2 = similar(a, Float64)
        @test eltype(s2) == Float64

        @test iszero(a) == false
        @test iszero(zeros(Taylor(2)))
        @test zero(a) == zeros(Taylor(2))
        @test zeros(Float64, Taylor(2)) == Sequence(Taylor(2), zeros(3))
        @test ones(ComplexF64, Taylor(2)) == Sequence(Taylor(2), ones(ComplexF64, 3))
        @test fill(3, Taylor(2)) == Sequence(Taylor(2), [3, 3, 3])
        z = zeros(Taylor(2))
        fill!(z, 9.0)
        @test coefficients(z) == [9.0, 9.0, 9.0]

        # zero/one from the *type*: falls back to the trivial (order 0) space
        @test zero(Sequence{Taylor,Vector{Float64}}) == Sequence(Taylor(0), [0.0])
        @test one(Sequence{Taylor,Vector{Float64}}) == Sequence(Taylor(0), [1.0])
        @test zero(Sequence{Fourier{Float64},Vector{Float64}}) == Sequence(Fourier(0, 1.0), [0.0])
        @test zero(Sequence{TensorSpace{Tuple{Taylor,Chebyshev}},Vector{Float64}}) ==
            Sequence(Taylor(0) ⊗ Chebyshev(0), [0.0])

        # one(a) for a SequenceSpace: zero sequence except the constant (index 0) mode is 1
        b = Sequence(Taylor(2), [5.0, 6.0, 7.0])
        @test one(b) == Sequence(Taylor(2), [1.0, 0.0, 0.0])
        c = Sequence(Fourier(2, 1.0), [1.0, 2.0, 3.0, 4.0, 5.0])
        @test one(c) == Sequence(Fourier(2, 1.0), [0.0, 0.0, 1.0, 0.0, 0.0])
        d = Sequence(𝒯 ⊗ ℱ, collect(1.0:6.0)) # constant mode is (0,0) -> position 3
        @test one(d) == Sequence(𝒯 ⊗ ℱ, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        @test one(Sequence(3)) == Sequence(1)
    end

    @testset "float, complex, real, imag, conj, conj!" begin
        coeffs = [1, 2, 3]
        a = Sequence(Taylor(2), coeffs)
        @test float(a) == Sequence(Taylor(2), float(coeffs))
        @test complex(a) == Sequence(Taylor(2), complex(coeffs))
        @test real(a) == Sequence(Taylor(2), real(coeffs))
        @test imag(a) == Sequence(Taylor(2), imag(coeffs))
        @test conj(a) == Sequence(Taylor(2), conj(coeffs))
        @test conj!(copy(a)) == Sequence(Taylor(2), conj!(copy(coeffs)))

        z = Sequence(Taylor(1), [1 + 2im, 3 - 1im])
        @test real(z) == Sequence(Taylor(1), [1, 3])
        @test imag(z) == Sequence(Taylor(1), [2, -1])
        @test conj(z) == Sequence(Taylor(1), [1 - 2im, 3 + 1im])

        # complex(a, b) combines two real sequences into a complex one
        re = Sequence(Taylor(1), [1, 3])
        im_ = Sequence(Taylor(1), [1.0, 2.0])
        @test complex(re, im_) == Sequence(Taylor(1), [1 + 1im, 3 + 2im])
        @test complex(typeof(im_)) == Sequence{Taylor,Vector{ComplexF64}}
    end

    @testset "component, eachcomponent, unpack" begin
        𝕊 = Taylor(1)^2 # CartesianPower
        b = Sequence(𝕊, [1.0, 2.0, 3.0, 4.0])

        c1 = component(b, 1)
        @test c1 == Sequence(Taylor(1), [1.0, 2.0])
        c1[0] = 100.0
        @test coefficients(b) == [100.0, 2.0, 3.0, 4.0] # component aliases the parent

        r = component(b, 1:2)
        @test space(r) == Taylor(1)^2
        @test coefficients(r) == coefficients(b)

        @test unpack(b) == [component(b, i) for i ∈ 1:nspaces(𝕊)]
        u = unpack(b)
        u[2][0] = -5.0
        @test coefficients(b) == [100.0, 2.0, -5.0, 4.0] # unpack still aliases

        @test collect(eachcomponent(b)) == unpack(b)

        # CartesianProduct of mixed spaces
        𝕋 = 𝒯 × ℱ
        d = Sequence(𝕋, [1.0, 2.0, 3.0, 4.0, 5.0])
        @test component(d, 1) == Sequence(𝒯, [1.0, 2.0])
        @test component(d, 2) == Sequence(ℱ, [3.0, 4.0, 5.0])
        cc = component(d, 1)
        cc[0] = 999.0
        @test coefficients(d)[1] == 999.0
    end

    @testset "conjugacy_symmetry!" begin
        # ScalarSpace: keep only the real part
        a = Sequence(ScalarSpace(), [1.0 + 2.0im])
        @test conjugacy_symmetry!(copy(a)) == Sequence(ScalarSpace(), [1.0 + 0.0im])

        # Fourier: A .= (A .+ conj.(reverse(A))) ./ 2
        f = Fourier(2, 1.0)
        b = Sequence(f, ComplexF64[1+1im, 2+2im, 3+0im, 4-1im, 5+3im])
        # reverse(A) = [5+3im, 4-1im, 3+0im, 2+2im, 1+1im]; conj(reverse(A)) = [5-3im, 4+1im, 3, 2-2im, 1-1im]
        # (A + conj(reverse(A)))/2 = [3-1im, 3+1.5im, 3+0im, 3-1.5im, 3+1im]
        expected_b = ComplexF64[3-1im, 3+1.5im, 3+0im, 3-1.5im, 3+1im]
        @test coefficients(conjugacy_symmetry!(copy(b))) == expected_b

        # TensorSpace of two Fourier factors: same rule on the reshaped array.
        # Values 1:25 are real, so entry k pairs with entry 26-k giving (k+(26-k))/2 = 13 everywhere.
        ff = f ⊗ f
        cplx = Sequence(ff, ComplexF64.(collect(1:25)))
        @test all(==(13.0 + 0.0im), coefficients(conjugacy_symmetry!(copy(cplx))))

        # CartesianSpace: applied component-wise.
        # Each block 1:5 (resp. 6:10) is real and palindromic-summing to 6 (resp. 16),
        # so every entry becomes 6/2=3 (resp. 16/2=8).
        dpow = Sequence(f^2, ComplexF64.(vcat(1:5, 6:10)))
        @test coefficients(conjugacy_symmetry!(copy(dpow))) == ComplexF64[3, 3, 3, 3, 3, 8, 8, 8, 8, 8]

        # CartesianProduct with a single factor
        dprod = Sequence(CartesianProduct((f,)), ComplexF64.(1:5))
        @test coefficients(conjugacy_symmetry!(copy(dprod))) == ComplexF64[3, 3, 3, 3, 3]

        # error path: `_conjugacy_symmetry!(::Sequence)` is only implemented
        # for ScalarSpace / Fourier / TensorSpace{<:Fourier...} / CartesianSpace
        @test_throws DomainError conjugacy_symmetry!(Sequence(Taylor(2), [1.0, 2.0, 3.0]))
        @test_throws DomainError conjugacy_symmetry!(Sequence(Chebyshev(2), [1.0, 2.0, 3.0]))
        @test_throws DomainError conjugacy_symmetry!(Sequence(Taylor(1) ⊗ Chebyshev(1), [1.0, 2.0, 3.0, 4.0]))
    end

    @testset "geometricweight, algebraicweight" begin
        # geometric decay 2^{-i} -> rate 2
        a = Sequence(Taylor(10), [inv(2.0^i) for i ∈ 0:10])
        @test rate(geometricweight(a)) ≈ 2 rtol=1e-8

        # tensor: independent geometric rates 2 and 3 along each factor
        b = Sequence(Taylor(10) ⊗ Fourier(3, 1.0), vec([inv(2.0^i * 3.0^abs(j)) for i ∈ 0:10, j ∈ -3:3]))
        rates_b = rate.(geometricweight(b))
        @test rates_b[1] ≈ 2 rtol=1e-8
        @test rates_b[2] ≈ 3 rtol=1e-8

        # algebraic decay (1+i)^{-2} -> rate 2
        c = Sequence(Taylor(10), [inv((1.0 + i)^2) for i ∈ 0:10])
        @test rate(algebraicweight(c)) ≈ 2 rtol=1e-8

        d = Sequence(Taylor(10) ⊗ Fourier(3, 1.0), vec([inv((1.0 + i)^2 * (1.0 + abs(j))^3) for i ∈ 0:10, j ∈ -3:3]))
        rates_d = rate.(algebraicweight(d))
        @test rates_d[1] ≈ 2 rtol=1e-8
        @test rates_d[2] ≈ 3 rtol=1e-8
    end

    @testset "polish!" begin
        # Clean geometric decay 2^{-i}, i = 0..4, plus a wild outlier at the last index (i=5).
        # The regression correctly identifies rate ≈ 2 from indices 0..4 (ord = 4) and,
        # since |a[5]| = 1000 vastly exceeds norm(a,1)/2^5 ≈ 31.3, that single entry is zeroed.
        a = Sequence(Taylor(5), [1.0, 0.5, 0.25, 0.125, 0.0625, 1000.0])
        pa = polish!(copy(a))
        @test coefficients(pa) == [1.0, 0.5, 0.25, 0.125, 0.0625, 0.0]

        # Same decay without any outlier: nothing exceeds the fitted model, so polish! is a no-op.
        b = Sequence(Taylor(5), [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])
        @test polish!(copy(b)) == b

        # CartesianSpace: polish! acts component-wise.
        𝕊 = Taylor(5)^2
        c = Sequence(𝕊, vcat([1.0, 0.5, 0.25, 0.125, 0.0625, 1000.0], [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]))
        pc = polish!(copy(c))
        @test coefficients(pc) == vcat([1.0, 0.5, 0.25, 0.125, 0.0625, 0.0], [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125])

        # no-ops by definition
        s = Sequence(3.0)
        @test polish!(s) === s
        t = Sequence(𝒯 ⊗ ℱ, collect(1.0:6.0))
        @test polish!(t) === t
    end

    @testset "symmetric spaces (getcoefficient)" begin
        # evensym(Taylor(4)): only even-order monomials survive, indices [0, 2, 4]
        se = evensym(Taylor(4))
        @test collect(indices(se)) == [0, 2, 4]
        a = Sequence(se, [1.0, 2.0, 3.0])
        @test a[0] == 1.0 && a[2] == 2.0 && a[4] == 3.0
        @test RadiiPolynomial.getcoefficient(a, (se, 0)) == 1.0
        # desymmetrized (plain Taylor(4)) view: odd indices are forced to zero
        @test RadiiPolynomial.getcoefficient(a, (Taylor(4), 0)) == 1.0 + 0.0im
        @test RadiiPolynomial.getcoefficient(a, (Taylor(4), 1)) == 0.0 + 0.0im
        @test RadiiPolynomial.getcoefficient(a, (Taylor(4), 2)) == 2.0 + 0.0im

        # oddsym(Fourier(3, 1.0)): only positive indices are representatives (index 0 is
        # excluded since an odd function vanishes there), indices [1, 2, 3]
        so = oddsym(Fourier(3, 1.0))
        @test collect(indices(so)) == [1, 2, 3]
        b = Sequence(so, [10.0, 20.0, 30.0])
        fo = Fourier(3, 1.0)
        @test RadiiPolynomial.getcoefficient(b, (fo, 1)) == 10.0 + 0.0im
        @test RadiiPolynomial.getcoefficient(b, (fo, -1)) == -10.0 + 0.0im # odd: a[-k] = -a[k]
        @test RadiiPolynomial.getcoefficient(b, (fo, 0)) == 0.0 + 0.0im    # odd function vanishes at 0

        # d4sym(Fourier ⊗ Fourier): the square symmetry identifies (i,j) with (j,i)
        sd = d4sym(Fourier(2, 1.0) ⊗ Fourier(2, 1.0))
        @test collect(indices(sd)) == [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 2)]
        c = Sequence(sd, collect(1.0:6.0))
        fd = Fourier(2, 1.0) ⊗ Fourier(2, 1.0)
        @test RadiiPolynomial.getcoefficient(c, (fd, (1, 0))) == RadiiPolynomial.getcoefficient(c, (fd, (0, 1))) == 2.0 + 0.0im

        # symmetry mismatch is a guarded error
        se2 = oddsym(Taylor(4))
        @test_throws DomainError RadiiPolynomial.getcoefficient(a, (se2, 0))
    end

    @testset "interval and complex coefficients" begin
        a = Sequence(Taylor(2), interval.([1, 2, 3]))
        @test isequal_interval(a[0], interval(1))
        @test in_interval(2, a[1])
        @test isequal_interval.(coefficients(zero(a)), interval(0)) |> all
        @test a == Sequence(Taylor(2), interval.([1, 2, 3])) # thin intervals: Sequence == is safe

        parent = Sequence((Taylor(2))^1, interval.([1, 2, 3]))
        b = component(parent, 1) # component still aliases with interval coefficients
        @test isequal_interval(b[0], interval(1))
        b[0] = interval(9)
        @test isequal_interval(component(parent, 1)[0], interval(9))

        z = Sequence(Taylor(1), Complex{Interval{Float64}}.(interval.([1, 2])))
        @test isequal_interval(real(z)[0], interval(1))
        @test isequal_interval(imag(z)[0], interval(0))
        @test isequal_interval(real(conj(z)[0]), interval(1))

        f = Fourier(1, 1.0)
        w = Sequence(f, Complex{Interval{Float64}}.(interval.([1, 2, 3])))
        ws = conjugacy_symmetry!(copy(w))
        # (1+3)/2 = 2 at both endpoints, 2 unchanged at the centre
        @test isequal_interval(real(ws[-1]), interval(2))
        @test isequal_interval(real(ws[0]), interval(2))
        @test isequal_interval(real(ws[1]), interval(2))
    end

    @testset "utilities: reverse, permutedims, selectdim" begin
        a = Sequence(Taylor(2), [1, 2, 3])
        @test reverse(a) == Sequence(Taylor(2), [3, 2, 1])
        @test reverse!(copy(a)) == Sequence(Taylor(2), [3, 2, 1])

        # permuting the two tensor factors swaps the index tuples
        d = Sequence(𝒯 ⊗ ℱ, collect(1:6))
        dp = permutedims(d, [2, 1])
        @test space(dp) == ℱ ⊗ 𝒯
        @test all(dp[(j, i)] == d[(i, j)] for i ∈ indices(𝒯), j ∈ indices(ℱ))

        a_𝑇 = Sequence(𝒯 ⊗ ℱ ⊗ Chebyshev(2), collect(1:18))
        @test selectdim(a_𝑇, 2, 0) == selectdim(reshape(collect(1:18), 2, 3, 3), 2, 2)
    end

    @testset "broadcast dotview" begin
        a = Sequence(Taylor(2), [1.0, 2.0, 3.0])
        a[0:1] .= (10.0, 20.0)
        @test coefficients(a) == [10.0, 20.0, 3.0]
    end

end
