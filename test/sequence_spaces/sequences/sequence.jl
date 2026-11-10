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
        @test coefficients(a) === coeffs
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

        @test_throws DimensionMismatch Sequence(𝒯, [1])
        @test_throws DimensionMismatch Sequence(𝒯, [1, 2, 3])
        # the coefficients must be indexed from 1
        @test_throws ArgumentError Sequence(𝒯, _OffsetVector([1, 2]))

        @test Sequence(3) == Sequence(ScalarSpace(), [3])
        @test Sequence([1, 2, 3]) == Sequence(ScalarSpace()^3, [1, 2, 3])
    end

    @testset "getindex, view, setindex!" begin
        a = Sequence(Taylor(2), [1, 2, 3])
        @test [a[k] for k ∈ 0:2] == a[0:2] == a[:] == [1, 2, 3]
        @test a[:] !== coefficients(a)
        @test a[0:1] == view(a, 0:1) == [1, 2]
        @test_throws BoundsError a[3]
        @test_throws BoundsError a[0:3]
        b = copy(a)
        b[0] = 10
        @test coefficients(b) == [10, 2, 3]
        b[[0, 2]] = [100, 300]
        @test coefficients(b) == [100, 2, 300]
        b[0:1] .= (10.0, 20.0)
        @test coefficients(b) == [10, 20, 300]

        f = Sequence(Fourier(2, 1.0), [1, 2, 3, 4, 5])
        @test [f[k] for k ∈ -2:2] == [1, 2, 3, 4, 5]
        @test f[-2:0] == [1, 2, 3]
        @test_throws BoundsError f[-3]
        f[-1] = 20
        @test coefficients(f) == [1, 20, 3, 4, 5]

        @test [Sequence(𝒞, [1, 2, 3])[k] for k ∈ 0:2] == [1, 2, 3]

        # the first index of a tensor space varies fastest
        t = Sequence(𝒯 ⊗ ℱ, collect(1:6))
        @test [t[(i, j)] for i ∈ 0:1, j ∈ -1:1] == [1 3 5 ; 2 4 6]
        @test_throws BoundsError t[(0, -2)]
        @test t[(:, 0)] == [3, 4]
        @test t[(0, :)] == [1, 3, 5]
        @test t[TensorIndices((0:1, -1:0))] == [1, 2, 3, 4]
        t[(:, 0)] = [999, 888]
        @test coefficients(t) == [1, 2, 999, 888, 5, 6]
        t[TensorIndices((0:1, -1:0))] = [11, 22, 33, 44]
        @test coefficients(t) == [11, 22, 33, 44, 5, 6]

        # a view on a range aliases the parent and is indexed from 1
        v = view(a, 0:1)
        v[2] = -7
        @test coefficients(a) == [1, -7, 3]

        # indexing by a space copies, a view on a space aliases
        a2 = Sequence(Taylor(3), [1.0, 2.0, 3.0, 4.0])
        p = a2[Taylor(1)]
        @test p == Sequence(Taylor(1), [1.0, 2.0])
        p[0] = 999.0
        @test coefficients(a2) == [1.0, 2.0, 3.0, 4.0]
        view(a2, Taylor(1))[0] = 999.0
        @test coefficients(a2) == [999.0, 2.0, 3.0, 4.0]
        @test_throws BoundsError a2[Taylor(5)]
    end

    @testset "copy, similar, zero, one, fill, iszero" begin
        a = Sequence(Taylor(2), [1, 2, 3])
        @test copy(a) == a
        @test copy(a) !== a
        @test coefficients(copy(a)) !== coefficients(a)

        s = similar(a)
        @test space(s) == space(a) && eltype(s) == eltype(a) && length(s) == length(a)
        @test eltype(similar(a, Float64)) == Float64

        @test !iszero(a)
        @test iszero(zeros(Taylor(2)))
        @test zero(a) == zeros(Taylor(2))
        @test zeros(Float64, Taylor(2)) == Sequence(Taylor(2), zeros(3))
        @test ones(ComplexF64, Taylor(2)) == Sequence(Taylor(2), ones(ComplexF64, 3))
        @test fill(3, Taylor(2)) == Sequence(Taylor(2), [3, 3, 3])
        @test coefficients(fill!(zeros(Taylor(2)), 9.0)) == [9.0, 9.0, 9.0]

        # from the type alone, `zero` and `one` live on the space of order 0
        @test zero(Sequence{Taylor,Vector{Float64}}) == Sequence(Taylor(0), [0.0])
        @test one(Sequence{Taylor,Vector{Float64}}) == Sequence(Taylor(0), [1.0])
        @test zero(Sequence{Fourier{Float64},Vector{Float64}}) == Sequence(Fourier(0, 1.0), [0.0])
        @test zero(Sequence{TensorSpace{Tuple{Taylor,Chebyshev}},Vector{Float64}}) ==
            Sequence(Taylor(0) ⊗ Chebyshev(0), [0.0])

        # `one` of a sequence is the constant mode on the same space
        @test one(Sequence(Taylor(2), [5.0, 6.0, 7.0])) == Sequence(Taylor(2), [1.0, 0.0, 0.0])
        c = Sequence(Fourier(2, 1.0), [1.0, 2.0, 3.0, 4.0, 5.0])
        @test one(c) == Sequence(Fourier(2, 1.0), [0.0, 0.0, 1.0, 0.0, 0.0])
        @test one(Sequence(𝒯 ⊗ ℱ, collect(1.0:6.0))) == Sequence(𝒯 ⊗ ℱ, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        @test one(Sequence(3)) == Sequence(1)
        # the constant keeps the symmetries fixing it: all of them on an even space, none on an odd one
        @test one(Sequence(evensym(Fourier(2, 1.0)), ones(3))) == Sequence(evensym(Fourier(2, 1.0)), [1.0, 0.0, 0.0])
        o = one(Sequence(oddsym(Fourier(2, 1.0)), ones(2)))
        @test project(o, Fourier(2, 1.0)) == one(c) && length(elements(symmetry(space(o)))) == 1
    end

    @testset "float, complex, real, imag, conj, conj!" begin
        coeffs = [1, 2, 3]
        a = Sequence(Taylor(2), coeffs)
        for f ∈ (float, complex, real, imag, conj)
            @test f(a) == Sequence(Taylor(2), f(coeffs))
        end
        @test conj!(copy(a)) == a

        z = Sequence(𝒯, [1 + 2im, 3 - 1im])
        @test real(z) == Sequence(𝒯, [1, 3])
        @test imag(z) == Sequence(𝒯, [2, -1])
        @test conj(z) == conj!(copy(z)) == Sequence(𝒯, [1 - 2im, 3 + 1im])

        re = Sequence(𝒯, [1, 3])
        im_ = Sequence(𝒯, [1.0, 2.0])
        @test complex(re, im_) == Sequence(𝒯, [1 + 1im, 3 + 2im])
        @test complex(typeof(im_)) == Sequence{Taylor,Vector{ComplexF64}}
        # the parts are paired on the sum of their spaces, not position by position
        @test complex(re, Sequence(Taylor(2), [1.0, 2.0, 3.0])) == Sequence(Taylor(2), [1 + 1im, 3 + 2im, 3im])
        ce, co = Sequence(evensym(ℱ), [1.0, 2.0]), Sequence(oddsym(ℱ), [3.0])
        @test project(complex(ce, co), ℱ) == complex(project(ce, ℱ), project(co, ℱ))
    end

    @testset "component, eachcomponent, unpack" begin
        𝕊 = 𝒯^2
        b = Sequence(𝕊, [1.0, 2.0, 3.0, 4.0])

        # components alias the parent
        c1 = component(b, 1)
        @test c1 == Sequence(𝒯, [1.0, 2.0])
        c1[0] = 100.0
        @test coefficients(b) == [100.0, 2.0, 3.0, 4.0]

        r = component(b, 1:2)
        @test space(r) == 𝕊
        @test coefficients(r) == coefficients(b)

        u = unpack(b)
        @test u == collect(eachcomponent(b)) == [component(b, i) for i ∈ 1:nspaces(𝕊)]
        u[2][0] = -5.0
        @test coefficients(b) == [100.0, 2.0, -5.0, 4.0]

        d = Sequence(𝒯 × ℱ, [1.0, 2.0, 3.0, 4.0, 5.0])
        @test component(d, 1) == Sequence(𝒯, [1.0, 2.0])
        @test component(d, 2) == Sequence(ℱ, [3.0, 4.0, 5.0])
        component(d, 1)[0] = 999.0
        @test coefficients(d)[1] == 999.0
    end

    @testset "conjugacy_symmetry!" begin
        # on a scalar only the real part survives
        @test conjugacy_symmetry!(Sequence(ScalarSpace(), [1.0 + 2.0im])) == Sequence(ScalarSpace(), [1.0 + 0.0im])

        # on Fourier, a ↦ (a + conj(reverse(a))) / 2
        f = Fourier(2, 1.0)
        b = Sequence(f, ComplexF64[1+1im, 2+2im, 3+0im, 4-1im, 5+3im])
        @test coefficients(conjugacy_symmetry!(b)) == ComplexF64[3-1im, 3+1.5im, 3+0im, 3-1.5im, 3+1im]

        # the real values 1:25 pair entry k with entry 26-k, hence 13 everywhere
        cplx = Sequence(f ⊗ f, ComplexF64.(1:25))
        @test all(==(13.0 + 0.0im), coefficients(conjugacy_symmetry!(cplx)))

        # component-wise on a cartesian space
        @test coefficients(conjugacy_symmetry!(Sequence(f^2, ComplexF64.(1:10)))) == ComplexF64[3, 3, 3, 3, 3, 8, 8, 8, 8, 8]
        @test coefficients(conjugacy_symmetry!(Sequence(CartesianProduct((f,)), ComplexF64.(1:5)))) == ComplexF64[3, 3, 3, 3, 3]

        # Taylor and Chebyshev coefficients of a real-valued function are real
        @test coefficients(conjugacy_symmetry!(Sequence(Taylor(2), ComplexF64[1+im, 2-im, 3im]))) == ComplexF64[1, 2, 0]
        @test coefficients(conjugacy_symmetry!(Sequence(𝒞, ComplexF64[1+im, 2-im, 3im]))) == ComplexF64[1, 2, 0]
        @test coefficients(conjugacy_symmetry!(Sequence(𝒯 ⊗ Chebyshev(1), ComplexF64[1+im, 2, 3-2im, 4im]))) == ComplexF64[1, 2, 3, 0]
        @test conjugacy_symmetry!(Sequence(Taylor(2), [1.0, 2.0, 3.0])) == Sequence(Taylor(2), [1.0, 2.0, 3.0])

        # mixed tensors are reflected along the Fourier axes only
        m = Sequence(ℱ ⊗ Chebyshev(1), ComplexF64[1+im, 2, 3-im, 4, 5+2im, 6im])
        ms = conjugacy_symmetry!(copy(m))
        @test all(ms[(-k, j)] == conj(ms[(k, j)]) for k ∈ -1:1, j ∈ 0:1)
        @test all(ms[(k, j)] == (m[(k, j)] + conj(m[(-k, j)])) / 2 for k ∈ -1:1, j ∈ 0:1)
        @test conjugacy_symmetry!(copy(ms)) == ms
        @test abs(imag(ms(0.3, 0.7))) < 1e-14
        t3 = Sequence(ℱ ⊗ 𝒯 ⊗ Fourier(1, 2.0), ComplexF64[cispi(0.1n) * n for n ∈ 1:18])
        t3s = conjugacy_symmetry!(copy(t3))
        @test all(t3s[(-k, j, -l)] == conj(t3s[(k, j, l)]) for k ∈ -1:1, j ∈ 0:1, l ∈ -1:1)
        @test all(t3s[(k, j, l)] == (t3[(k, j, l)] + conj(t3[(-k, j, -l)])) / 2 for k ∈ -1:1, j ∈ 0:1, l ∈ -1:1)
    end

    @testset "geometricweight, algebraicweight" begin
        # the fitted rates are 2 along Taylor and 3 along Fourier
        for (weight, decay) ∈ ((geometricweight, (x, k) -> x^k), (algebraicweight, (x, k) -> (1.0 + k)^x))
            a = Sequence(Taylor(10), [inv(decay(2, i)) for i ∈ 0:10])
            @test rate(weight(a)) ≈ 2 rtol=1e-8
            b = Sequence(Taylor(10) ⊗ Fourier(3, 1.0), vec([inv(decay(2, i) * decay(3, abs(j))) for i ∈ 0:10, j ∈ -3:3]))
            @test all(isapprox.(rate.(weight(b)), (2, 3); rtol = 1e-8))
        end
    end

    @testset "symmetric spaces (getcoefficient)" begin
        se = evensym(Taylor(4))
        @test collect(indices(se)) == [0, 2, 4]
        a = Sequence(se, [1.0, 2.0, 3.0])
        @test a[0] == 1.0 && a[2] == 2.0 && a[4] == 3.0
        @test RadiiPolynomial.getcoefficient(a, (se, 0)) == 1.0
        # read on the full space, the odd indices are zero
        @test [RadiiPolynomial.getcoefficient(a, (Taylor(4), k)) for k ∈ 0:2] == [1.0, 0.0, 2.0]
        # read through another symmetry, the sequence is projected: the odd part of an even sequence vanishes
        @test RadiiPolynomial.getcoefficient(a, (oddsym(Taylor(4)), 1)) == 0.0

        # an odd function vanishes at 0, which is not a representative
        fo = Fourier(3, 1.0)
        @test collect(indices(oddsym(fo))) == [1, 2, 3]
        b = Sequence(oddsym(fo), [10.0, 20.0, 30.0])
        @test [RadiiPolynomial.getcoefficient(b, (fo, k)) for k ∈ -1:1] == [-10.0, 0.0, 10.0]

        # the square symmetry identifies (i, j) with (j, i)
        fd = Fourier(2, 1.0) ⊗ Fourier(2, 1.0)
        @test collect(indices(d4sym(fd))) == [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 2)]
        c = Sequence(d4sym(fd), collect(1.0:6.0))
        @test RadiiPolynomial.getcoefficient(c, (fd, (1, 0))) == RadiiPolynomial.getcoefficient(c, (fd, (0, 1))) == 2.0
    end

    @testset "interval and complex coefficients" begin
        a = Sequence(Taylor(2), interval.([1, 2, 3]))
        @test isequal_interval(a[0], interval(1))
        @test in_interval(2, a[1])
        @test all(isequal_interval.(coefficients(zero(a)), interval(0)))
        @test a == Sequence(Taylor(2), interval.([1, 2, 3])) # thin intervals

        parent = Sequence(Taylor(2)^1, interval.([1, 2, 3]))
        b = component(parent, 1)
        @test isequal_interval(b[0], interval(1))
        b[0] = interval(9)
        @test isequal_interval(component(parent, 1)[0], interval(9))

        z = Sequence(𝒯, Complex{Interval{Float64}}.(interval.([1, 2])))
        @test isequal_interval(real(z)[0], interval(1))
        @test isequal_interval(imag(z)[0], interval(0))
        @test isequal_interval(real(conj(z)[0]), interval(1))

        # (1 + 3) / 2 = 2 at both ends, 2 unchanged at the centre
        ws = conjugacy_symmetry!(Sequence(ℱ, Complex{Interval{Float64}}.(interval.([1, 2, 3]))))
        @test all(isequal_interval.(real.(coefficients(ws)), interval(2)))
    end

    @testset "reverse, permutedims, selectdim" begin
        a = Sequence(Taylor(2), [1, 2, 3])
        @test reverse(a) == reverse!(copy(a)) == Sequence(Taylor(2), [3, 2, 1])

        d = Sequence(𝒯 ⊗ ℱ, collect(1:6))
        dp = permutedims(d, [2, 1])
        @test space(dp) == ℱ ⊗ 𝒯
        @test all(dp[(j, i)] == d[(i, j)] for i ∈ indices(𝒯), j ∈ indices(ℱ))

        a_𝑇 = Sequence(𝒯 ⊗ ℱ ⊗ 𝒞, collect(1:18))
        @test selectdim(a_𝑇, 2, 0) == selectdim(reshape(collect(1:18), 2, 3, 3), 2, 2)
    end

end
