@testset "Vector spaces" begin
    @testset "VectorSpace" begin
        𝒫 = ScalarSpace()
        𝒯 = Taylor(0)
        # different types of vector spaces cannot be equal
        @test !(𝒫 == 𝒯)
        # different types of vector spaces cannot be subset of one another
        @test !(𝒫 ⊆ 𝒯)
        # # intersection and union between different types of vector spaces is not allowed
        # @test_throws MethodError intersect(𝒫, 𝒯)
        # @test_throws MethodError union(𝒫, 𝒯)
    end

    @testset "ScalarSpace" begin
        𝒫 = ScalarSpace()
        @test 𝒫 ⊆ 𝒫
        @test 𝒫 ∩ 𝒫 == 𝒫 ∪ 𝒫 == 𝒫
        @test dimension(𝒫) == 1
        @test RadiiPolynomial._firstindex(𝒫) == 1
        @test RadiiPolynomial._lastindex(𝒫) == 1
        @test indices(𝒫) == Base.OneTo(1)
        @test RadiiPolynomial._findposition(1, 𝒫) == 1
    end

    @testset "Taylor" begin
        𝒯₂, 𝒯₃ = Taylor(2), Taylor(3)
        @test !(𝒯₂ == 𝒯₃)
        @test 𝒯₂ ⊆ 𝒯₃
        @test 𝒯₂ ∩ 𝒯₃ == 𝒯₂
        @test 𝒯₂ ∪ 𝒯₃ == 𝒯₃
        @test order(𝒯₂) == 2
        @test dimension(𝒯₂) == 3
        @test RadiiPolynomial._firstindex(𝒯₂) == 0
        @test RadiiPolynomial._lastindex(𝒯₂) == 2
        @test indices(𝒯₂) == 0:2
        @test RadiiPolynomial._findindex_constant(𝒯₂) == 0
        @test RadiiPolynomial._findposition(1, 𝒯₂) == 2
        @test RadiiPolynomial._findposition(0:2, 𝒯₂) == 1:3
        @test RadiiPolynomial._findposition([0, 2], 𝒯₂) == [1, 3]
        @test RadiiPolynomial._findposition(:, 𝒯₂) == Colon()
    end

    @testset "Fourier" begin
        ℱ₂, ℱ₃ = Fourier(2, 1.0), Fourier(3, 1.0)
        @test !(ℱ₂ == ℱ₃)
        @test ℱ₂ ⊆ ℱ₃
        @test ℱ₂ ∩ ℱ₃ == ℱ₂
        @test ℱ₂ ∪ ℱ₃ == ℱ₃
        @test order(ℱ₂) == 2
        @test frequency(ℱ₂) == 1.0
        @test dimension(ℱ₂) == 5
        @test RadiiPolynomial._firstindex(ℱ₂) == -2
        @test RadiiPolynomial._lastindex(ℱ₂) == 2
        @test indices(ℱ₂) == -2:2
        @test RadiiPolynomial._findindex_constant(ℱ₂) == 0
        @test RadiiPolynomial._findposition(-1, ℱ₂) == 2
        @test RadiiPolynomial._findposition(0:1, ℱ₂) == 3:4
        @test RadiiPolynomial._findposition([-2, 2], ℱ₂) == [1, 5]
        @test RadiiPolynomial._findposition(:, ℱ₂) == Colon()
        @test convert(Fourier{Float64}, ℱ₂) == ℱ₂
        @test convert(Fourier{Int}, ℱ₂) == Fourier(2, 1)
        @test promote_type(Fourier{Float64}, Fourier{Float64}) == Fourier{Float64}
        @test promote_type(Fourier{Float64}, Fourier{Int}) == Fourier{Float64}
    end

    @testset "Chebyshev" begin
        𝒞₂, 𝒞₃ = Chebyshev(2), Chebyshev(3)
        @test !(𝒞₂ == 𝒞₃)
        @test 𝒞₂ ⊆ 𝒞₃
        @test 𝒞₂ ∩ 𝒞₃ == 𝒞₂
        @test 𝒞₂ ∪ 𝒞₃ == 𝒞₃
        @test order(𝒞₂) == 2
        @test dimension(𝒞₂) == 3
        @test RadiiPolynomial._firstindex(𝒞₂) == 0
        @test RadiiPolynomial._lastindex(𝒞₂) == 2
        @test indices(𝒞₂) == 0:2
        @test RadiiPolynomial._findindex_constant(𝒞₂) == 0
        @test RadiiPolynomial._findposition(1, 𝒞₂) == 2
        @test RadiiPolynomial._findposition(0:2, 𝒞₂) == 1:3
        @test RadiiPolynomial._findposition([0, 2], 𝒞₂) == [1, 3]
        @test RadiiPolynomial._findposition(:, 𝒞₂) == Colon()
    end

    @testset "TensorSpace" begin
        𝒯₂, ℱ₂, 𝒞₂ = Taylor(2), Fourier(2, 1.0), Chebyshev(2)
        𝑇 = TensorSpace((𝒯₂, ℱ₂, 𝒞₂))
        @test spaces(𝑇) == (𝒯₂, ℱ₂, 𝒞₂)
        @test nspaces(𝑇) == 3
        @test 𝒯₂ ⊗ ℱ₂ ⊗ 𝒞₂ == (𝒯₂ ⊗ ℱ₂) ⊗ 𝒞₂ == 𝒯₂ ⊗ (ℱ₂ ⊗ 𝒞₂) == 𝑇
        @test 𝑇 ⊗ 𝑇 == TensorSpace((𝒯₂, ℱ₂, 𝒞₂, 𝒯₂, ℱ₂, 𝒞₂))
        @test 𝑇[1] == 𝒯₂
        @test 𝑇[1:3] == 𝑇[[1, 2, 3]] == 𝑇[:] == 𝑇
        @test Base.tail(𝑇) == TensorSpace((ℱ₂, 𝒞₂))
        @test Base.front(𝑇) == TensorSpace((𝒯₂, ℱ₂))
        @test 𝑇 ⊆ Taylor(3) ⊗ ℱ₂ ⊗ 𝒞₂
        @test 𝑇 ∩ (Taylor(3) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(3)) == 𝒯₂ ⊗ Fourier(1, 1.0) ⊗ 𝒞₂
        @test 𝑇 ∪ (Taylor(3) ⊗ Fourier(1, 1.0) ⊗ Chebyshev(3)) == Taylor(3) ⊗ ℱ₂ ⊗ Chebyshev(3)
        @test dimension(𝑇) == 45
        @test dimension(𝑇, 1) == 3
        @test dimensions(𝑇) == (3, 5, 3)
        @test RadiiPolynomial._firstindex(𝑇) == (0, -2, 0)
        @test RadiiPolynomial._lastindex(𝑇) == (2, 2, 2)
        @test indices(𝑇) == TensorIndices((0:2, -2:2, 0:2))
        @test RadiiPolynomial._findindex_constant(𝑇) == (0, 0, 0)
        @test RadiiPolynomial._findposition((1, 2, 0), 𝑇) == 14
        @test RadiiPolynomial._findposition((:, -2:2, :), 𝑇) == collect(1:45)
        @test RadiiPolynomial._findposition([(1, 2, 0), (2, 2, 2)], 𝑇) == [14, 45]
        @test RadiiPolynomial._findposition((:, :, :), 𝑇) == Colon()
        @test RadiiPolynomial._findposition(:, 𝑇) == Colon()
        @test order(𝑇) == (2, 2, 2)
        @test order(𝑇, 1) == 2
        @test_throws MethodError frequency(𝑇)
        @test frequency(𝑇, 2) == 1.0
        @test convert(TensorSpace{Tuple{Taylor,Fourier{Float64},Chebyshev}}, 𝑇) == 𝑇
        @test convert(TensorSpace{Tuple{Taylor,Fourier{Int},Chebyshev}}, 𝑇) == Taylor(2) ⊗ Fourier(2, 1) ⊗ Chebyshev(2)
    end

    @testset "CartesianSpace" begin
        𝒯₁¹ = CartesianPower(Taylor(1), 1)
        𝒞₂² = CartesianPower(Chebyshev(2), 2)
        ℱ₃³ = CartesianPower(Fourier(3, 1.0), 3)
        𝒯₁¹𝒞₂²ℱ₃³ = CartesianProduct((𝒯₁¹, 𝒞₂², ℱ₃³))
        𝒯₂¹𝒞₃²ℱ₃³ = CartesianProduct((CartesianPower(Taylor(2), 1), CartesianPower(Chebyshev(3), 2), ℱ₃³))
        @test space(𝒯₁¹) == Taylor(1)
        @test spaces(𝒯₁¹) == [Taylor(1)]
        @test spaces(𝒯₁¹𝒞₂²ℱ₃³) == (𝒯₁¹, 𝒞₂², ℱ₃³)
        @test nspaces(ℱ₃³) == nspaces(𝒯₁¹𝒞₂²ℱ₃³) == 3
        @test 𝒯₁¹^1 == CartesianPower(𝒯₁¹, 1)
        @test 𝒯₁¹ × 𝒞₂² × ℱ₃³ == 𝒯₁¹ × (𝒞₂² × ℱ₃³) == (𝒯₁¹ × 𝒞₂²) × ℱ₃³ == (𝒯₁¹ × 𝒞₂²) × CartesianProduct(tuple(ℱ₃³)) == 𝒯₁¹𝒞₂²ℱ₃³
        @test Base.front(𝒯₁¹𝒞₂²ℱ₃³) == 𝒯₁¹ × 𝒞₂²
        @test Base.tail(𝒯₁¹𝒞₂²ℱ₃³) == 𝒞₂² × ℱ₃³
        @test 𝒞₂²[1] == Chebyshev(2)
        @test 𝒞₂²[1:2] == 𝒞₂²[[1, 2]] == 𝒞₂²[:] == 𝒞₂²
        @test 𝒯₁¹𝒞₂²ℱ₃³[1] == 𝒯₁¹
        @test 𝒯₁¹𝒞₂²ℱ₃³[1:3] == 𝒯₁¹𝒞₂²ℱ₃³[[1, 2, 3]] == 𝒯₁¹𝒞₂²ℱ₃³[:] == 𝒯₁¹𝒞₂²ℱ₃³
        @test 𝒯₁¹𝒞₂²ℱ₃³ ⊆ 𝒯₂¹𝒞₃²ℱ₃³
        @test 𝒯₁¹𝒞₂²ℱ₃³ ∩ 𝒯₂¹𝒞₃²ℱ₃³ == 𝒯₁¹𝒞₂²ℱ₃³
        @test 𝒯₁¹𝒞₂²ℱ₃³ ∪ 𝒯₂¹𝒞₃²ℱ₃³ == 𝒯₂¹𝒞₃²ℱ₃³
        @test dimension(𝒯₁¹𝒞₂²ℱ₃³) == 29
        @test dimension(𝒯₁¹𝒞₂²ℱ₃³, 1) == dimension(𝒯₁¹, 1) == 2
        @test dimensions(𝒯₁¹) == [2]
        @test dimensions(𝒯₁¹𝒞₂²ℱ₃³) == (2, 6, 21)
        @test RadiiPolynomial._firstindex(𝒯₁¹𝒞₂²ℱ₃³) == RadiiPolynomial._firstindex(𝒞₂²) == 1
        @test RadiiPolynomial._lastindex(𝒞₂²) == 6
        @test RadiiPolynomial._lastindex(𝒯₁¹𝒞₂²ℱ₃³) == 29
        @test indices(𝒞₂²) == Base.OneTo(6)
        @test indices(𝒯₁¹𝒞₂²ℱ₃³) == Base.OneTo(29)
        @test order(𝒯₁¹𝒞₂²ℱ₃³) == ([1], [2, 2], [3, 3, 3])
        @test order(𝒯₁¹𝒞₂²ℱ₃³, 1) == [1]
        @test order(𝒯₁¹, 1) == 1
        @test_throws MethodError frequency(𝒯₁¹𝒞₂²ℱ₃³)
        @test frequency(𝒯₁¹𝒞₂²ℱ₃³, 3) == frequency(ℱ₃³) == [1.0, 1.0, 1.0]
        @test frequency(ℱ₃³, 1) == 1.0
        @test RadiiPolynomial._component_findposition(1, 𝒯₁¹) == RadiiPolynomial._component_findposition(1:1, 𝒯₁¹) == 1:2
        @test RadiiPolynomial._component_findposition(2, 𝒞₂²) == RadiiPolynomial._component_findposition(2:2, 𝒞₂²) == 4:6
        @test RadiiPolynomial._component_findposition(1, 𝒯₁¹𝒞₂²ℱ₃³) == RadiiPolynomial._component_findposition(1:1, 𝒯₁¹𝒞₂²ℱ₃³) == 1:2
        @test RadiiPolynomial._component_findposition(2, 𝒯₁¹𝒞₂²ℱ₃³) == RadiiPolynomial._component_findposition(2:2, 𝒯₁¹𝒞₂²ℱ₃³) == 3:8
        @test convert(CartesianProduct{Tuple{CartesianPower{Taylor}, CartesianPower{Chebyshev}, CartesianPower{Fourier{Float64}}}}, 𝒯₁¹𝒞₂²ℱ₃³) == 𝒯₁¹𝒞₂²ℱ₃³
        @test convert(CartesianProduct{Tuple{CartesianPower{Taylor}, CartesianPower{Chebyshev}, CartesianPower{Fourier{Int}}}}, 𝒯₁¹𝒞₂²ℱ₃³) == 𝒯₁¹ × 𝒞₂² × Fourier(3, 1)^3
    end
end
