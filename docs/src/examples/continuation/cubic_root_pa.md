```@contents
Pages = ["cubic_root_pa.md"]
Depth = 3
```

# Pseudo-arclength continuation for the cubic root

In this example, we prove the existence of a one-parameter family of solutions to

```math
f(u, \lambda) \bydef u^3 - \lambda, \qquad \lambda \in [-1, 1].
```

### Step 1: Problem definition

The map ``f`` and the derivatives ``D_u f`` and ``D_\lambda f`` are implemented as follows:

```@example cubic_root_pa
f(u, λ) = u^3 - λ

Duf(u, λ) = exact(3) * u^2

Dλf(u, λ) = exact(-1)
nothing # hide
```

Define the mapping ``F : \mathbb{R}^2 \times [-1,1] \to \mathbb{R}^2`` by

```math
F(x, s) \bydef
\begin{pmatrix}
f(x) \\
(x - \bx(s)) \cdot \bar{v}(s)
\end{pmatrix}, \qquad x = (u, \lambda) \in \mathbb{R}^2, \quad s \in [-1,1].
```

The operator ``F`` and its derivative are implemented as follows:

```@example cubic_root_pa
F(x, v, w) = [f(x[1], x[2])
              sum((x - w) .* v)]

DF(x, v) = [Duf(x[1], x[2]) Dλf(x[1], x[2])
            v[1]            v[2]]
nothing # hide
```

Consider the fixed-point operator ``T : \mathbb{R}^2 \times [-1,1] \to \mathbb{R}^2`` given by

```math
T(x, s) \bydef x - A(s) F(x, s),
```

where ``A(s) : \mathbb{R}^2 \to \mathbb{R}^2`` is an injective operator corresponding to a numerical approximation of ``DF(\bx(s), s)^{-1}``, for some approximate zero ``\bx(s) = (\bar{u}(s), \bar{\lambda}(s)) \in \mathbb{R}^2`` of ``F``, for all ``s \in [-1, 1]``.

### Step 2: Approximate zero (floating-point arithmetic)

We use the [pseudo-arclength continuation method](https://en.wikipedia.org/wiki/Numerical_continuation#Pseudo-arclength_continuation) to retrieve a numerical approximation of the curve

```math
\bx(s) \bydef \bx_0 + 2 \sum_{k = 1}^K \bx_k \phi_k (s), \qquad \text{for all } s \in [-1,1],
```

where ``\phi_k`` are the [Chebyshev polynomials of the first kind](https://en.wikipedia.org/wiki/Chebyshev_polynomials).

```@example cubic_root_pa
using RadiiPolynomial, LinearAlgebra

K = 25
K_fft = fft_size(Chebyshev(K))
npts = K_fft ÷ 2 + 1

arclength = 3.14
arclength_grid = [0.5 * arclength - 0.5 * cospi(2j/K_fft) * arclength for j = 0:npts-1]
x_grid = Vector{Vector{Float64}}(undef, npts)
v_grid = Vector{Vector{Float64}}(undef, npts)

# initialize

λ_init = -1.0
u_init = -1.0
u_init, success_newton = newton(u -> (f(u, λ_init), Duf(u, λ_init)), u_init)

direction = [0, 1] # increase the parameter
x_init = [u_init, λ_init]
x_grid[1] = x_init
v_grid[1] = vec(nullspace([Duf(x_grid[1][1], x_grid[1][2]) Dλf(x_grid[1][1], x_grid[1][2])]))
if sum(direction .* v_grid[1]) < 0 # enforce direction
    v_grid[1] .*= -1
end

# run continuation scheme

for j = 2:npts
    δ = arclength_grid[j] - arclength_grid[j-1]

    w = x_grid[j-1] + δ * v_grid[j-1] # predictor

    x_bar, success_newton_j = newton(x -> (F(x, v_grid[j-1], w), DF(x, v_grid[j-1])), w)
    success_newton_j || error()

    x_grid[j] = x_bar
    v_grid[j] = vec(nullspace([Duf(x_grid[j][1], x_grid[j][2]) Dλf(x_grid[j][1], x_grid[j][2])]))
    if sum(v_grid[j-1] .* v_grid[j]) < 0 # keep the same direction
        v_grid[j] .*= -1
    end
end

# construct the approximations

x_fft = [reverse(x_grid) ; x_grid[begin+1:end-1]]
x_cheb = [real(to_seq([z[i] for z = x_fft], Chebyshev(K))) for i = 1:2]

v_fft = [reverse(v_grid) ; v_grid[begin+1:end-1]]
v_cheb = [real(to_seq([z[i] for z = v_fft], Chebyshev(K))) for i = 1:2]
nothing # hide
```

### Step 3: Approximate inver (floating-point arithmetic)

```@example cubic_root_pa
A_grid = inv.(DF.(x_grid, v_grid))
A_fft = [reverse(A_grid) ; A_grid[begin+1:end-1]]
A_cheb = [real(to_seq([z[i,j] for z = A_fft], Chebyshev(K))) for i = 1:2, j = 1:2]
nothing # hide
```

### Step 4: Bounds estimation (interval arithmetic)

Let ``R > 0``. We use a uniform version of the second-order Radii Polynomial Theorem (cf. Section [Radii polynomial approach](@ref radii_polynomial_approach)) such that we need to estimate ``\|T(\bx(s), s) - \bx(s)\|_1``, ``\|D_x T(\bx(s), s)\|_1`` and ``\sup_{x \in B(\bx(s), R)} \|D_x^2 T(x, s)\|_1`` for all ``s \in [-1,1]``. In particular, we have

```math
\|T(\bx(s), s) - \bx(s)\|_1 = \left\|A(s) \begin{pmatrix} f(\bx(s)) \\ 0 \end{pmatrix} \right\|_1, \qquad \text{for all } s \in [-1,1].
```

The computer-assisted proof may be implemented as follows:

```@example cubic_root_pa
x_cheb_interval = interval.(x_cheb)
v_cheb_interval = interval.(v_cheb)
A_cheb_interval = interval.(A_cheb)

#- Y bound

Y = norm(norm.(A_cheb_interval * F(x_cheb_interval, v_cheb_interval, x_cheb_interval), 1), 1)

#- Z₁ bound

Z₁ = opnorm(norm.(Diagonal([exact(1), exact(1)]) - A_cheb_interval * DF(x_cheb_interval, v_cheb_interval), 1), 1)

#- Z₂ bound

R = 10 * sup(Y)
Z₂ = exact(3) * opnorm(norm.(A_cheb_interval, 1), 1) * (norm(norm.(x_cheb_interval, 1), 1) + exact(2 * R))

# verify the contraction

ie, contraction_success = interval_of_existence(Y, Z₁, Z₂, R; verbose = true)
nothing # hide
```

```@example cubic_root_pa
inf(ie) # smallest error
```

The following figure[^1] shows the numerical approximation of the proven branch of the cubic root which goes through the singular point ``0``.

[^1]: S. Danisch and J. Krumbiegel, [Makie.jl: Flexible high-performance data visualization for Julia](https://doi.org/10.21105/joss.03349), *Journal of Open Source Software*, **6** (2021), 3349.

```@example cubic_root_pa
using GLMakie

fig = Figure()

ax1 = Axis(fig[1,1], xticks = -2:2)
lines!(ax1, [Point2f(λ, cbrt(λ)) for λ = LinRange(-2, 2, 501)];
    color = :black, label = L"(s, s^{1/3})")
lines!(ax1, [Point2f(x_cheb[2](s), x_cheb[1](s)) for s = LinRange(-1, 1, 501)];
    color = :blue, label = L"(\bar{\lambda}(s),\bar{u}(s))")
scatter!(ax1, [Point2f(x_grid[j][2], x_grid[j][1]) for j = 1:npts];
    color = :red)
axislegend(ax1; position = :lt)

ax2 = Axis(fig[1,2])
lines!(ax2, [Point2f(s, x_cheb[1](s)) for s = LinRange(-1, 1, 501)];
    label = L"\bar{u}(s)")
lines!(ax2, [Point2f(s, x_cheb[2](s)) for s = LinRange(-1, 1, 501)];
    label = L"\bar{\lambda}(s)")
axislegend(ax2; position = :lt)

fig
```
