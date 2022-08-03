# Pseudo-arclength continuation of equilibria of the FitzHugh-Nagumo model

In this example, we will prove the existence of a branch of equilibria of the FitzHugh-Nagumo model

```math
\begin{cases}
\displaystyle \frac{d}{dt} u(t) = f(\gamma, u(t)) := \begin{pmatrix} u_1(t)(u_1(t) - a)(1 - u_1(t)) - u_2(t) \\ \varepsilon(u_1(t) - \gamma u_2(t)) \end{pmatrix},\\
u(0) = u_0 \in \mathbb{R}^2,
\end{cases}
```

where ``a = 5`` and ``\varepsilon = 1``.

The vector-field ``f`` and its Jacobian, denoted ``Df``, may be implemented as follows:

```@example fhn_pseudo_arclength
function f(x)
    a, ϵ = 5, 1
    γ, u₁, u₂ = x
    return [u₁*(u₁ - a)*(1 - u₁) - u₂, ϵ*(u₁ - γ*u₂)]
end

function Df(x)
    a, ϵ = 5, 1
    γ, u₁, u₂ = x
    return [0 a*(2u₁-1)+(2-3u₁)*u₁ -1 ; -ϵ*u₂ ϵ -ϵ*γ]
end
nothing # hide
```

To this end, we use the [pseudo-arclength continuation](https://en.wikipedia.org/wiki/Numerical_continuation#Pseudo-arclength_continuation) and prove, at each step, that there exists a box, surrounding the linear numerical approximation, which contains the desired curve.

In a nutshell, the pseudo-arclength continuation consists in computing a sequence of numerical zeros of ``f``. Starting with an initial approximate zero ``x_\text{init} \in \mathbb{R}^3``, we retrieve an approximate tangent vector ``v`` to the curve at ``x_\text{init}`` by looking at ``\ker Df(x_\text{init})``. Then, our predictor for the next zero is set to ``w := x_\text{init} + \delta v`` where ``\delta > 0`` represents the step size. The Newton's method is applied on the mapping ``F_\text{Newton} : \mathbb{R}^3 \to \mathbb{R}^3`` given by

```math
F_\text{Newton}(x) :=
\begin{pmatrix}
(x - w) \cdot v\\
f(x)
\end{pmatrix}.
```

The mapping ``F_\text{Newton}`` and its Jacobian may be implemented as follows:

```@example fhn_pseudo_arclength
import LinearAlgebra: ⋅

F(x, v, w) = [(x - w) ⋅ v ; f(x)]

DF(x, v) = [transpose(v) ; Df(x)]
nothing # hide
```

Next, we perform Newton's method:

```@example fhn_pseudo_arclength
using RadiiPolynomial
import LinearAlgebra: nullspace

x_init = [2, 1.129171306613029, 0.564585653306514] # initial point on the branch of equilibria

v = vec(nullspace(Df(x_init))) # initial tangent vector

δ = 5e-2 # step size

w = x_init + δ * v # predictor

x_final, success = newton(x -> (F(x, v, w), DF(x, v)), w)
nothing # hide
```

Once the Newton's method converged to some ``x_\text{final} \in \mathbb{R}^3``, we make a linear approximation of the curve of zeros

```math
x_0(s) := x_\text{init} + s (x_\text{final} - x_\text{init}), \qquad \text{for all } s \in [0,1].
```

Define the mapping ``F : \mathbb{R}^3 \times [0,1] \to \mathbb{R}^3`` by

```math
F(x, s) :=
\begin{pmatrix}
(x - x_0(s)) \cdot v\\
f(x)
\end{pmatrix}
```

and the fixed-point operator ``T : \mathbb{R}^3 \times [0,1] \to \mathbb{R}^3`` by

```math
T(x, s) := x - A F(x, s),
```

where ``A : \mathbb{R}^3 \to \mathbb{R}^3`` is the injective operator corresponding to a numerical approximation of ``D_x F(x_0(s), s)^{-1}`` for all ``s \in [0, 1]``.

Let ``R > 0``. We use a uniform version of the [second-order Radii Polynomial Theorem with ``C^2`` condition](@ref C2_condition_RPT) such that we need to estimate ``|T(x_0(s), s) - x_0(s)|_\infty``, ``|D_x T(x_0(s), s)|_\infty`` and ``\sup_{x \in \text{cl}( B_R(x_0(s)) )} |D_x^2 T(x, s)|_\infty`` for all ``s \in [0,1]``. In particular, we have

```math
|T(x_0(s), s) - x_0(s)|_\infty = \left|A \begin{pmatrix} 0 \\ f(x_0(s)) \end{pmatrix} \right|_\infty, \qquad \text{for all } s \in [0,1].
```

The computer-assisted proof may be implemented as follows:

```@example fhn_pseudo_arclength
R = 1e-1

x₀_interval = Interval.(x_init) .+ Interval(0.0, 1.0) .* (Interval.(x_final) .- Interval.(x_init))
x₀R_interval = Interval.(inf.(x₀_interval .- R), sup.(x₀_interval .+ R))

F_interval = F(x₀_interval, v, x₀_interval)
F_interval[1] = 0 # the first component is trivial by definition
DF_interval = DF(x₀_interval, v)

A = inv(mid.(DF_interval))

Y = norm(Sequence(A * F_interval), Inf)

Z₁ = opnorm(LinearOperator(A * DF_interval - I), Inf)

a, ϵ = 5, 1
Z₂ = opnorm(LinearOperator(Interval.(A)), Inf) * max(2abs(a) + 2 + 6(abs(x₀_interval[2]) + R), 2abs(ϵ))

showfull(interval_of_existence(Y, Z₁, Z₂, R, C²Condition()))
```

Whenever the proof is successful, we proceed to the next iteration of the pseudo-arclength continuation and repeat the above strategy.

The following animation[^1] shows the successive steps of a rigorous pseudo-arclength continuation of equilibria of the FitzHugh-Nagumo model.

[^1]: S. Danisch and J. Krumbiegel, [Makie.jl: Flexible high-performance data visualization for Julia](https://doi.org/10.21105/joss.03349), *Journal of Open Source Software*, **6** (2021), 3349.

```@raw html
<video width="800" height="400" controls autoplay>
  <source src="../fhn_pseudo_arclength.mp4" type="video/mp4">
</video>
```
