
using PlotlyJS
using Aritmetics: rel_stable_diff, ω

@inline rel_stablee_diff(l::Tuple{T,T}) where T = rel_stablee_diff(l[1], l[2]) 
@inline rel_stablee_diff(lₜ::Float32, lₜ₋₁::Float32) = (lₜ-lₜ₋₁) / sqrt((abs(lₜ)^2 + abs(lₜ₋₁)^2) + ω)   + 1f0
m=rel_stablee_diff.(Base.product(0f0:0.1f0:10f0,0f0:0.1f0:10f0))

plot(surface(z=m))


#%%

using Aritmetics: rel_diff


@show rel_diff(2000f0,     1999f0),  2000f0 / 1999f0  # 2000/1999f0
@show rel_diff(200f0,      199.9f0), 200f0  / 199.9f0
@show rel_diff(20f0,       19.99f0), 20f0   / 19.99f0
@show rel_diff(2f0,        1.999f0), 2f0    / 1.999f0
@show rel_diff(0.2f0,      0.1999f0),    0.2f0  / 0.1999f0
@show rel_diff(0.02f0,     0.01999f0),   0.02f0 / 0.01999f0
@show rel_diff(0.02f0,     0.001999f0),  0.02f0 / 0.001999f0
@show rel_diff(0.02f0,     0.0001999f0), 0.02f0 / 0.0001999f0
@show rel_diff(2f-5,       1.999f-5),    2f-5   / 1.999f-5
@show rel_diff(2f-10,      1.999f-10),   2f-10  / 1.999f-10
@show rel_diff(2f-20,      1.999f-20),   2f-20  / 1.999f-20
@show rel_diff(2f-30,      1.999f-30),   2f-30  / 1.999f-30
@show rel_diff(2f-36,      1.999f-36),   2f-36  / 1.999f-36
@show rel_diff(0.01999f0,  0.02f0),      0.01999f0   / 0.02f0
@show rel_diff(0.001999f0, 0.02f0),      0.001999f0  / 0.02f0
@show rel_diff(0.0001999f0,0.02f0),      0.0001999f0 / 0.02f0
println()
@show rel_diff(-0.1f0,      0.2f0), -0.1f0       /  0.2f0
@show rel_diff(-0.0001999f0,0.2f0), -0.0001999f0 /  0.2f0
@show rel_diff(0.1f0,      -0.2f0),  0.1f0       / -0.2f0 
@show rel_diff(0.0001999f0,-0.2f0),  0.0001999f0 / -0.2f0
println()
@show rel_diff(-0.1999f0,   -0.2f0),  -0.1999f0    / -0.2f0
@show rel_diff(-0.01999f0,  -0.2f0),  -0.01999f0   / -0.2f0
@show rel_diff(-0.001999f0, -0.2f0),  -0.001999f0  / -0.2f0
@show rel_diff(-0.0001999f0,-0.2f0),  -0.0001999f0 / -0.2f0
@show rel_diff(-0.02f0,-0.01999f0),   -0.02f0 / -0.01999f0
@show rel_diff(-0.2f0, -0.1999f0),    -0.2f0  / -0.1999f0
@show rel_diff(-0.2f0, -0.01999f0),   -0.2f0  / -0.01999f0
@show rel_diff(-0.2f0, -0.001999f0),  -0.2f0  / -0.001999f0
@show rel_diff(-0.2f0, -0.0001999f0), -0.2f0  / -0.0001999f0

;

