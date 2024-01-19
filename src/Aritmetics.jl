module Aritmetics
using Revise
using Boilerplate
using Printf


const ⌂ = 1f27
const epsilon = 1f-8
const ϵ   = epsilon
const ϵ64 = 1e-11
const ω   = 1f-36
const ω64 = 1e-111

mean(arr; dims) = sum(arr, dims=dims) / prod(size(arr, r) for r in dims)
# mean(arr, dims::Nothing=nothing) = sum(arr) / length(arr)
mean(arr) = sum(arr) / length(arr)
std(arr) = sqrt(mean((arr .- mean(arr)) .^ 2))
std_deviation = std



@inline relu(v)          = ifelse(v > 0f0, v, 0f0)  # 
@inline leakyrelu(v)     = ifelse(v > 0f0, v, v * 0.02f0) 
@inline sigmoid(v)       = 1f0 / (1f0 + exp(-v)) 
@inline neg(v)           = -v
const ∠ = relu
const α = leakyrelu
const ∇ = abs
const σ = sigmoid
const e = exp
const ⌐ = neg

@inline ∂relu(g, v)      		= ifelse(v > 0f0, g, 0f0)  
@inline ∂relu(g, v, out)    = ifelse(out > 0f0, g, 0f0)  
@inline ∂leakyrelu(g, v) 		= ifelse(v > 0f0, g, g * 0.02f0)  
@inline ∂leakyrelu(g,v,out) = ifelse(out > 0f0, g, g * 0.02f0)  
@inline ∂abs(g, v)       		= ifelse(v ≥ 0f0, g, -g)
@inline ∂abs(g, v, out)     = @assert false "Not possible to calculate"
@inline ∂sigmoid(g, v)   		= (out = σ(v); out * (1f0 - out) * g)
@inline ∂sigmoid(g, v, out) = out * (1f0 - out) * g
@inline ∂exp(g, v)       		= exp(v) * g
@inline ∂exp(g, v, out)  		= out * g
@inline ∂neg(g, v)       		= -g
@inline ∂neg(g, v, out)  		= -g
const ∂∠ = ∂relu
const ∂α = ∂leakyrelu
const ∂∇ = ∂abs
const ∂σ = ∂sigmoid
const ∂e = ∂exp
const ∂⌐ = ∂neg



# Please be careful, the square with 1f19 or more, break the whole aritmetic!  Also at 1f14 it becomes shit! 
@inline rel_stable_diff(l::Tuple{T,T}) where T = rel_stable_diff(l[1], l[2]) 
@inline rel_stable_diff(lₜ::Float32, lₜ₋₁::Float32, ::Val{:NOCHECK}) = (lₜ-lₜ₋₁) / sqrt((abs(lₜ)^2 + abs(lₜ₋₁)^2) + ω)   + 1f0
@inline rel_stable_diff(lₜ::Float64, lₜ₋₁::Float64, ::Val{:NOCHECK}) = (lₜ-lₜ₋₁) / sqrt((abs(lₜ)^2 + abs(lₜ₋₁)^2) + ω64) + 1e0
@inline rel_stable_diff(lₜ::Float32, lₜ₋₁::Float32) = (((lₜ > 1f18  || lₜ₋₁ > 1f18 ) && @warn "rel_stable_diff accuracy breaks"); 
																						                    return (lₜ-lₜ₋₁) / sqrt((abs(lₜ)^2 + abs(lₜ₋₁)^2) + ω)   + 1f0)
@inline rel_stable_diff(lₜ::Float64, lₜ₋₁::Float64) = (((lₜ > 1e111 || lₜ₋₁ > 1e111) && @warn "rel_stable_diff accuracy breaks"); 
																																return (lₜ-lₜ₋₁) / sqrt((abs(lₜ)^2 + abs(lₜ₋₁)^2) + ω64) + 1e0) 

# const α = 1f-11
# @inline rel_diff(lₜ, lₜ₋₁) = (@assert lₜ > 0 "this rel_diff doesn't work with negative values..."; return lₜ / lₜ₋₁)
@inline rel_diff(lₜ, lₜ₋₁) = (@assert lₜ > 0 "this rel_diff doesn't work with negative values..."; return lₜ₋₁ / lₜ)
# @inline rel_diff(lₜ, lₜ₋₁) = 1/(1.1^(lₜ) / 1.1^(lₜ₋₁)) * 1f0 # * log(abs(lₜ)+abs(lₜ₋₁)+1)
# lₜ > 0 && lₜ₋₁ > 0 ? lₜ₋₁ / (lₜ + α) :
														# lₜ < 0 && lₜ₋₁ < 0 ? 2 / α - lₜ / (lₜ₋₁ + α) :
														# lₜ₋₁ > lₜ ? lₜ / (lₜ + α) + lₜ₋₁ / α : lₜ₋₁ / (lₜ + α) - 1 / α





validate_dims(list, dim, ref_dim) = begin
	range = [di for di in 1:length(ref_dim) if di !== dim]
	ref_dim_focused = ref_dim[range]
	for l_i in list[2:end] 
		!all(size(l_i)[range] .=== ref_dim_focused[range]) && return false
	end
	return true
end
get_max_size(list::Vector, dim) = get_max_size(list, Val(true))
get_max_size(list::Vector, dim, strict::Val{true}) = begin
  @assert validate_dims(list, dim, size(list[1])) "All sizes needs to be the same." # size($(i-1))!=size($i) $(last_s) and $(size(l_i))" # TODO Tomi these aren't existing variables in this scope... what did we want to do with there?
  size(list[1])
end
get_max_size(list::Vector, dim, strict::Val{false}) = begin
  max_size = [size(list[1])...]
  for l_i in list[2:end]
		for j in 1:ndims(l_i)
			size(l_i,j) > max_size[j] && (max_size[j] = size(l_i,j))
		end
  end
  max_size
end
sum_concat_dims(list, dim) = ndims(list[1]) <= dim ? sum([size(l, dim) for l in list]) : ndims(list[1]) + 1 == dim : length(list) : 1 # (println("dims are too strange... rank is: $(ndims(list[1]) requested dim: $dim)") ; 1)
cat_nospread(list::Vector{Array{T,N}}, dim, strict::Val=Val(true)) where {T,N} = begin
  max_size::NTuple{N,Int64} = get_max_size(list, dim, strict)
	concat_dim = sum_concat_dims(list, dim)
	NSize = dim <= N ? N : dim
  data::Array{T, NSize} = zeros(Float32, [i == dim ? concat_dim : max_size[i] for i in 1:NSize]...)
	assign_list(data, list, Val(dim))
  data
end
vcat_nospread(list::Vector{Array{T,N}}, strict::Val=Val(true)) where {T,N} = cat_nospread(list, 1, strict)
hcat_nospread(list::Vector{Array{T,N}}, strict::Val=Val(true)) where {T,N} = cat_nospread(list, 2, strict)

assign_list(trg::Matrix, src_list, dim::Val{1}) = @inbounds for (i, d) in enumerate(src_list)
	s1 = size(d,1)
	for j in 0:div(length(d),s1)-1
		for k in 1:s1
			trg[(i-1)*s1 + j*s1*length(list) + k] = d[j*s1+k]
		end
	end
end
# TODO make this better... Also Broadcast is pretty slow sometimes...
assign_list(trg::Matrix, src_list, dim::Val{2}) = begin
	i = 1
	for l in src_list 
		trg[:,i:i+size(l, 2)-1] .= l 
		i += size(l, 2)
	end
end

stack1(l::Vector{Array{T,N}}, strict::Val=Val(true)) where {T,N} = begin
  max_size::NTuple{N,Int64} = get_max_size(l, strict)
  data::Array{T, N+1} = zeros(Float32, length(l), max_size...)
  @inbounds for (i, d) in enumerate(l)
    for j in eachindex(d)
      data[i + (j-1)*length(l)] = d[j]
    end
  end
  data
end

# Formatting:
fp_2_floor(x) = floor(x, digits=Int(floor(-log10(x)))+1)
fp_2_round(x::ANY) where ANY = x	
fp_2_round(x::AbstractFloat) = (
	if     0.0000001 > x > -0.000001 return x
	elseif 0.00001   > x > -0.00001  return round(x,digits=8)
	elseif 0.001     > x > -0.001    return round(x,digits=6)
	elseif 0.1       > x > -0.1      return round(x,digits=4)
	elseif 10.       > x > -10.      return round(x,digits=2)
	elseif 1000.     > x > -1000.    return round(x,digits=0)
	else return x
  end)
fp_2_str(val::ANY) where ANY = return "$val"
fp_2_str(val::AbstractFloat) = (
	if     val >= 10000f0              || -10000f0             >= val return @sprintf("%.0f",val)
	elseif val >= 10f0                 || -10f0                >= val return @sprintf("%.2f", val)
	elseif val >= 0.01f0               || -0.01f0              >= val return @sprintf("%.4f", val)
	elseif val >= 0.0001f0             || -0.0001f0            >= val return @sprintf("%.6f", val)
	elseif val >= 0.000001f0           || -0.000001f0          >= val return @sprintf("%.8f", val)
	elseif val >= 0.00000001f0         || -0.00000001f0        >= val return @sprintf("%.10f", val)
	elseif val >= 0.0000000001f0       || -0.0000000001f0      >= val return @sprintf("%.12f", val)
	elseif val >= 0.000000000001f0     || -0.000000000001f0    >= val return @sprintf("%.14f", val)
	elseif val >= 0.000000000000001f0  || -0.000000000000001f0 >= val return @sprintf("%.22f", val)
	elseif val >= 1f-20                || -1f-20 >= val return "0" # @sprintf("%.20f", val)
	# elseif val >= 1f-20         || -1f-20 >= val return @sprintf("%.20f", val)
	# elseif val >= 1f-23         || -1f-23 >= val return @sprintf("%.23f", val)
	# elseif val >= 1f-26         || -1f-26 >= val return @sprintf("%.31f", val)
	else  return "0"
  end)

big_round(x) = (
	for mod in [
		1_000_000_000,
		100_000_000,
		10_000_000,
		1_000_000,
		100_000,
		10_000,
		1_000,
		100,
		10,
		]
		if x>mod
			return round(x/mod, digits=0)*mod
		end
	end;
	return round(x,digits=0)
)



end