## Aritmetics.jl

Basic & advanced arithmetic operations and their derivative (for backpropagation) + formatting functions.

## Features
- Arithmetic operations.
- Activation functions.
- Precision formatting.

## Key Functions
- `mean`, `std`: Basic statistics.
- `relu`, `leakyrelu`, `sigmoid`...: Activation functions.
- `rel_stable_diff`: Stable differentiation.
- `fp_2_floor`, `fp_2_round`: Formatting.
- `ϵ`, `ϵ64`: constans for standardization.

## Usages.
```julia
result = relu(input_value)   # Example: Activation Function
```
