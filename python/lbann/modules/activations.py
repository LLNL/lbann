import math
import lbann

def Gelu(x):
    x_erf = lbann.Erf(lbann.Scale(x, constant=(1/math.sqrt(2))))
    return lbann.Multiply(x, lbann.Scale(lbann.AddConstant(x_erf, constant=1), constant=0.5))

def Gelu_approx(x):
    # This approximates gelu and may be more performant
    # return 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x ** 3)))
    # Based on: https://paperswithcode.com/method/gelu
    sqrt_2_over_pi = math.sqrt(2 / math.pi)
    b_coef = 0.044715
    x_cubed = lbann.Multiply(lbann.Multiply(lbann.Identity(x), x), x)
    inner_tanh_x_comp = lbann.Add(x, lbann.Scale(x_cubed, constant=b_coef))
    tanh_x = lbann.Tanh(lbann.Scale(inner_tanh_x_comp, constant=sqrt_2_over_pi))
    return lbann.Scale(
        lbann.Multiply(x, lbann.AddConstant(tanh_x, constant=1)), constant=0.5
    )

def Silu(x):
    return lbann.Multiply(x, lbann.Sigmoid(x))
