import math
import lbann

def Gelu(x):
    # return 0.5 * x * (1 + tanh(sqrt(pi / 2) * (x + 0.044715 * x ** 3)))
    # Based on: https://github.com/pytorch/pytorch/issues/20464#issuecomment-492339005
    sqrt_pi_over_2 = math.sqrt(math.pi / 2)
    b_coef = 0.044715
    x_cubed = lbann.Multiply(lbann.Multiply(lbann.Identity(x), x), x)
    inner_tanh_x_comp = lbann.Add(x, lbann.Scale(x_cubed, constant=b_coef))
    tanh_x = lbann.Tanh(lbann.Scale(inner_tanh_x_comp, constant=sqrt_pi_over_2))
    return lbann.Scale(
        lbann.Multiply(x, lbann.AddConstant(tanh_x, constant=1)), constant=0.5
    )

def Silu(x):
    return lbann.Multiple(x, lbann.Sigmoid(x))
