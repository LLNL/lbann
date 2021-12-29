import math
import lbann

def Gelu(x):
    # return 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x ** 3)))
    # Based on: https://github.com/huggingface/transformers/blob/38a716cd41f22f6a7d5ff3dc081903090198803a/examples/research_projects/bertabs/modeling_bertabs.py#L658
    sqrt_2_over_pi = math.sqrt(2 / math.pi)
    b_coef = 0.044715
    x_cubed = lbann.Multiply(lbann.Multiply(lbann.Identity(x), x), x)
    inner_tanh_x_comp = lbann.Add(x, lbann.Scale(x_cubed, constant=b_coef))
    tanh_x = lbann.Tanh(lbann.Scale(inner_tanh_x_comp, constant=sqrt_2_over_pi))
    return lbann.Scale(
        lbann.Multiply(x, lbann.AddConstant(tanh_x, constant=1)), constant=0.5
    )

def Silu(x):
    return lbann.Multiple(x, lbann.Sigmoid(x))
