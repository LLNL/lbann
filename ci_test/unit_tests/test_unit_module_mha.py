import lbann
import numpy as np
import pytest
import test_util

try:
    import torch
except (ModuleNotFoundError, ImportError):
    pytest.skip('PyTorch is required for this test', allow_module_level=True)

num_samples = 2
sequence_length = 32
embed_dim = 64
num_heads = 8


def make_mha_module(self_attention):
    np.random.seed(20230510)
    torch.manual_seed(20230510)
    samples = np.random.normal(size=(num_samples, 3, sequence_length,
                                     embed_dim)).astype(np.float32)

    # PyTorch implementation
    mha_module = torch.nn.MultiheadAttention(embed_dim=embed_dim,
                                             num_heads=num_heads,
                                             batch_first=True,
                                             bias=True).eval()
    # Compute reference values
    with torch.no_grad():
        # Randomize bias for testing
        mha_module.in_proj_bias[:] = torch.randn(embed_dim * 3)

        q, k, v = torch.from_numpy(samples).unbind(dim=1)
        if self_attention:
            acts_pt, _ = mha_module(q, q, q)
        else:
            acts_pt, _ = mha_module(q, k, v)
        acts_np = acts_pt.detach().cpu().numpy()

    return mha_module, (q, k, v), acts_np


@test_util.lbann_test(check_gradients=False)
def test_multihead_attention():
    mha_module, samples, activations = make_mha_module(self_attention=False)

    # Weights are packed in the PyTorch module by default, unpack to q/k/v
    q_w, k_w, v_w = mha_module.in_proj_weight.detach().split(embed_dim)
    q_b, k_b, v_b = mha_module.in_proj_bias.detach().split(embed_dim)

    # LBANN implementation
    tester = test_util.ModelTester()
    q, k, v = tester.inputs_like(*samples)
    tester.make_reference(activations)

    # Test module
    from lbann.modules.transformer.attention import MultiheadAttention
    mha = MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        self_attention=False,
    )

    mha.query_weights = [
        lbann.Weights(initializer=lbann.ValueInitializer(
            values=np.nditer(q_w.transpose(0, 1).contiguous()))),
        lbann.Weights(initializer=lbann.ValueInitializer(
            values=np.nditer(q_b.contiguous()))),
    ]
    mha.key_weights = [
        lbann.Weights(initializer=lbann.ValueInitializer(
            values=np.nditer(k_w.transpose(0, 1).contiguous()))),
        lbann.Weights(initializer=lbann.ValueInitializer(
            values=np.nditer(k_b.contiguous()))),
    ]
    mha.value_weights = [
        lbann.Weights(initializer=lbann.ValueInitializer(
            values=np.nditer(v_w.transpose(0, 1).contiguous()))),
        lbann.Weights(initializer=lbann.ValueInitializer(
            values=np.nditer(v_b.contiguous()))),
    ]

    mha.output_weights = [
        lbann.Weights(initializer=lbann.ValueInitializer(
            values=np.nditer(mha_module.out_proj.weight.detach().cpu().
                             transpose(0, 1).contiguous().numpy()))),
        lbann.Weights(initializer=lbann.ValueInitializer(values=np.nditer(
            mha_module.out_proj.bias.detach().cpu().contiguous().numpy()))),
    ]

    acts = mha(q, k, v)

    # Compute MSE loss w.r.t. verification tensor
    tester.set_loss_function(lbann.MeanSquaredError, acts)
    return tester


@test_util.lbann_test(check_gradients=False)
def test_self_attention():
    mha_module, samples, activations = make_mha_module(self_attention=True)

    # LBANN implementation
    tester = test_util.ModelTester()
    q = tester.inputs(samples[0])
    tester.make_reference(activations)

    # Test module
    from lbann.modules.transformer.attention import MultiheadAttention
    mha = MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        self_attention=True,
    )

    mha.qkv_weights = [
        lbann.Weights(initializer=lbann.ValueInitializer(
            values=np.nditer(mha_module.in_proj_weight.detach().cpu().transpose(0, 1).contiguous().numpy())
        )),
        lbann.Weights(initializer=lbann.ValueInitializer(
            values=np.nditer(mha_module.in_proj_bias.detach().cpu().numpy()))),
    ]

    mha.output_weights = [
        lbann.Weights(initializer=lbann.ValueInitializer(
            values=np.nditer(mha_module.out_proj.weight.detach().cpu().
                             transpose(0, 1).contiguous().numpy()))),
        lbann.Weights(initializer=lbann.ValueInitializer(values=np.nditer(
            mha_module.out_proj.bias.detach().cpu().contiguous().numpy()))),
    ]

    acts = mha(q, q, q)

    # Compute MSE loss w.r.t. verification tensor
    tester.set_loss_function(lbann.MeanSquaredError, acts)
    tester.extra_callbacks.append(lbann.CallbackPrintModelDescription())
    return tester
