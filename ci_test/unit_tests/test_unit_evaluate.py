""" Tests the LBANN evaluator. """
import lbann
import numpy as np


def test_evaluate_simple():
    a = lbann.Input(data_field='samples')
    b = lbann.AddConstant(a, constant=1)

    inputarr = np.random.rand(2, 3)
    outputarr = lbann.evaluate(b, inputarr)
    assert np.allclose(outputarr, inputarr + 1)


def test_evaluate_model():
    a = lbann.Input(data_field='samples')
    b = lbann.AddConstant(a, constant=2)
    model = lbann.Model(20, [a, b],
                        callbacks=[lbann.CallbackPrintModelDescription()])

    inputarr = np.random.rand(2, 3)
    outputarr = lbann.evaluate(model, inputarr)

    # Test outputs
    assert np.allclose(outputarr, inputarr + 2)

    # Test properties
    assert (len(model.callbacks) == 1 and isinstance(
        model.callbacks[0], lbann.CallbackPrintModelDescription))
    assert model.epochs == 20


def test_evaluate_multioutput():
    a = lbann.Input(data_field='samples')
    lbann.AddConstant(a, constant=1, name='one')
    lbann.AddConstant(a, constant=2, name='two')
    lbann.AddConstant(a, constant=3, name='three')

    inputarr = np.random.rand(2, 3)
    out1, out3 = lbann.evaluate(a, inputarr, ['one', 'three'])

    # Test outputs
    assert np.allclose(out1, inputarr + 1)
    assert np.allclose(out3, inputarr + 3)
