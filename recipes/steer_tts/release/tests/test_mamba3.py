"""The Mamba-3 layer's own invariant suite (paper-faithfulness vs two references, gradients, causality, streaming)."""


def test_mamba3_suite():
    from steertts.mamba3.tests import run
    assert run(False)
