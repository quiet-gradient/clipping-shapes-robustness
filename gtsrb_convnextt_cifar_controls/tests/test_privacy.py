"""Unit tests for the privacy-suite helpers (no training / no GPU)."""

from trustml.attacks.oslo_surrogates import available_validation_specs
from trustml.evaluation.privacy import _parse_tau_list


def test_parse_tau_list_accepts_string_scalar_and_sequence():
    assert _parse_tau_list("0.3,0.1,0.03") == [0.3, 0.1, 0.03]
    # A single CLI override like attacks.oslo.tau_list=0.1 arrives as a float.
    assert _parse_tau_list(0.1) == [0.1]
    assert _parse_tau_list([0.2, 0.05]) == [0.2, 0.05]
    # Trailing / empty fields in the string form are ignored.
    assert _parse_tau_list("0.3, 0.1, ") == [0.3, 0.1]


def test_available_validation_specs_skips_missing_vgg13(tmp_path):
    # Artifact weights are intentionally stored outside Git. Model discovery
    # should be tested from a synthetic file layout, not from a local backup.
    for family in ("densenet121", "shufflenet"):
        for index in range(3):
            (tmp_path / f"val_{family}_{index}.pth").touch()
    present, missing = available_validation_specs(tmp_path)
    families = {name for name, _ in present}
    # This repo ships densenet121 + shufflenet weights but never vgg13.
    assert "densenet121" in families
    assert "shufflenet" in families
    assert missing == ["val_vgg13_0.pth", "val_vgg13_1.pth", "val_vgg13_2.pth"]
    assert len(present) == 6


def test_available_validation_specs_empty_dir(tmp_path):
    present, missing = available_validation_specs(tmp_path)
    assert present == []
    assert len(missing) == 9
