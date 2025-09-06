import os, pytest

MAIN_PDF = "Wasserstein_Distributionally_Robust_Optimization__DRO__Module.pdf"
TESTS_ZIP = "Test.zip"

def test_artifacts_exist_or_skip():
    missing = [p for p in [MAIN_PDF, TESTS_ZIP] if not os.path.exists(p)]
    if missing:
        pytest.skip("Missing research artifacts: " + ", ".join(missing))
    assert os.path.getsize(MAIN_PDF) > 0, f"{MAIN_PDF} exists but is empty"
