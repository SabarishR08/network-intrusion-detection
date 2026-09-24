import sys
from pathlib import Path

# Ensure src is on the path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import src  # noqa: F401


def test_src_module_loads():
    """Verify the src package imports without error."""
    assert src is not None


if __name__ == "__main__":
    test_src_module_loads()
    print("Smoke test passed: src module loads successfully")
