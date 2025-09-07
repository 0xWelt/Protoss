"""Test version output."""

import protoss


def test_version_output():
    """Test that version can be accessed and printed."""
    print(f'Version: {protoss.__version__}')
    assert protoss.__version__ is not None
