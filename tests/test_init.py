"""Tests for RoSE package initialization and metadata."""

import pytest


class TestPackageMetadata:
    """Test the package metadata and version handling."""

    def test_package_metadata_attributes(self):
        """Test that package metadata attributes are correctly set."""
        import RoSE

        # Test that basic package attributes exist
        assert hasattr(RoSE, "__version__")
        assert hasattr(RoSE, "__author__")
        assert RoSE.__author__ == "Jeff Rhoades"

    def test_main_classes_importable(self):
        """Test that main classes can be imported from the package."""
        import RoSE

        # Test that the main classes exist and are importable
        assert hasattr(RoSE, "RotarySpatialEmbedding")
        assert hasattr(RoSE, "RoSEMultiHeadCrossAttention")
        assert hasattr(RoSE, "MultiRes_RoSE_TransformerBlock")

        # Test that they are actually classes/callable
        assert callable(RoSE.RotarySpatialEmbedding)
        assert callable(RoSE.RoSEMultiHeadCrossAttention)
        assert callable(RoSE.MultiRes_RoSE_TransformerBlock)

    def test_package_docstring(self):
        """Test that the package has a proper docstring."""
        import RoSE

        assert RoSE.__doc__ is not None
        assert len(RoSE.__doc__.strip()) > 0
        # Check for key terms in the docstring
        docstring_lower = RoSE.__doc__.lower()
        assert "rotary" in docstring_lower
        assert "spatial" in docstring_lower
        assert "embeddings" in docstring_lower

    def test_version_fallback_logic(self):
        """Test the version fallback logic by examining the __init__.py file."""
        # Instead of mocking imports, test the logic by reading the code
        with open("src/RoSE/__init__.py", "r") as f:
            content = f.read()

        # Check that the version fallback logic is present
        assert "try:" in content
        assert "version(" in content
        assert "except PackageNotFoundError:" in content
        assert '"0.1.1"' in content
        assert "rotary-spatial-embeddings" in content

    def test_package_name_in_version_call(self):
        """Test that the correct package name is used for version lookup."""
        # Read the __init__.py file to check the package name
        with open("src/RoSE/__init__.py", "r") as f:
            content = f.read()

        # Check that the correct package name is used
        assert 'version("rotary-spatial-embeddings")' in content


class TestVersionValue:
    """Test version value behavior."""

    def test_version_is_string(self):
        """Test that __version__ is a string."""
        import RoSE

        assert isinstance(RoSE.__version__, str)
        assert len(RoSE.__version__) > 0

    def test_version_format(self):
        """Test that version follows expected format."""
        import RoSE

        # Should be in format like "x.y.z" or at least have dots for version
        version_parts = RoSE.__version__.split(".")
        assert len(version_parts) >= 2  # At least major.minor

        # First part should be numeric
        assert version_parts[0].isdigit()


if __name__ == "__main__":
    pytest.main([__file__])
