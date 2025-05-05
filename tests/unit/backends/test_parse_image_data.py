import base64
import pytest

from app.backends.utils import parse_data_uri
from app.backends.exceptions import InvalidBase64DataError, InvalidImageURLError


def test_convert_base64_image_to_bytes():
    """It should return a dict with correct image format and conversion of base64 to bytes"""

    original_bytes = b'abcd'
    base_64 = base64.b64encode(original_bytes).decode()
    image_uri = f"data:image/png;base64,{base_64}"

    res = parse_data_uri(image_uri)
    assert res['format'] == 'png'
    assert res['data'] == original_bytes


def test_raises_unsupported_file_type():
    """It should raise when we don't support the file type"""

    original_bytes = b'abcd'
    base_64 = base64.b64encode(original_bytes).decode()
    image_uri = f"data:image/tif;base64,{base_64}"

    with pytest.raises(InvalidImageURLError) as exc_info:
        parse_data_uri(image_uri)
    assert "Invalid or unsupported image data URI format." in str(exc_info.value)


def test_raises_bad_base64_type():
    """It should raise with invalid base64 data"""
    bad_base64 = "ABCXYZ"
    image_uri = f"data:image/jpeg;base64,{bad_base64}"

    with pytest.raises(InvalidBase64DataError) as exc_info:
        parse_data_uri(image_uri)
    assert "Invalid Base64 data: Incorrect padding" == str(exc_info.value)
