from typing import cast
import pytest
from app.providers.core.chat_schema import ImagePart, FilePart
from app.providers.open_ai.adapter_to_core import openai_chat_request_to_core
from app.providers.exceptions import InvalidBase64DataError, InvalidImageURLError

def test_request_to_core(core_chat_request, openai_chat_request):
    converted = openai_chat_request_to_core(openai_chat_request)
    assert converted == core_chat_request


def test_full_request_to_core(core_full_chat_request, openai_full_chat_request):
    converted = openai_chat_request_to_core(openai_full_chat_request)
    assert converted == core_full_chat_request


@pytest.mark.parametrize(
    ("open_ai_example_image", "expected_exc", "msg_part"),
    [
        pytest.param("abci23", InvalidImageURLError, "Invalid or unsupported image data URI format."),
        pytest.param("data:image/xls", InvalidImageURLError, "Invalid or unsupported image data URI format."),
        pytest.param("data:image/jpeg;base64,abcde", InvalidBase64DataError, "Invalid Base64 data:"),
        pytest.param("data:image/jpeg;base64,abcd=", None, [b'i\xb7\x1d', "jpeg"]),
    ], 
    indirect=("open_ai_example_image",))
def test_image_request_to_core(open_ai_example_image, expected_exc, msg_part):
    if expected_exc is not None:
        with pytest.raises(expected_exc) as exc_info:
            openai_chat_request_to_core(open_ai_example_image)
        assert msg_part in str(exc_info.value)
    else:
        data_bytes, format = msg_part
        converted = openai_chat_request_to_core(open_ai_example_image)
        assert len(converted.messages) == 1

        image_part:ImagePart = cast(ImagePart, converted.messages[0].content[0])
        assert image_part.file_type == format
        assert image_part.bytes_ == data_bytes


@pytest.mark.parametrize(
    ("open_ai_example_file", "msg_part"),
    [
        pytest.param("SGVsbG8=", b'Hello'),
    ], 
    indirect=("open_ai_example_file", ))
def test_convert_open_ai_request_with_file(open_ai_example_file, msg_part):
    ''' It should produce the correct bytes for good document formats.
        Pydantic will raise on bad format before it hits our code'''
    converted = openai_chat_request_to_core(open_ai_example_file)
    assert len(converted.messages) == 1
    document_block:FilePart = cast(FilePart, converted.messages[0].content[0])
    assert document_block.bytes_ == msg_part 
