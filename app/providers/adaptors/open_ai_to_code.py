from typing import Any
import base64
import re

from functools import singledispatch
from ..core.schema import ChatRequest, Message, TextPart, ImagePart
from app.schema.open_ai import  (
    ChatCompletionRequest,
    TextContentPart as OA_Text,
    ImageContentPart as OA_Image,
)

@singledispatch
def _part_to_ir(part) -> Any:
    raise TypeError(f"No converter for {type(part)}")


@_part_to_ir.register
def _(part: OA_Text) -> TextPart:
    return TextPart(text=part.text)


@_part_to_ir.register
def _(part: OA_Image) -> ImagePart:
    b64 = re.sub("^data:[^,]+,", "", part.image_url.url)
    return ImagePart(bytes=base64.b64decode(b64))
