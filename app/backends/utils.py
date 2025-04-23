import base64
import binascii
from typing import Dict, Any
import re

def parse_data_uri(uri: str) -> Dict[str, Any]:
    """Parses a data URI (e.g., data:image/jpeg;base64,...)"""
    match = re.match(r'data:image/(?P<format>jpeg|png|gif|webp);base64,(?P<data>.*)', uri)
    if not match:
        raise ValueError("Invalid or unsupported image data URI format. Must be data:image/[jpeg|png|gif|webp];base64,...")
    
    img_format = match.group('format')
    base64_data = match.group('data')
    
    try:
        decoded_bytes = base64.b64decode(base64_data)
    except binascii.Error as e:
        raise ValueError(f"Invalid Base64 data: {e}") from e

    return {"format": img_format, "data": decoded_bytes}