from fastapi import HTTPException
import pytest

from app.api.upload_validation import validate_upload


def test_upload_validation_accepts_supported_file():
    assert validate_upload("guide.pdf", b"content") == "guide.pdf"


@pytest.mark.parametrize("filename", ["../guide.pdf", "", "guide.exe", "guide.pdf/"])
def test_upload_validation_rejects_unsafe_or_unsupported_name(filename):
    with pytest.raises(HTTPException):
        validate_upload(filename, b"content")


def test_upload_validation_rejects_empty_file():
    with pytest.raises(HTTPException):
        validate_upload("guide.txt", b"")


def test_upload_validation_rejects_oversized_file():
    with pytest.raises(HTTPException):
        validate_upload("guide.txt", b"12345", max_bytes=4)
