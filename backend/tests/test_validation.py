import pytest

from app.config import settings
from app.pipeline.validate import check_photo
from tests.conftest import make_photo


def codes(data: bytes) -> list[str]:
    return [i.code for i in check_photo(data, settings).issues]


def test_good_leaf_photo_passes():
    checked = check_photo(make_photo("leaf"), settings)
    assert checked.ok and checked.image is not None


@pytest.mark.parametrize(
    "kind,code",
    [("blurry", "too_blurry"), ("dark", "too_dark"), ("bright", "too_bright"), ("not_plant", "not_a_plant")],
)
def test_bad_photos_get_the_right_reason(kind, code):
    assert code in codes(make_photo(kind))


def test_empty_file():
    assert codes(b"") == ["empty_file"]


def test_not_an_image():
    assert codes(b"this is a text file, not a photo") == ["unreadable_image"]


def test_truncated_image():
    assert codes(make_photo("leaf")[:500]) == ["unreadable_image"]


def test_too_small():
    assert codes(make_photo("leaf", size=100)) == ["too_small"]


def test_unsupported_format():
    assert codes(make_photo("leaf", fmt="BMP")) == ["unsupported_type"]


def test_too_large():
    assert codes(b"x" * (settings.max_upload_bytes + 1)) == ["file_too_large"]


def test_every_issue_has_a_message_and_a_tip():
    for issue in check_photo(make_photo("dark"), settings).issues:
        assert issue.message and issue.tip
