"""Pictures given by a file are read when the call is bound (contract/programs.md,
*Binding a call's inputs*): the request, the reply cache's key and the call's
record hold the picture, not its name."""
import base64
import shutil

import pytest
from lm15.types import ImagePart, image

import functai
from functai import InterfaceError, ai

PNG_A = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
PNG_B = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


@ai
def colour(picture: ImagePart) -> str:
    """The main colour."""
    ...


@ai
def colour_dict(picture: {"media": "image"}) -> str:  # noqa: F821
    """The main colour."""
    ...


def images_sent(request):
    return [p for m in request.messages for p in m.parts if p.type == "image"]


def test_a_picture_given_by_its_path_is_sent_as_its_bytes(tmp_path, fake):
    fake()
    (tmp_path / "cat.png").write_bytes(PNG_A)
    [sent] = images_sent(colour.render(image(path=tmp_path / "cat.png")))
    assert sent.data == base64.b64encode(PNG_A).decode() and sent.path is None and sent.media_type == "image/png"
    [d] = images_sent(colour_dict.render({"media_type": "image/png", "path": str(tmp_path / "cat.png")}))
    assert d.data == base64.b64encode(PNG_A).decode() and d.path is None


def test_a_file_changed_under_the_same_name_is_a_new_request(tmp_path, fake):
    r = fake("<result>\nyellow\n</result>", "<result>\nred\n</result>")
    functai.configure(cache_replies=tmp_path / "replies.sqlite")
    picture = tmp_path / "now.png"
    picture.write_bytes(PNG_A)
    assert colour(image(path=picture)) == "yellow"
    assert colour(image(path=picture)) == "yellow" and len(r.requests) == 1       # the same picture: from the cache
    shutil.copy(tmp_path / "now.png", tmp_path / "old.png")
    picture.write_bytes(PNG_B)
    assert colour(image(path=picture)) == "red" and len(r.requests) == 2          # another picture: asked


def test_the_calls_record_holds_the_picture_not_its_name(tmp_path, fake):
    fake("<result>\nyellow\n</result>")
    (tmp_path / "cat.png").write_bytes(PNG_A)
    with functai.configure(log_calls=tmp_path / "calls"):
        colour(image(path=tmp_path / "cat.png"))
    import json
    [line] = [x for f in (tmp_path / "calls").rglob("*.jsonl") for x in f.read_text().splitlines()]
    picture = json.loads(line)["inputs"]["picture"]
    assert picture["data"] == base64.b64encode(PNG_A).decode() and "path" not in picture


def test_a_picture_that_cannot_be_read_refuses_the_call_before_any_request(tmp_path, fake):
    r = fake()
    with pytest.raises(InterfaceError, match=r"input 'picture': the file .*gone\.png cannot be read"):
        colour(image(path=tmp_path / "gone.png"))
    assert not r.requests


def test_a_picture_addressed_by_url_is_left_for_the_provider(fake):
    fake()
    [sent] = images_sent(colour.render(image(url="https://example.com/cat.png")))
    assert sent.url == "https://example.com/cat.png" and sent.data is None


def test_a_list_of_pictures_is_sent_as_pictures_and_null_as_text(fake):
    """lmcc 0.8.7 (D-64): each picture of a list is its own image part; an optional one left null is the text null."""
    fake()

    @ai
    def compare(pictures: list[ImagePart], one: ImagePart | None = None) -> str:
        """Which is brighter?"""
        ...
    a, b = image(data=PNG_A, media_type="image/png"), image(data=PNG_B, media_type="image/png")
    request = compare.render([a, b])
    assert [p.data for p in images_sent(request)] == [base64.b64encode(PNG_A).decode(), base64.b64encode(PNG_B).decode()]
    assert all("iVBOR" not in (p.text or "") for m in request.messages for p in m.parts if p.type == "text")
    assert "null" in "".join(p.text or "" for m in request.messages for p in m.parts if p.type == "text")
    assert images_sent(compare.render([])) == []


def test_a_record_holding_a_picture_is_refused_not_sent_as_text(fake):
    """No layout writes a record's picture as a part, and none may write it as base64 text: refused at bind."""
    import dataclasses
    import lmcc
    fake()

    @dataclasses.dataclass
    class Listing:
        title: str
        photo: ImagePart

    @ai
    def describe(listing: Listing) -> str:
        """Describe the item."""
        ...
    with pytest.raises(lmcc.Refusal) as err:
        describe.render(Listing("mug", image(data=PNG_A, media_type="image/png")))
    assert err.value.code == "no-format"


def test_a_picture_function_has_a_version_of_its_own():
    """calls.md, Versions: a picture's sample is a picture by web address, so the version holds the instruction
    (it was one version for every picture function, the sample being refused)."""
    @ai
    def colour_too(picture: ImagePart) -> str:
        """Another instruction."""
        ...
    from functai.calllog import _sha, canonical
    refused = _sha(canonical({"request": "refused:value-invalid"}))
    assert colour.version != colour_too.version and refused not in (colour.version, colour_too.version)
