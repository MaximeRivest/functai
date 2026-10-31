"""Files as inputs (functai/files.py): an Attachments result as an AI function's
input. What the model is shown, what the log and the cache hold, replays, served
calls, files that failed, and when the binding is made."""
import base64
import json
import subprocess
import sys
import textwrap

import pytest

attachments = pytest.importorskip("attachments")
from attachments import Artifacts, att  # noqa: E402
from attachments.types import error_artifact, make_artifact  # noqa: E402

import functai  # noqa: E402
from functai import InterfaceError, ai  # noqa: E402

PNG_1 = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
PNG_2 = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def report() -> Artifacts:
    """A two-page document: each page's text and its page image."""
    text = "Revenue grew 12%.\n\nCosts fell 3%."
    return Artifacts([make_artifact(
        text=text,
        images=[{"name": "report.pdf-page-1.png", "mimetype": "image/png", "bytes": PNG_1, "page": 1},
                {"name": "report.pdf-page-2.png", "mimetype": "image/png", "bytes": PNG_2, "page": 2}],
        meta={"source": "report.pdf", "kind": "pdf",
              "segments": [{"kind": "page", "label": "page 1", "start": 0, "end": 17, "page": 1},
                           {"kind": "page", "label": "page 2", "start": 19, "end": 33, "page": 2}]})])


@ai
def summary(doc: Artifacts) -> str:
    """The document in one sentence."""
    ...


def parts_of(request):
    return [p for m in request.messages for p in m.parts]


def test_the_model_sees_each_pages_text_then_its_picture(fake):
    fake()
    parts = parts_of(summary.render(report()))
    kinds = [p.type for p in parts]
    assert kinds.count("image") == 2 and "text" in kinds
    texts = [i for i, p in enumerate(parts) if p.type == "text" and "Revenue" in p.text]
    images = [i for i, p in enumerate(parts) if p.type == "image"]
    costs = [i for i, p in enumerate(parts) if p.type == "text" and "Costs" in p.text]
    assert texts[0] < images[0] < costs[0] < images[1]                        # page by page
    assert parts[images[0]].data == base64.b64encode(PNG_1).decode()          # an image part, not base64 text
    assert not any(base64.b64encode(PNG_1).decode()[:20] in p.text for p in parts if p.type == "text")
    assert any("report.pdf" in p.text for p in parts if p.type == "text")     # the file's name, as att() prints it


@pytest.mark.parametrize("layout", ["xml", "json", "chat"])
def test_every_layout_sends_the_pictures_as_pictures(fake, layout):
    fake()
    assert [p.type for p in parts_of(summary.using(adapter=layout).render(report()))].count("image") == 2


def test_a_file_that_could_not_be_read_refuses_the_call_before_any_request(fake):
    r = fake("<result>\nok\n</result>")
    missing = Artifacts([error_artifact("missing.pdf", "unpack-error", "no such file")])
    with pytest.raises(InterfaceError) as caught:
        summary(missing)
    assert caught.value.code == "interface-input" and caught.value.field == "doc"
    assert "missing.pdf: unpack-error — no such file" in str(caught.value) and not r.requests
    with pytest.raises(InterfaceError, match="could not be read"):
        summary(report() + missing)                                             # one bad file in a folder too


def test_a_failed_file_in_its_wire_form_is_refused_too(fake):
    """A served call or a replay gives the wire form: a file that failed is refused all the same, and so is a value
    Attachments' from_wire refuses."""
    r = fake("<result>\nok\n</result>")
    failed = Artifacts([error_artifact("missing.pdf", "unpack-error", "no such file")]).to_wire()
    for bad in (failed, [{"text": 3}]):
        with pytest.raises(InterfaceError) as caught:
            summary(bad)
        assert caught.value.code == "interface-input" and caught.value.field == "doc"
    assert not r.requests


@pytest.mark.filterwarnings("ignore:.*defined by exec")
@pytest.mark.parametrize("annotation", ["Optional[Artifacts]", "list[Artifacts]", "dict[str, Artifacts]", "Holder"])
def test_files_are_a_whole_input(annotation):
    """Inside another type a layout would write the artifacts as JSON text (their pictures as base64): refused when
    the function is defined, saying how to give no files."""
    import dataclasses
    from typing import Optional

    @dataclasses.dataclass
    class Holder:
        doc: Artifacts
    space = {"Artifacts": Artifacts, "Optional": Optional, "Holder": Holder, "ai": ai}
    with pytest.raises(InterfaceError) as caught:
        exec(f"@ai\ndef read(doc: {annotation}) -> str:\n    'Read it.'\n    ...\n", space)
    assert caught.value.code == "interface-malformed" and "att([])" in str(caught.value)
    with pytest.raises(InterfaceError, match="an input"):
        exec("@ai\ndef make(topic: str) -> Artifacts:\n    'Write it.'\n    ...\n", space)


def test_no_files_is_no_content(fake):
    fake()
    texts = [p.text for p in parts_of(summary.render(Artifacts([]))) if p.type == "text"]
    assert any("(no content)" in t for t in texts)


def test_the_log_holds_the_wire_form_and_the_cache_answers_the_same_document(fake, tmp_path):
    r = fake("<result>\nA report.\n</result>")
    functai.configure(cache_replies=tmp_path / "replies.sqlite", log_calls=tmp_path / "calls")
    assert summary(report()) == "A report." and summary(report()) == "A report." and len(r.requests) == 1
    lines = [json.loads(x) for f in (tmp_path / "calls").rglob("*.jsonl") for x in f.read_text().splitlines()]
    logged = lines[0]["inputs"]["doc"]
    assert logged == report().to_wire()                                        # Attachments' own wire form
    assert Artifacts.from_wire(logged) == report()


def test_its_wire_form_is_accepted_as_the_input(fake):
    fake()
    from_wire = parts_of(summary.render(report().to_wire()))                  # a served call, another language
    assert [p.type for p in from_wire] == [p.type for p in parts_of(summary.render(report()))]


def test_a_conversation_shows_an_earlier_document_again_as_pictures(fake, tmp_path):
    r = fake("<result>\nA report.\n</result>", "<result>\nStill a report.\n</result>")
    chat = summary.conversation("c1", store=tmp_path / "turns")
    chat(report())
    chat2 = summary.conversation("c1", store=tmp_path / "turns")             # read back from the folder: JSON
    chat2(Artifacts([make_artifact(text="A note.", meta={"source": "note.txt"})]))
    second = parts_of(r.requests[1])
    assert [p.type for p in second].count("image") == 2                       # the first turn's pages, rebuilt
    assert not any("bytes_b64" in p.text for p in second if p.type == "text")


def test_binding_is_made_when_attachments_is_imported_and_never_imports_it(tmp_path):
    code = textwrap.dedent("""
        import sys, functai
        assert "attachments" not in sys.modules, "functai imported attachments"
        import lmcc, attachments
        bound = [h for h in lmcc.default_registry.host_types if h.host_type is attachments.Artifacts]
        print(len(bound))
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "1"
    first = textwrap.dedent("""
        import attachments, functai, lmcc
        print(len([h for h in lmcc.default_registry.host_types if h.host_type is attachments.Artifacts]))
    """)
    assert subprocess.run([sys.executable, "-c", first], capture_output=True, text=True, check=True).stdout.strip() == "1"


def test_a_saved_program_reading_files_loads_where_attachments_was_never_imported(tmp_path, fake):
    fake()
    (tmp_path / "prog.py").write_text(textwrap.dedent('''
        from attachments import Artifacts
        from functai import ai

        @ai
        def summary(doc: Artifacts) -> str:
            """The document in one sentence."""
    '''))
    sys.path.insert(0, str(tmp_path))
    try:
        import prog
        functai.save(prog.summary, tmp_path / "saved")
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("prog", None)
    wire = json.dumps(report().to_wire())
    code = textwrap.dedent(f"""
        import json, sys, functai
        functai.configure(lm="gpt-4.1-mini", api_key="sk-test")
        loaded = functai.load({str(tmp_path / "saved")!r}, trust=True)
        request = loaded.render(json.loads({wire!r}))
        print([p.type for m in request.messages for p in m.parts].count("image"), "attachments" in sys.modules)
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split() == ["2", "True"]


def test_att_reads_a_real_file(tmp_path, fake):
    fake()
    (tmp_path / "note.md").write_text("# Plan\n\nShip on Friday.")
    parts = parts_of(summary.render(att(str(tmp_path / "note.md"))))
    assert any("Ship on Friday." in p.text for p in parts if p.type == "text")
