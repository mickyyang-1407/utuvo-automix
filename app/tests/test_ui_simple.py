"""R118: a simple page (the owner, 2026-11-05: "the UI should be simple and
clean, not complicated -- the complicated part is the code, not the user").

First screen: stems, style, start.  Everything else is folded into one "More
options".  After a mix: the result comes first, then a few group faders
("almost there?"), and every detail is folded away.  These tests read the
page itself, so moving a control back out of the fold fails them.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "web" / "index.html"
HTML = PAGE.read_text(encoding="utf-8")


class Tree(HTMLParser):
    """Each element id -> the ids of the elements it sits inside."""

    VOID = {"input", "br", "img", "meta", "link", "hr", "source", "wbr"}

    def __init__(self):
        super().__init__()
        self.stack: list[str | None] = []
        self.parents: dict[str, list[str]] = {}
        self.order: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        el_id = a.get("id")
        if el_id:
            self.parents[el_id] = [x for x in self.stack if x]
            self.order.append(el_id)
        if tag not in self.VOID:
            self.stack.append(el_id)

    def handle_endtag(self, tag):
        if tag not in self.VOID and self.stack:
            self.stack.pop()


def _tree():
    body = HTML[: HTML.index("<script>")]
    t = Tree()
    t.feed(body)
    return t


def test_the_first_screen_is_stems_style_and_start():
    t = _tree()
    for control in ("target", "export", "intensity", "reverb-amount", "sourceprofile", "balance",
                    "sec-ref", "refdrop", "blend", "sec-project", "project-name", "save-project"):
        assert "more-options" in t.parents[control], control
    for visible in ("drop", "styles", "go"):
        assert "more-options" not in t.parents[visible], visible
    # the fold is closed when the page opens
    assert re.search(r'<details class="card more" id="more-options">', HTML)


def test_after_a_mix_the_result_comes_first_then_the_faders_then_details():
    t = _tree()
    order = t.order
    assert order.index("result") < order.index("sec-stems") < order.index("sec-style")
    for inside in ("player", "dlmix", "tune", "faders", "tune-apply", "details"):
        assert "result" in t.parents[inside], inside
    assert order.index("player") < order.index("tune") < order.index("details")
    for folded in ("checks", "sections", "warnbox", "trackbox", "tracks", "dlplan", "dlreport",
                   "abnote", "metrics-more"):
        assert "details" in t.parents[folded], folded
    assert "details" not in t.parents["tune"]


def test_every_new_line_of_text_has_an_english_version():
    for zh in ("更多選項", "響度、殘響量、匯出分軌、參考曲、專案存檔——不改也可以", "參考曲（選用）", "差一點點？",
               "往右＝大聲一點、往左＝小聲一點，再按「套用並重新混音」。自動混音的其他決定都不變。",
               "全部歸零", "套用並重新混音", "細節：檢查結果、每一軌做了什麼、報告"):
        assert re.search(r'^\s*"' + re.escape(zh) + r'":\s*"[^"]+"', HTML, re.M), zh


def test_the_faders_are_the_engines_groups_and_go_with_every_mix():
    groups = re.findall(r'^\s*\["(\w+)", \(\) => L\(', HTML[HTML.index("const FADER_GROUPS"):], re.M)[:5]
    assert groups == ["vocal", "drum", "bass", "music", "fx"]
    from automix import plan as planner
    assert tuple(groups) == planner.USER_TRIM_GROUPS
    body = HTML[HTML.index("function runBody()"):HTML.index("function runBody()") + 900]
    assert "group_trims: faderValues()" in body
    # the faders on screen are the ones the published render applied
    block = HTML[HTML.index("function renderFaders(r)"):HTML.index("function renderFaders(r)") + 1500]
    assert "r.group_trims" in block and "Array.isArray(r.groups)" in block
    # one name for the fx group, and "other" stems belong to the instruments
    assert 'L("效果與空間", "FX & space")' in HTML
    assert 'fx: "fx", other: "music"}' in HTML
