"""Tests for the manuscript copyedit tooling (scripts/tex_*.py, copyedit.py).

The whole pipeline rests on one invariant: **de-TeXing preserves offsets**.
`tex_plain.to_prose` overwrites markup in place with same-length runs of spaces
or same-width stand-in words, so a checker's character offset maps back to the
real source line with no translation step. If that ever breaks, every grammar
finding silently points at the wrong line, and nothing in the report says so.
`test_length_preserved_on_every_manuscript_file` is the guard.

The other invariant is scope: `resolve_inputs` follows \\input from 00_main.tex,
so files sitting in the paper directory but not in the document (scraps.tex,
TADA_1pager.tex) must never produce findings.

Coverage:

- to_prose / strip_comments preserve length, line count, and newline positions
- accents survive as letters, so "Les Mis\\'{e}rables" is not read as "Mis"
- refs, citations and math become stand-in words, and is_standin identifies them
- verbatim files (prompts/, "DO NOT EDIT BY HAND") are recognized
- resolve_inputs reaches the real document and excludes scraps.tex
- undefined refs, duplicate labels and literal `??` are detected on a fixture
- the live manuscript has no blocking reference defect
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

tex_plain = pytest.importorskip("tex_plain")
tex_integrity = pytest.importorskip("tex_integrity")
tex_grammar = pytest.importorskip("tex_grammar")

PAPER = tex_integrity.PAPER
pytestmark = pytest.mark.skipif(
    not PAPER.is_dir(), reason="manuscript submodule not checked out"
)


def manuscript_files() -> list[Path]:
    return tex_integrity.resolve_inputs(PAPER)


class TestOffsetPreservation:
    """The invariant every downstream offset depends on."""

    def test_length_preserved_on_every_manuscript_file(self):
        for f in manuscript_files():
            raw = f.read_text(encoding="utf-8", errors="replace")
            prose = tex_plain.to_prose(raw)
            assert len(prose) == len(raw), f"{f.name}: {len(prose)} != {len(raw)}"

    def test_newlines_stay_put(self):
        # Line numbers are derived by counting newlines, so a newline that
        # moves (or is blanked) shifts every finding after it.
        for f in manuscript_files():
            raw = f.read_text(encoding="utf-8", errors="replace")
            prose = tex_plain.to_prose(raw)
            assert [i for i, c in enumerate(raw) if c == "\n"] == \
                   [i for i, c in enumerate(prose) if c == "\n"], f.name

    def test_strip_comments_preserves_length(self):
        src = "text before % a comment\nnext line with 50\\% escaped\n"
        out = tex_plain.strip_comments(src)
        assert len(out) == len(src)
        assert "a comment" not in out
        assert "50\\%" in out  # an escaped percent is not a comment


class TestDeTexing:
    def test_accented_name_stays_one_word(self):
        # Blanking \'{e} would split the title into the word "Mis", which
        # codespell then reports as a misspelling of "miss".
        prose = tex_plain.to_prose(r"\textit{Les Mis\'{e}rables} (Hugo)")
        assert "Miserables" in prose.replace(" ", "") or "Mise" in prose
        assert "Mis " not in prose

    def test_crossref_becomes_its_rendered_noun(self):
        assert "Figure" in tex_plain.to_prose(r"see \autoref{fig:x} here")
        assert "Table" in tex_plain.to_prose(r"see \autoref{tab:x} here")

    def test_citet_becomes_a_noun_and_citep_vanishes(self):
        assert "Author" in tex_plain.to_prose(r"\citet{doe} shows")
        assert "Author" not in tex_plain.to_prose(r"holds~\citep{doe}.")

    def test_math_becomes_a_noun_placeholder(self):
        prose = tex_plain.to_prose(r"the reward $R_{\text{direct}}$ is fixed")
        assert "$" not in prose
        assert "x" in prose

    def test_latex_quotes_become_real_quotes(self):
        # Otherwise LanguageTool reports every quoted phrase as unpaired.
        prose = tex_plain.to_prose(r"the ``ought'' of a norm")
        assert "“" in prose and "”" in prose

    def test_no_prose_environments_are_blanked(self):
        prose = tex_plain.to_prose(
            "before\n\\begin{tabular}{ll}\na & b \\\\\n\\end{tabular}\nafter")
        assert "before" in prose and "after" in prose
        assert "a & b" not in prose

    def test_caption_text_survives(self):
        # Captions are prose and must be checked; tabular bodies must not.
        prose = tex_plain.to_prose(r"\caption{Which norm each flow retrieves.}")
        assert "Which norm each flow retrieves." in prose

    def test_is_standin_distinguishes_placeholder_from_authored_text(self):
        raw = r"see \autoref{fig:x} and x"
        prose = tex_plain.to_prose(raw)
        assert tex_plain.is_standin(raw, prose.index("Figure"))
        assert not tex_plain.is_standin(raw, raw.index("see"))


class TestScope:
    def test_resolve_inputs_reaches_the_document(self):
        names = {f.name for f in manuscript_files()}
        assert "00_main.tex" in names
        assert "04_results.tex" in names

    def test_files_outside_the_document_are_excluded(self):
        # Present in the directory, not \input by 00_main.tex.
        names = {f.name for f in manuscript_files()}
        assert "scraps.tex" not in names
        assert "TADA_1pager.tex" not in names

    def test_verbatim_material_is_recognized(self):
        assert tex_grammar.is_verbatim(Path("prompts/x.tex"), "anything")
        assert tex_grammar.is_verbatim(
            Path("a.tex"), "% Auto-generated. DO NOT EDIT BY HAND.\n")
        assert not tex_grammar.is_verbatim(Path("04_results.tex"), "We show that")


class TestReferenceIntegrity:
    """The `??` detector, against a fixture with known defects."""

    @staticmethod
    def _analyze(tmp_path: Path, body: str):
        (tmp_path / "00_main.tex").write_text(body, encoding="utf-8")
        files = tex_integrity.resolve_inputs(tmp_path)
        return {(f.code, f.line) for f in tex_integrity.analyze(tmp_path, files)}, \
               [f for f in tex_integrity.analyze(tmp_path, files)]

    def test_undefined_ref_is_an_error(self, tmp_path):
        _, findings = self._analyze(tmp_path, "See \\autoref{fig:missing}.\n")
        undefined = [f for f in findings if f.code == "UNDEFINED_REF"]
        assert len(undefined) == 1
        assert undefined[0].severity == "error"
        assert "fig:missing" in undefined[0].message

    def test_defined_ref_is_clean(self, tmp_path):
        _, findings = self._analyze(
            tmp_path,
            "\\begin{figure}\\caption{c}\\label{fig:a}\\end{figure}\n"
            "See \\autoref{fig:a}.\n")
        assert not [f for f in findings if f.code == "UNDEFINED_REF"]

    def test_cref_list_checks_every_key(self, tmp_path):
        _, findings = self._analyze(
            tmp_path,
            "\\begin{figure}\\caption{c}\\label{fig:a}\\end{figure}\n"
            "See \\cref{fig:a,fig:ghost}.\n")
        undefined = [f for f in findings if f.code == "UNDEFINED_REF"]
        assert [f.message for f in undefined] and "fig:ghost" in undefined[0].message

    def test_duplicate_label_is_an_error(self, tmp_path):
        _, findings = self._analyze(
            tmp_path, "\\label{sec:a}\ntext\n\\label{sec:a}\n\\autoref{sec:a}\n")
        dupes = [f for f in findings if f.code == "DUPLICATE_LABEL"]
        assert len(dupes) == 1 and dupes[0].severity == "error"

    def test_literal_question_marks_are_flagged(self, tmp_path):
        _, findings = self._analyze(tmp_path, "As shown in Figure ??.\n")
        assert [f for f in findings if f.code == "LITERAL_QQ"]

    def test_label_before_caption_is_flagged(self, tmp_path):
        # A label above the caption captures the *previous* float's number.
        _, findings = self._analyze(
            tmp_path,
            "\\begin{figure}\n\\label{fig:a}\n\\caption{c}\n\\end{figure}\n"
            "\\autoref{fig:a}\n")
        assert [f for f in findings if f.code == "LABEL_BEFORE_CAPTION"]

    def test_layout_float_without_caption_is_not_nagged(self, tmp_path):
        # A float with neither caption nor label is being used for layout.
        _, findings = self._analyze(
            tmp_path, "\\begin{table}\\begin{tabular}{l}a\\\\\\end{tabular}\\end{table}\n")
        assert not [f for f in findings if f.code == "FLOAT_NO_LABEL"]

    def test_section_sign_satisfies_the_bare_ref_rule(self, tmp_path):
        # "\S\ref{sec:x}" sets "§4" and needs no extra noun.
        _, findings = self._analyze(
            tmp_path, "\\label{sec:x}\nas in \\S\\ref{sec:x}.\n")
        assert not [f for f in findings if f.code == "BARE_REF"]


class TestLiveManuscript:
    def test_no_blocking_reference_defect(self):
        """The camera-ready must not print `??` anywhere.

        Static only: no compile, so this is safe to run anywhere. The build and
        rendered-PDF checks live behind `scripts/copyedit.py --build`.
        """
        findings = tex_integrity.analyze(PAPER, manuscript_files())
        blocking = [f for f in findings if f.severity == "error"]
        assert not blocking, "\n".join(
            f"{f.file}:{f.line} {f.code} {f.message}" for f in blocking)
