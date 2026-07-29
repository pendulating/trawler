"""Tests for the m-series deontic-distance scorer (`T-VIGNETTE` reward).

Covers the task-vignettes.md contract (re-anchored scale, 2026-07-28):
  * AXIS map + the full 25-force-pair item_score table (incl. unknown→permitted);
  * the re-anchored [0,1] item scale — exact 1.0 / same-side 0.4 / hedge 0.15 /
    cross-side 0.0 — which killed the m1 hedge sanctuary (hedge was mid-range
    0.5 under the old rescaled distance and the policy measurably drifted
    into it: hedge_frac 0.217→0.311);
  * battery_score = plain mean (the [−1,1]→[0,1] rescale is gone);
  * cite Jaccard overlap, empty-side → 0 — DIAGNOSTIC only, r_vig = battery;
  * parse_battery_completion id-alignment + json_repair on truncated JSON;
  * score_battery aggregate keys (battery / cite / r_vig / hedge_frac /
    antithesis_frac / parsed_frac).
"""

import json

import pytest

from dagspaces.grpo_training.stages import deontic_distance as dd
from dagspaces.grpo_training.stages.deontic import FORCE_TO_GOLD


# --------------------------------------------------------------------------- #
# AXIS + item_score: the full 25-pair table                                   #
# --------------------------------------------------------------------------- #

_FORCES = ["obligatory", "recommended", "permitted", "discouraged", "prohibited"]

# Hardcoded re-anchored scale (2026-07-28): exact 1.0 / same-side commit 0.4 /
# hedge-vs-decisive 0.15 / decisive-vs-permitted-gold 0.4 (mild) or 0.15
# (extreme) / cross-side 0.0. Rows = model force, columns = gold force.
_ITEM_TABLE = {
    "obligatory":  {"obligatory": 1.0, "recommended": 0.4, "permitted": 0.15, "discouraged": 0.0, "prohibited": 0.0},
    "recommended": {"obligatory": 0.4, "recommended": 1.0, "permitted": 0.4, "discouraged": 0.0, "prohibited": 0.0},
    "permitted":   {"obligatory": 0.15, "recommended": 0.15, "permitted": 1.0, "discouraged": 0.15, "prohibited": 0.15},
    "discouraged": {"obligatory": 0.0, "recommended": 0.0, "permitted": 0.4, "discouraged": 1.0, "prohibited": 0.4},
    "prohibited":  {"obligatory": 0.0, "recommended": 0.0, "permitted": 0.15, "discouraged": 0.4, "prohibited": 1.0},
}


class TestAxis:
    def test_axis_values(self):
        assert dd.AXIS == {
            "obligatory": 2,
            "recommended": 1,
            "permitted": 0,
            "discouraged": -1,
            "prohibited": -2,
        }

    def test_axis_is_the_force_to_gold_vocabulary_plus_permitted(self):
        # Every decisive (gold-bearing) force sits on the axis; permitted is the
        # extra neutral point.
        assert set(FORCE_TO_GOLD).issubset(set(dd.AXIS))
        assert set(dd.AXIS) - set(FORCE_TO_GOLD) == {"permitted"}

    def test_axis_sign_agrees_with_force_to_gold_polarity(self):
        # AXIS is a parallel literal to FORCE_TO_GOLD (not derived); pin their
        # POLARITY agreement so a future sign flip in one map is caught
        # (2026-07-24 review J2 — membership alone did not guard this).
        for force, gold in FORCE_TO_GOLD.items():
            axis = dd.AXIS[force]
            if gold == "yes":
                assert axis > 0, f"{force}: gold=yes but axis={axis}"
            elif gold == "no":
                assert axis < 0, f"{force}: gold=no but axis={axis}"


class TestItemScoreTable:
    @pytest.mark.parametrize("model", _FORCES)
    @pytest.mark.parametrize("gold", _FORCES)
    def test_all_25_pairs(self, model, gold):
        assert dd.item_score(model, gold) == pytest.approx(_ITEM_TABLE[model][gold])

    def test_range_bounds(self):
        assert dd.item_score("obligatory", "prohibited") == 0.0   # min (cross)
        assert dd.item_score("prohibited", "prohibited") == 1.0   # max

    def test_named_cells_from_wiki_table(self):
        # gold = obligatory: exact / same-side / hedge / cross / cross.
        assert dd.item_score("obligatory", "obligatory") == 1.0
        assert dd.item_score("recommended", "obligatory") == dd.ITEM_SCORE_SAME_SIDE
        assert dd.item_score("permitted", "obligatory") == dd.ITEM_SCORE_HEDGE
        assert dd.item_score("discouraged", "obligatory") == dd.ITEM_SCORE_CROSS
        assert dd.item_score("prohibited", "obligatory") == dd.ITEM_SCORE_CROSS

    def test_ordering_invariant_hedge_below_same_side_above_cross(self):
        # THE design ordering the re-anchor exists to enforce: committing to
        # the right side always beats hedging, hedging always beats the
        # antithesis. Under the m1 scale hedge (rescaled 0.5) beat wrong-side
        # commits AND capped downside — the policy learned to hedge.
        assert (dd.ITEM_SCORE_EXACT > dd.ITEM_SCORE_SAME_SIDE
                > dd.ITEM_SCORE_HEDGE > dd.ITEM_SCORE_CROSS)


class TestItemScoreHedgeFallback:
    def test_none_is_permitted(self):
        assert dd.item_score(None, "obligatory") == _ITEM_TABLE["permitted"]["obligatory"]
        assert dd.item_score(None, "prohibited") == _ITEM_TABLE["permitted"]["prohibited"]

    def test_unknown_force_is_permitted(self):
        # "unknown → permitted" — a non-force string collapses to the hedge point.
        assert dd.item_score("banana", "obligatory") == dd.item_score("permitted", "obligatory")
        assert dd.item_score("", "prohibited") == dd.item_score("permitted", "prohibited")
        assert dd.item_score(42, "recommended") == dd.item_score("permitted", "recommended")

    def test_case_and_whitespace_insensitive(self):
        assert dd.item_score("  Prohibited ", "Obligatory") == 0.0

    def test_unknown_gold_collapses_to_permitted(self):
        assert dd.item_score("obligatory", "nonsense") == dd.item_score("obligatory", "permitted")


# --------------------------------------------------------------------------- #
# battery_score: plain mean + empty raises                                     #
# --------------------------------------------------------------------------- #

class TestBatteryScore:
    def test_plain_mean_no_rescale(self):
        # Items are already in [0, 1]; the m1-era [−1,1]→[0,1] rescale is gone
        # (it parked hedge at 0.5 of full range).
        assert dd.battery_score([1.0]) == 1.0
        assert dd.battery_score([dd.ITEM_SCORE_HEDGE]) == dd.ITEM_SCORE_HEDGE
        assert dd.battery_score([0.0]) == 0.0
        assert dd.battery_score([1.0, 0.0]) == 0.5
        assert dd.battery_score([1.0, 0.4, 0.15]) == pytest.approx(1.55 / 3)

    def test_empty_raises_valueerror(self):
        with pytest.raises(ValueError):
            dd.battery_score([])


# --------------------------------------------------------------------------- #
# cite_score: Jaccard                                                          #
# --------------------------------------------------------------------------- #

class TestCiteScore:
    def test_identical_is_one(self):
        assert dd.cite_score("the patriarch must stay silent", "the patriarch must stay silent") == 1.0

    def test_disjoint_is_zero(self):
        assert dd.cite_score("alpha beta gamma", "delta epsilon zeta") == 0.0

    def test_partial_overlap_jaccard(self):
        # A = {the, patriarch, shall, not, disclose}  (5)
        # B = {patriarch, disclose, secrets}          (3)
        # inter = {patriarch, disclose} = 2 ; union = 6 → 1/3
        s = dd.cite_score("the patriarch shall not disclose", "patriarch disclose secrets")
        assert s == pytest.approx(2 / 6)

    def test_empty_either_side_zero(self):
        assert dd.cite_score("", "the norm") == 0.0
        assert dd.cite_score("the norm", "") == 0.0
        assert dd.cite_score("", "") == 0.0

    def test_tokenization_lowercases_and_strips_punct(self):
        # "Patriarch, disclose!" tokenizes to {patriarch, disclose}; set-based so
        # duplicate words don't change the score.
        assert dd.cite_score("Patriarch, disclose! disclose", "patriarch disclose") == 1.0


# --------------------------------------------------------------------------- #
# parse_battery_completion: alignment + repair                                 #
# --------------------------------------------------------------------------- #

def _items_json(*items):
    return json.dumps({"items": list(items)})


class TestParseBatteryCompletion:
    def test_full_aligned(self):
        text = _items_json(
            {"id": 1, "force": "obligatory", "reasoning": "r", "governing_norm": "n1"},
            {"id": 2, "force": "prohibited", "reasoning": "r", "governing_norm": "n2"},
            {"id": 3, "force": "permitted", "reasoning": "r", "governing_norm": "n3"},
        )
        parsed = dd.parse_battery_completion(text, 3)
        assert [p["force"] for p in parsed] == ["obligatory", "prohibited", "permitted"]

    def test_missing_id_is_none(self):
        text = _items_json(
            {"id": 1, "force": "obligatory"},
            {"id": 3, "force": "prohibited"},
        )
        parsed = dd.parse_battery_completion(text, 3)
        assert parsed[0]["force"] == "obligatory"
        assert parsed[1] is None
        assert parsed[2]["force"] == "prohibited"

    def test_out_of_range_and_string_ids(self):
        text = _items_json(
            {"id": "2", "force": "recommended"},  # string id coerces
            {"id": 9, "force": "obligatory"},     # out of range → dropped
            {"id": "x", "force": "prohibited"},   # non-int → dropped
        )
        parsed = dd.parse_battery_completion(text, 3)
        assert parsed[0] is None
        assert parsed[1]["force"] == "recommended"
        assert parsed[2] is None

    def test_duplicate_id_first_wins(self):
        text = _items_json(
            {"id": 1, "force": "obligatory"},
            {"id": 1, "force": "prohibited"},
        )
        parsed = dd.parse_battery_completion(text, 2)
        assert parsed[0]["force"] == "obligatory"
        assert parsed[1] is None

    def test_unparseable_all_none(self):
        parsed = dd.parse_battery_completion("this is not json at all", 4)
        assert parsed == [None, None, None, None]

    def test_empty_string_all_none(self):
        assert dd.parse_battery_completion("", 3) == [None, None, None]

    def test_truncated_json_repaired(self):
        # A max_tokens truncation mid-list: no closing brackets. json_repair must
        # salvage the complete leading items (the 2026-07-23 build lesson).
        truncated = (
            '{"items": ['
            '{"id": 1, "force": "obligatory", "reasoning": "because", "governing_norm": "n1"}, '
            '{"id": 2, "force": "prohibited", "reasoning": "the rule forbids it", "governing_norm": "n2"}, '
            '{"id": 3, "force": "recomm'
        )
        parsed = dd.parse_battery_completion(truncated, 4)
        assert parsed[0]["force"] == "obligatory"
        assert parsed[1]["force"] == "prohibited"
        # item 4 never emitted → None; item 3 truncated (repair may salvage a
        # partial dict or drop it, but must not raise and must keep 1 & 2).
        assert parsed[3] is None

    def test_wrapped_in_prose_and_fences(self):
        text = (
            "Sure! Here is my answer:\n```json\n"
            + _items_json({"id": 1, "force": "discouraged"})
            + "\n```\nHope that helps."
        )
        parsed = dd.parse_battery_completion(text, 1)
        assert parsed[0]["force"] == "discouraged"

    def test_k_zero_returns_empty(self):
        assert dd.parse_battery_completion(_items_json({"id": 1, "force": "x"}), 0) == []


# --------------------------------------------------------------------------- #
# score_battery: aggregate keys                                                #
# --------------------------------------------------------------------------- #

def _gold(force, articulation=""):
    return {"gold_force": force, "articulation": articulation}


class TestScoreBattery:
    def test_all_exact_full_credit(self):
        gold = [_gold("obligatory", "share the record"), _gold("prohibited", "never disclose")]
        parsed = [
            {"force": "obligatory", "governing_norm": "share the record"},
            {"force": "prohibited", "governing_norm": "never disclose"},
        ]
        out = dd.score_battery(parsed, gold)
        assert out["battery"] == 1.0
        assert out["cite"] == 1.0
        assert out["r_vig"] == 1.0  # cite is diagnostic-only (2026-07-28)
        assert out["hedge_frac"] == 0.0
        assert out["antithesis_frac"] == 0.0
        assert out["parsed_frac"] == 1.0

    def test_all_hedge(self):
        gold = [_gold("obligatory"), _gold("prohibited")]
        parsed = [{"force": "permitted"}, {"force": "permitted"}]
        out = dd.score_battery(parsed, gold)
        # The m1 sanctuary is closed: hedging everything scores 0.15, not 0.5.
        assert out["battery"] == pytest.approx(dd.ITEM_SCORE_HEDGE)
        assert out["hedge_frac"] == 1.0
        assert out["antithesis_frac"] == 0.0   # permitted has axis 0, product 0
        assert out["parsed_frac"] == 1.0

    def test_hedging_never_beats_a_same_side_commit(self):
        gold = [_gold("obligatory")]
        hedge = dd.score_battery([{"force": "permitted"}], gold)
        commit = dd.score_battery([{"force": "recommended"}], gold)
        assert commit["battery"] > hedge["battery"]

    def test_full_antithesis(self):
        gold = [_gold("obligatory"), _gold("prohibited")]
        parsed = [{"force": "prohibited"}, {"force": "obligatory"}]
        out = dd.score_battery(parsed, gold)
        assert out["battery"] == 0.0           # cross-side scores 0
        assert out["antithesis_frac"] == 1.0
        assert out["hedge_frac"] == 0.0

    def test_unparsed_items_hedge_and_zero_cite(self):
        gold = [_gold("obligatory", "share it"), _gold("prohibited", "never tell")]
        parsed = [{"force": "obligatory", "governing_norm": "share it"}, None]
        out = dd.score_battery(parsed, gold)
        # item 1: s=1 cite=1 ; item 2 (None): hedge vs prohibited = 0.15, cite=0
        assert out["battery"] == pytest.approx((1.0 + dd.ITEM_SCORE_HEDGE) / 2)
        assert out["cite"] == pytest.approx(0.5)
        assert out["parsed_frac"] == 0.5
        assert out["hedge_frac"] == 0.5        # the None counts as a hedge

    def test_short_parsed_list_right_padded(self):
        gold = [_gold("obligatory"), _gold("prohibited"), _gold("recommended")]
        parsed = [{"force": "obligatory"}]  # only one item returned
        out = dd.score_battery(parsed, gold)
        assert out["parsed_frac"] == pytest.approx(1 / 3)

    def test_mild_antithesis_not_counted_when_same_polarity(self):
        # discouraged vs prohibited: same (negative) polarity — a same-side
        # commit (0.4), NOT an antithesis (axis product > 0).
        gold = [_gold("prohibited")]
        parsed = [{"force": "discouraged"}]
        out = dd.score_battery(parsed, gold)
        assert out["antithesis_frac"] == 0.0
        assert out["battery"] == pytest.approx(dd.ITEM_SCORE_SAME_SIDE)

    def test_empty_gold_raises(self):
        with pytest.raises(ValueError):
            dd.score_battery([], [])


class TestNonAnswerIsNotPermitted:
    """Audit 2026-07-28: a missing/unparseable answer must NOT score as a
    genuine "permitted" commit — axis-collapsing both let an unparsed slot
    earn a full 1.0 on a permitted-gold item (~4% of battery items)."""

    def test_none_vs_permitted_gold_is_hedge_not_exact(self):
        assert dd.item_score(None, "permitted") == dd.ITEM_SCORE_HEDGE
        assert dd.item_score("banana", "permitted") == dd.ITEM_SCORE_HEDGE

    def test_genuine_permitted_answer_still_exact(self):
        assert dd.item_score("permitted", "permitted") == dd.ITEM_SCORE_EXACT

    def test_unparsed_battery_item_on_permitted_gold(self):
        out = dd.score_battery([None], [{"gold_force": "permitted",
                                         "articulation": "a"}])
        assert out["battery"] == dd.ITEM_SCORE_HEDGE
        assert out["parsed_frac"] == 0.0

    def test_synonym_forces_canonicalise(self):
        # Latent-universe hardening: synonyms route through canonical_force
        # instead of collapsing to the hedge point.
        assert dd.item_score("forbidden", "prohibited") == dd.ITEM_SCORE_EXACT
        assert dd.axis_of("forbidden") == dd.AXIS["prohibited"]
