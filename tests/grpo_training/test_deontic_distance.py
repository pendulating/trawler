"""Tests for the m-series deontic-distance scorer (`T-VIGNETTE` reward).

Covers the task-vignettes.md contract:
  * AXIS map + the full 25-force-pair item_score table (incl. unknown→permitted);
  * battery_score rescale, and the antithesis cells (exact 1.0 / adjacent 0.5 /
    hedge 0 / mild antithesis −0.5 / full −1.0) matching the wiki table;
  * cite Jaccard overlap, empty-side → 0;
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

# Hardcoded s_i = 1 − |axis(m) − axis(g)| / 2 for every (model, gold) pair.
# Rows = model force, columns = gold force, in _FORCES order.
_ITEM_TABLE = {
    "obligatory":  {"obligatory": 1.0, "recommended": 0.5, "permitted": 0.0, "discouraged": -0.5, "prohibited": -1.0},
    "recommended": {"obligatory": 0.5, "recommended": 1.0, "permitted": 0.5, "discouraged": 0.0, "prohibited": -0.5},
    "permitted":   {"obligatory": 0.0, "recommended": 0.5, "permitted": 1.0, "discouraged": 0.5, "prohibited": 0.0},
    "discouraged": {"obligatory": -0.5, "recommended": 0.0, "permitted": 0.5, "discouraged": 1.0, "prohibited": 0.5},
    "prohibited":  {"obligatory": -1.0, "recommended": -0.5, "permitted": 0.0, "discouraged": 0.5, "prohibited": 1.0},
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


class TestItemScoreTable:
    @pytest.mark.parametrize("model", _FORCES)
    @pytest.mark.parametrize("gold", _FORCES)
    def test_all_25_pairs(self, model, gold):
        assert dd.item_score(model, gold) == pytest.approx(_ITEM_TABLE[model][gold])

    def test_range_bounds(self):
        assert dd.item_score("obligatory", "prohibited") == -1.0  # min
        assert dd.item_score("prohibited", "prohibited") == 1.0   # max

    def test_named_cells_from_wiki_table(self):
        # gold = obligatory: exact / adjacent / hedge / mild anti / full anti.
        assert dd.item_score("obligatory", "obligatory") == 1.0
        assert dd.item_score("recommended", "obligatory") == 0.5
        assert dd.item_score("permitted", "obligatory") == 0.0
        assert dd.item_score("discouraged", "obligatory") == -0.5
        assert dd.item_score("prohibited", "obligatory") == -1.0


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
        assert dd.item_score("  Prohibited ", "Obligatory") == -1.0

    def test_unknown_gold_collapses_to_permitted(self):
        assert dd.item_score("obligatory", "nonsense") == dd.item_score("obligatory", "permitted")


# --------------------------------------------------------------------------- #
# battery_score: rescale + empty raises                                        #
# --------------------------------------------------------------------------- #

class TestBatteryScore:
    def test_rescale_endpoints(self):
        assert dd.battery_score([1.0]) == 1.0     # exact-force battery
        assert dd.battery_score([0.0]) == 0.5     # all-hedge battery
        assert dd.battery_score([-1.0]) == 0.0    # full-antithesis battery

    def test_mean_then_rescale(self):
        # mean of [1.0, -1.0] = 0 → 0.5
        assert dd.battery_score([1.0, -1.0]) == 0.5
        # mean of [0.5, -0.5] = 0 → 0.5
        assert dd.battery_score([0.5, -0.5]) == 0.5
        # mean of [1.0, 0.5, 0.0] = 0.5 → 0.75
        assert dd.battery_score([1.0, 0.5, 0.0]) == pytest.approx(0.75)

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
        assert out["r_vig"] == pytest.approx(0.7 * 1.0 + 0.3 * 1.0)
        assert out["hedge_frac"] == 0.0
        assert out["antithesis_frac"] == 0.0
        assert out["parsed_frac"] == 1.0

    def test_all_hedge(self):
        gold = [_gold("obligatory"), _gold("prohibited")]
        parsed = [{"force": "permitted"}, {"force": "permitted"}]
        out = dd.score_battery(parsed, gold)
        assert out["battery"] == 0.5           # mean s_i = 0 → 0.5
        assert out["hedge_frac"] == 1.0
        assert out["antithesis_frac"] == 0.0   # permitted has axis 0, product 0
        assert out["parsed_frac"] == 1.0

    def test_full_antithesis(self):
        gold = [_gold("obligatory"), _gold("prohibited")]
        parsed = [{"force": "prohibited"}, {"force": "obligatory"}]
        out = dd.score_battery(parsed, gold)
        assert out["battery"] == 0.0           # both s_i = -1 → mean -1 → 0
        assert out["antithesis_frac"] == 1.0
        assert out["hedge_frac"] == 0.0

    def test_unparsed_items_hedge_and_zero_cite(self):
        gold = [_gold("obligatory", "share it"), _gold("prohibited", "never tell")]
        parsed = [{"force": "obligatory", "governing_norm": "share it"}, None]
        out = dd.score_battery(parsed, gold)
        # item 1: s=1 cite=1 ; item 2 (None): s=item_score(None, prohibited)=0, cite=0
        assert out["battery"] == pytest.approx((( 1.0 + 0.0) / 2 + 1) / 2)  # (0.5+1)/2 = 0.75
        assert out["cite"] == pytest.approx(0.5)
        assert out["parsed_frac"] == 0.5
        assert out["hedge_frac"] == 0.5        # the None counts as a hedge

    def test_short_parsed_list_right_padded(self):
        gold = [_gold("obligatory"), _gold("prohibited"), _gold("recommended")]
        parsed = [{"force": "obligatory"}]  # only one item returned
        out = dd.score_battery(parsed, gold)
        assert out["parsed_frac"] == pytest.approx(1 / 3)

    def test_mild_antithesis_not_counted_when_same_polarity(self):
        # discouraged vs prohibited: same (negative) polarity, adjacent degree —
        # s=0.5, NOT an antithesis (axis product > 0).
        gold = [_gold("prohibited")]
        parsed = [{"force": "discouraged"}]
        out = dd.score_battery(parsed, gold)
        assert out["antithesis_frac"] == 0.0
        assert out["battery"] == pytest.approx((0.5 + 1) / 2)

    def test_empty_gold_raises(self):
        with pytest.raises(ValueError):
            dd.score_battery([], [])
