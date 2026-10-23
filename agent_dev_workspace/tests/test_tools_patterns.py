"""Tests for tools_patterns.py."""

import pytest

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


def _setup_session():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )
    pattern_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in pattern_tools if t.name == "mine_patterns_tool")
    return session, mine_patterns_tool


@pytest.mark.parametrize("filt_method", ["variance", "term_frequency"])
@pytest.mark.parametrize("category", ["catA", "catB"])
def test_zero_terms_survived(filt_method, category):
    # Known answer #17a: variance/term_frequency leave 0 terms for both categories
    session, tool = _setup_session()
    res = tool.invoke(
        {
            "category_name": category,
            "filtering_method": filt_method,
            "algorithm": "fpgrowth",
            "min_sup": 1,
        }
    )
    assert res["patterns"] == []
    assert res["kept_terms"] == []
    assert res["n_patterns"] == 0
    assert "no terms survived" in res.get("note", "")


@pytest.mark.parametrize("min_sup,expected_patterns", [
    (1, [{"items": ["alpha"], "support": 3}]),
    (2, [{"items": ["alpha"], "support": 3}]),
    (3, [{"items": ["alpha"], "support": 3}]),
    (4, []),
])
def test_fpgrowth_tfidf_catA(min_sup, expected_patterns):
    # Known answer #17b: catA + tfidf + fpgrowth
    session, tool = _setup_session()
    res = tool.invoke(
        {
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": min_sup,
        }
    )
    assert res["kept_terms"] == ["alpha"]
    assert res["patterns"] == expected_patterns
    assert res["n_patterns"] == len(expected_patterns)

    # Check session.artifacts contains the full pattern list
    artifact_key = "patterns_catA_tfidf_fpgrowth"
    assert artifact_key in session.artifacts
    assert session.artifacts[artifact_key] == expected_patterns


def test_fpgrowth_tfidf_catB():
    # Known answer #17b: catB + tfidf + fpgrowth
    session, tool = _setup_session()
    res = tool.invoke(
        {
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "fpgrowth",
            "min_sup": 4,
        }
    )
    assert res["kept_terms"] == ["gamma"]
    assert res["patterns"] == [{"items": ["gamma"], "support": 4}]
    assert res["n_patterns"] == 1


@pytest.mark.parametrize("k,category,expected_item,expected_sup", [
    (1, "catA", ["alpha"], 3),
    (2, "catA", ["alpha"], 3),
    (3, "catA", ["alpha"], 3),
    (1, "catB", ["gamma"], 4),
    (2, "catB", ["gamma"], 4),
    (3, "catB", ["gamma"], 4),
])
def test_topk_tfidf(k, category, expected_item, expected_sup):
    # Known answer #20: topk on tfidf filtered data
    session, tool = _setup_session()
    res = tool.invoke(
        {
            "category_name": category,
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k,
        }
    )
    assert res["patterns"] == [{"items": expected_item, "support": expected_sup}]
    assert res["n_patterns"] == 1


@pytest.mark.parametrize("min_sup,category", [
    (1, "catA"),
    (4, "catA"),
    (1, "catB"),
    (4, "catB"),
])
def test_maxfpgrowth_tfidf(min_sup, category):
    # Known answer #20: maxfpgrowth returns 0 patterns on single surviving term
    session, tool = _setup_session()
    res = tool.invoke(
        {
            "category_name": category,
            "filtering_method": "tfidf",
            "algorithm": "maxfpgrowth",
            "min_sup": min_sup,
        }
    )
    assert res["patterns"] == []
    assert res["n_patterns"] == 0


def test_min_sup_validation():
    # Validation error check
    session, tool = _setup_session()
    with pytest.raises(ValueError):
        tool.invoke(
            {
                "category_name": "catA",
                "filtering_method": "tfidf",
                "algorithm": "fpgrowth",
                "min_sup": 0,
            }
        )
