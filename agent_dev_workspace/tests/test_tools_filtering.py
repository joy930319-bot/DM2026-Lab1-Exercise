import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_known_answer_5():
    """Test variance_filter_tool against known answer #5 in TEST_FIXTURE.md.

    Fixture per-term variance values on sample_fixture.csv:
      - alpha: 1.1875
      - always: 0.109375
      - beta: 0.234375
      - delta: 0.25
      - gamma: 1.109375

    With threshold=0.15, alpha, beta, delta, and gamma should be kept (4 terms),
    and always should be removed (1 term).
    """
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM to populate session.feature_matrix and session.feature_names
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({"ngram_range": [1, 1]})

    # Get variance_filter_tool
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Run variance_filter_tool with threshold=0.15
    result = variance_filter_tool.invoke({"threshold": 0.15})

    assert result["threshold"] == 0.15
    assert result["total_features"] == 5
    assert result["kept_features_count"] == 4
    assert result["removed_features_count"] == 1

    assert set(result["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert result["removed_terms"] == ["always"]

    # Verify expected variance numbers from TEST_FIXTURE.md #5
    expected_variances = {
        "alpha": 1.1875,
        "always": 0.109375,
        "beta": 0.234375,
        "delta": 0.25,
        "gamma": 1.109375,
    }

    stored_result = session.results_store.get(result["result_id"])
    assert stored_result is not None
    full_report = stored_result.get("full_report")
    assert full_report is not None

    for term, expected_var in expected_variances.items():
        actual_var = full_report.loc[full_report["term"] == term, "variance"].values[0]
        assert np.isclose(actual_var, expected_var, atol=1e-3)
