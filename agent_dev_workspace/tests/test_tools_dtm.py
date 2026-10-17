import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool():
    # Instantiate session state and load premade tools
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate student's build_dtm_tool using the same session
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # Invoke build_dtm_tool with default settings
    res = build_dtm_tool.invoke({})

    # Assert known answer 4 from TEST_FIXTURE.md
    # Vocabulary (5 terms: alpha, always, beta, delta, gamma)
    assert res["vocabulary_size"] == 5
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]

    # Sparsity (21 non-zero cells out of 40 total -> 47.5%)
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert res["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    # Check vectorizer artifact stored in session
    assert "count_vectorizer" in session.artifacts
