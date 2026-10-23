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


def test_term_frequency_tool():
    # Each test gets its own fresh session to avoid state mutation side effects
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate student's tools using the same session
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    # Build DTM first
    build_dtm_tool.invoke({})

    # Run term_frequency_tool
    res = term_frequency_tool.invoke({"top_n": 20})

    # Assert known answer 14 from TEST_FIXTURE.md
    # Column sums / total counts per term:
    # alpha: 6, always: 7, beta: 3, delta: 4, gamma: 7
    expected_top_terms = {
        "always": 7,
        "gamma": 7,
        "alpha": 6,
        "delta": 4,
        "beta": 3
    }

    assert res["total_terms"] == 5
    assert res["total_count"] == 27
    assert res["top_terms"] == expected_top_terms


def test_dtm_heatmap_tool():
    # Each test gets its own fresh session
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate student's tools using the same session
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    # Build DTM first
    build_dtm_tool.invoke({})

    # 1. Default call (n_terms=20, n_documents=20)
    res_default = dtm_heatmap_tool.invoke({})

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]

    assert res_default["requested_n_terms"] == 20
    assert res_default["requested_n_documents"] == 20
    assert res_default["actual_n_terms"] == 5
    assert res_default["actual_n_documents"] == 8
    assert res_default["terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert res_default["matrix"] == expected_full_matrix
    assert session.pending_figure is not None

    # Reset pending figure to check slice call independently
    session.pending_figure = None

    # 2. Slice call (n_terms=2, n_documents=3)
    res_slice = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    expected_slice_matrix = [
        [3, 1],
        [2, 1],
        [1, 1]
    ]

    assert res_slice["requested_n_terms"] == 2
    assert res_slice["requested_n_documents"] == 3
    assert res_slice["actual_n_terms"] == 2
    assert res_slice["actual_n_documents"] == 3
    assert res_slice["terms"] == ["alpha", "always"]
    assert res_slice["matrix"] == expected_slice_matrix
    assert session.pending_figure is not None
