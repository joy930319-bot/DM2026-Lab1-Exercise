"""Tests for tools_exploration.py."""

import numpy as np
import pytest

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_cosine_similarity_tool():
    # 1. Initialize session and load fixture
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

    # 2. Build DTM to populate session.feature_matrix
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. Retrieve cosine_similarity_tool
    exploration_tools = make_exploration_tools(session)
    cosine_similarity_tool = next(
        t for t in exploration_tools if t.name == "cosine_similarity_tool"
    )

    # 4. Assert Known Answer #16 from TEST_FIXTURE.md
    res_0_1 = cosine_similarity_tool.invoke({"doc_index_a": 0, "doc_index_b": 1})
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-4)

    res_0_3 = cosine_similarity_tool.invoke({"doc_index_a": 0, "doc_index_b": 3})
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-4)

    res_3_4 = cosine_similarity_tool.invoke({"doc_index_a": 3, "doc_index_b": 4})
    assert res_3_4["similarity"] == pytest.approx(0.9847, abs=1e-4)

    res_0_7 = cosine_similarity_tool.invoke({"doc_index_a": 0, "doc_index_b": 7})
    assert res_0_7["similarity"] == 0.0
    assert not np.isnan(res_0_7["similarity"])

    # 5. Assert out-of-bounds index raises ValueError
    with pytest.raises(ValueError):
        cosine_similarity_tool.invoke({"doc_index_a": 99, "doc_index_b": 0})


def test_feature_correlation_matrix_tool():
    # 1. Initialize session and load fixture
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

    # 2. Build DTM to populate session.feature_matrix and session.feature_names
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. Retrieve feature_correlation_matrix_tool
    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(
        t for t in exploration_tools if t.name == "feature_correlation_matrix_tool"
    )

    # 4. Invoke tool and check outputs matching Known Answer #19
    res = feature_corr_tool.invoke({})

    assert res["terms"] == ["alpha", "gamma", "delta", "beta", "always"]

    matrix = np.array(res["matrix"])
    assert matrix.shape == (5, 5)

    # Check symmetry
    assert np.allclose(matrix, matrix.T)

    # Check diagonal elements are 1.0
    for i in range(5):
        assert matrix[i, i] == pytest.approx(1.0, abs=1e-4)

    # Check specific term-term correlation values
    # alpha (0), gamma (1), delta (2), beta (3), always (4)
    assert matrix[0, 3] == pytest.approx(0.8885, abs=1e-4)  # alpha vs beta
    assert matrix[1, 2] == pytest.approx(0.8307, abs=1e-4)  # gamma vs delta
    assert matrix[0, 1] == pytest.approx(-0.5718, abs=1e-4) # alpha vs gamma
    assert matrix[2, 4] == pytest.approx(0.3780, abs=1e-4)  # delta vs always

    # Check pending_figure is set
    assert session.pending_figure is not None
