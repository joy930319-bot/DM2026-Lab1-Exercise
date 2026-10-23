"""Tests for tools_reduction.py."""

import numpy as np
import pytest

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def test_binarize_labels_tool():
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

    # 2. Retrieve binarize_labels_tool
    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(
        t for t in reduction_tools if t.name == "binarize_labels_tool"
    )

    # 3. Invoke tool
    res = binarize_labels_tool.invoke({})

    # 4. Assert response dictionary contents
    assert res["classes"] == ["catA", "catB"]
    assert res["shape"] == [8, 2]
    assert res["counts_per_class"] == {"catA": 4, "catB": 4}
    assert res["first_rows"][0] == [1, 0]
    assert res["first_rows"][3] == [0, 1]

    # 5. Assert session state and artifact updates
    matrix = session.artifacts["binarized_labels"]
    assert list(matrix[7]) == [1, 0]  # Empty text row is catA
    assert np.all(matrix.sum(axis=1) == 1)  # Every row sums to 1
    assert len(session.dataframe) == 8  # DataFrame un-modified


def test_reduce_dimensions_pca():
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

    # 2. Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. Retrieve reduce_dimensions_tool
    reduction_tools = make_reduction_tools(session)
    reduce_dims_tool = next(
        t for t in reduction_tools if t.name == "reduce_dimensions_tool"
    )

    # 4. Invoke with method="pca"
    res = reduce_dims_tool.invoke({"method": "pca"})

    # Known answer #7 coordinates
    expected_coords = np.array(
        [
            [2.3669, 0.8871],
            [1.7039, 0.2642],
            [1.0408, -0.3587],
            [-2.0759, 1.0129],
            [-1.4554, 0.3405],
            [-0.8349, -0.3319],
            [-0.8349, -0.3319],
            [0.0894, -1.4822],
        ]
    )

    actual_coords = session.artifacts["reduced_coordinates"]
    assert np.allclose(actual_coords, expected_coords, atol=1e-3)

    # Check explained variance ratio
    assert res["explained_variance_ratio"][0] == pytest.approx(0.7532, abs=1e-4)
    assert res["explained_variance_ratio"][1] == pytest.approx(0.1965, abs=1e-4)

    # Check plot was created
    assert session.pending_figure is not None


def test_reduce_dimensions_tsne():
    # Note: t-SNE's exact coordinate layout varies across scikit-learn versions and OS architectures,
    # so rather than asserting against absolute coordinate values from TEST_FIXTURE.md #8,
    # we assert invariant properties of the output.

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

    # 2. Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. Retrieve reduce_dimensions_tool
    reduction_tools = make_reduction_tools(session)
    reduce_dims_tool = next(
        t for t in reduction_tools if t.name == "reduce_dimensions_tool"
    )

    # 4. Invoke with method="tsne", perplexity=2, random_state=42
    reduce_dims_tool.invoke({"method": "tsne", "perplexity": 2, "random_state": 42})
    coords = session.artifacts["reduced_coordinates"]

    # Invariant 1: Shape is (8, 2) with no NaN values
    assert coords.shape == (8, 2)
    assert not np.isnan(coords).any()

    # Invariant 2: Identical text documents (docs 5 and 6) receive identical coordinates
    assert np.allclose(coords[5], coords[6], atol=1e-3)

    # Invariant 3: Reproducibility with random_state
    reduce_dims_tool.invoke({"method": "tsne", "perplexity": 2, "random_state": 42})
    coords_repeat = session.artifacts["reduced_coordinates"]
    assert np.allclose(coords, coords_repeat)

    # Invariant 4: Nearest neighbor clustering within same category
    # Doc 1 (catA) nearest neighbor among all other docs should be in catA (indices 0, 2, 7)
    dists_from_1 = np.linalg.norm(coords - coords[1], axis=1)
    dists_from_1[1] = np.inf  # exclude self
    nearest_to_1 = np.argmin(dists_from_1)
    assert nearest_to_1 in [0, 2, 7]

    # Doc 4 (catB) nearest neighbor among all other docs should be in catB (indices 3, 5, 6)
    dists_from_4 = np.linalg.norm(coords - coords[4], axis=1)
    dists_from_4[4] = np.inf  # exclude self
    nearest_to_4 = np.argmin(dists_from_4)
    assert nearest_to_4 in [3, 5, 6]

    # Invariant 5: Invalid perplexity >= n_samples (8) raises ValueError
    with pytest.raises(ValueError):
        reduce_dims_tool.invoke({"method": "tsne", "perplexity": 8, "random_state": 42})

    # Invariant 6: Plot was created
    assert session.pending_figure is not None


def test_reduce_dimensions_umap():
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

    # 2. Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # 3. Retrieve reduce_dimensions_tool
    reduction_tools = make_reduction_tools(session)
    reduce_dims_tool = next(
        t for t in reduction_tools if t.name == "reduce_dimensions_tool"
    )

    # 4. Invoke with method="umap", n_neighbors=3, random_state=42
    reduce_dims_tool.invoke({"method": "umap", "n_neighbors": 3, "random_state": 42})

    # Known answer #9 coordinates
    expected_coords = np.array(
        [
            [6.6811, -6.8218],
            [6.9907, -7.2245],
            [7.7022, -7.3712],
            [10.2228, -6.2310],
            [9.8759, -5.6537],
            [9.3831, -6.5748],
            [9.2153, -5.8297],
            [8.5860, -7.2622],
        ]
    )

    actual_coords = session.artifacts["reduced_coordinates"]
    assert np.allclose(actual_coords, expected_coords, atol=1e-3)

    # Check plot was created
    assert session.pending_figure is not None
