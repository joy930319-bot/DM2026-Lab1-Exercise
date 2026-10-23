import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools as make_data_tools


def test_check_missing_tool_known_answer_2():
    """Test check_missing_tool against known answer #2 in TEST_FIXTURE.md.

    Fixture sample_fixture.csv has 8 rows total:
      - 1 empty text row at index 7 (label: catA).
      - missing_count = 1
      - missing_indices = [7]
      - missing_by_category = {"catA": 1}
      - session.dataframe length remains 8 (not mutated).
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_data_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    res = check_missing_tool.invoke({})

    assert res["total_rows"] == 8
    assert res["missing_count"] == 1
    assert res["missing_indices"] == [7]
    assert res["missing_by_category"] == {"catA": 1}

    # Confirm session.dataframe was not mutated
    assert len(session.dataframe) == 8


def test_check_duplicates_tool_no_drop_known_answer_3():
    """Test check_duplicates_tool with drop=False against known answer #3 in TEST_FIXTURE.md.

    Fixture sample_fixture.csv has row 6 as an exact copy of row 5 ("always gamma delta"):
      - duplicate_count = 1
      - rows_involved = [5, 6]
      - initial_rows = 8, final_rows = 8 (not dropped)
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_data_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    res = check_duplicates_tool.invoke({"drop": False})

    assert res["duplicate_count"] == 1
    assert res["rows_involved"] == [5, 6]
    assert res["initial_rows"] == 8
    assert res["final_rows"] == 8
    assert len(session.dataframe) == 8


def test_check_duplicates_tool_drop_known_answer_3():
    """Test check_duplicates_tool with drop=True against known answer #3 in TEST_FIXTURE.md.

    Dropping duplicates with keep=False removes EVERY copy of "always gamma delta"
    (both row 5 and row 6), leaving exactly 6 rows total.
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_data_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    res = check_duplicates_tool.invoke({"drop": True})

    assert res["initial_rows"] == 8
    assert res["final_rows"] == 6
    assert len(session.dataframe) == 6

    # Verify that neither copy of "always gamma delta" remains in session.dataframe
    remaining_texts = session.dataframe["text"].tolist()
    assert "always gamma delta" not in remaining_texts


def test_sample_data_tool_known_answer_13():
    """Test sample_data_tool against known answer #13 in TEST_FIXTURE.md.

    Calling sample_data_tool(n=4, random_state=42) on sample_fixture.csv:
      - sampled_indices = [1, 5, 0, 7]
      - counts_after = {"catA": 3, "catB": 1}
      - session.dataframe length = 4, indexed 0..3
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_data_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    assert res["sampled_indices"] == [1, 5, 0, 7]
    assert res["counts_after"] == {"catA": 3, "catB": 1}
    assert res["sampled_rows"] == 4
    assert len(session.dataframe) == 4
    assert list(session.dataframe.index) == [0, 1, 2, 3]


def test_sample_data_tool_clamping():
    """Test sample_data_tool when requested n exceeds total row count."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_data_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    res = sample_data_tool.invoke({"n": 100, "random_state": 42})

    assert res["initial_rows"] == 8
    assert res["sampled_rows"] == 8
    assert len(session.dataframe) == 8
    assert "note" in res
    assert "exceeded total rows" in res["note"] or "clamped" in res["note"]


def test_tokenize_tool_known_answer_10():
    """Test tokenize_tool against known answer #10 in TEST_FIXTURE.md.

    Verifies lowercased unigram token lists across all 8 documents,
    ensuring doc 7 is an empty list and total token count equals 27.
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_data_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({})

    assert res["n_documents"] == 8
    assert res["total_tokens"] == 27
    assert "unigrams" in session.dataframe.columns

    expected_unigrams = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]

    actual_unigrams = session.dataframe["unigrams"].tolist()
    assert actual_unigrams == expected_unigrams


def test_list_files_tool_known_answer_11():
    """Test list_files_tool against known answer #11 in TEST_FIXTURE.md.

    Listing subdirectory="newdataset" should return:
      - directories == []
      - files == ["Reddit-stock-sentiment.csv"]
    """
    session = SessionState()
    data_tools = make_data_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]


def test_list_files_tool_path_traversal_protection():
    """Test list_files_tool prevents path traversal outside repository root."""
    session = SessionState()
    data_tools = make_data_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    with pytest.raises(ValueError):
        list_files_tool.invoke({"subdirectory": "../.."})


def test_describe_data_tool_known_answer_18():
    """Test describe_data_tool against known answer #18 in TEST_FIXTURE.md.

    Verifies text_length statistics overall and per category (catA and catB),
    ensures session.pending_figure is populated, and confirms session.dataframe
    was not mutated with a "text_length" column.
    """
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_data_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({})

    overall = res["overall"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-4)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-4)
    assert overall["min"] == pytest.approx(0.0, abs=1e-4)
    assert overall["25%"] == pytest.approx(17.75, abs=1e-4)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-4)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-4)
    assert overall["max"] == pytest.approx(30.0, abs=1e-4)

    per_cat = res["per_category"]
    assert "catA" in per_cat
    assert "catB" in per_cat

    catA = per_cat["catA"]
    assert catA["count"] == 4
    assert catA["mean"] == pytest.approx(17.25, abs=1e-4)
    assert catA["std"] == pytest.approx(12.5, abs=1e-4)
    assert catA["min"] == pytest.approx(0.0, abs=1e-4)
    assert catA["25%"] == pytest.approx(12.75, abs=1e-4)
    assert catA["50%"] == pytest.approx(20.0, abs=1e-4)
    assert catA["75%"] == pytest.approx(24.5, abs=1e-4)
    assert catA["max"] == pytest.approx(29.0, abs=1e-4)

    catB = per_cat["catB"]
    assert catB["count"] == 4
    assert catB["mean"] == pytest.approx(22.5, abs=1e-4)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-4)
    assert catB["min"] == pytest.approx(18.0, abs=1e-4)
    assert catB["25%"] == pytest.approx(18.0, abs=1e-4)
    assert catB["50%"] == pytest.approx(21.0, abs=1e-4)
    assert catB["75%"] == pytest.approx(25.5, abs=1e-4)
    assert catB["max"] == pytest.approx(30.0, abs=1e-4)

    # Check that pending_figure was set
    assert session.pending_figure is not None

    # Check that session.dataframe was not mutated to add text_length
    assert "text_length" not in session.dataframe.columns


def test_inspect_data_tool_n_rows_3():
    """Test inspect_data_tool with n_rows=3."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_data_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({"n_rows": 3})

    assert res["total_rows"] == 8
    assert "text" in res["columns"]
    assert "category_name" in res["columns"]

    preview = res["preview"]
    assert len(preview) == 3
    assert [p["index"] for p in preview] == [0, 1, 2]
    assert preview[0]["text"] == "always alpha alpha alpha beta"

    assert len(session.dataframe) == 8


def test_inspect_data_tool_large_n_rows():
    """Test inspect_data_tool with n_rows larger than dataframe row count."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_data_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    res = inspect_data_tool.invoke({"n_rows": 100})

    preview = res["preview"]
    assert len(preview) == 8
    assert preview[-1]["text"] == ""


def test_inspect_data_tool_invalid_n_rows():
    """Test inspect_data_tool with invalid n_rows < 1."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_data_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    with pytest.raises(ValueError):
        inspect_data_tool.invoke({"n_rows": 0})
