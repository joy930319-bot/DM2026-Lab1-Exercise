from typing import Dict, Any, List
from pathlib import Path
import os
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool

REPO_ROOT = Path(__file__).resolve().parent.parent


def make_tools(session: Any) -> List[Any]:
    """Factory function creating tools for data inspection and quality checks."""

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Checks session.dataframe for missing (NaN/None) or empty/whitespace-only text rows.

        Calculates missing text counts globally and per category, capped row indices,
        and missing percentage without mutating session.dataframe.

        Returns:
            Dict containing result_id, total_rows, missing_count, missing_percentage,
            missing_by_category, and missing_indices (capped at first 50).
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe must be loaded and contain a 'text' column before checking missing text.")

        df = session.dataframe
        text_col = df["text"]

        # Boolean mask for NaN/None or empty/whitespace-only strings
        is_missing = text_col.isna() | (text_col.astype(str).str.strip() == "")

        total_rows = len(df)
        missing_count = int(is_missing.sum())
        missing_percentage = float(round((missing_count / total_rows * 100.0), 4)) if total_rows > 0 else 0.0

        # Capped list of missing indices (first 50) as plain Python ints
        missing_indices_all = df.index[is_missing].tolist()
        missing_indices = [int(idx) for idx in missing_indices_all[:50]]

        # Count missing per category_name if available
        missing_by_category: Dict[str, int] = {}
        if "category_name" in df.columns:
            cat_counts = df[is_missing]["category_name"].value_counts()
            for cat, count in cat_counts.items():
                missing_by_category[str(cat)] = int(count)

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
            "missing_by_category": missing_by_category,
            "missing_indices": missing_indices
        }

        session.store_result(
            tool_name="check_missing_tool",
            args={},
            summary=summary
        )

        return summary

    @tool
    def check_duplicates_tool(
        drop: bool = False
    ) -> Dict[str, Any]:
        """Checks for duplicate documents in session.dataframe based on text column.

        Identifies extra duplicates (keep='first') and all rows involved (keep=False).
        If drop=True, removes every copy of duplicated rows (keep=False), updates session.dataframe,
        resets DTM state if present, and returns updated row counts.

        Args:
            drop: Whether to drop all duplicate instances from session.dataframe (default False).

        Returns:
            Dict containing result_id, drop, initial_rows, final_rows, duplicate_count,
            rows_involved, and an optional note if DTM state was reset.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe must be loaded and contain a 'text' column before checking duplicates.")

        df = session.dataframe
        initial_rows = len(df)

        # duplicate_count using keep='first'
        dup_extra_mask = df.duplicated(subset=["text"], keep="first")
        duplicate_count = int(dup_extra_mask.sum())

        # rows_involved using keep=False
        dup_all_mask = df.duplicated(subset=["text"], keep=False)
        rows_involved_all = df.index[dup_all_mask].tolist()
        rows_involved = [int(idx) for idx in rows_involved_all[:50]]

        note = None

        if drop:
            # Drop every copy of a duplicated row using keep=False
            new_df = df.drop_duplicates(subset=["text"], keep=False).reset_index(drop=True)
            session.dataframe = new_df

            # If DTM was built, clear state to prevent misalignment
            if session.feature_matrix is not None:
                session.feature_matrix = None
                session.feature_names = None
                session.labels = None
                note = "Duplicates dropped using keep=False. DTM state was cleared; rerun build_dtm_tool."

        final_rows = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "drop": drop,
            "initial_rows": initial_rows,
            "final_rows": final_rows,
            "duplicate_count": duplicate_count,
            "rows_involved": rows_involved
        }

        if note is not None:
            summary["note"] = note

        session.store_result(
            tool_name="check_duplicates_tool",
            args={"drop": drop},
            summary=summary
        )

        return summary

    @tool
    def sample_data_tool(
        n: int,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Subsamples session.dataframe using pandas DataFrame.sample.

        Replaces session.dataframe with the sampled rows, resets index, and updates state.
        If a DTM was built previously, clears feature_matrix, feature_names, and labels to keep session state in sync.

        Args:
            n: Number of rows to sample.
            random_state: Seed for reproducible sampling (default 42).

        Returns:
            Dict containing result_id, initial_rows, sampled_rows, sampled_indices (capped at 50),
            counts_before, counts_after, and an optional note if clamping or DTM reset occurred.
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe must be loaded before sampling.")

        df = session.dataframe
        initial_rows = len(df)

        counts_before: Dict[str, int] = {}
        if "category_name" in df.columns:
            for cat, count in df["category_name"].value_counts().items():
                counts_before[str(cat)] = int(count)

        notes: List[str] = []

        actual_n = n
        if n > initial_rows:
            actual_n = initial_rows
            notes.append(f"Requested sample size n={n} exceeded total rows {initial_rows}; clamped to {initial_rows}.")

        sampled_df = df.sample(n=actual_n, random_state=random_state)

        # Original row indices prior to resetting index (capped at 50 plain Python ints)
        sampled_indices = [int(idx) for idx in sampled_df.index.tolist()[:50]]

        # Reset index on sampled dataframe and replace session.dataframe
        sampled_df = sampled_df.reset_index(drop=True)
        session.dataframe = sampled_df

        counts_after: Dict[str, int] = {}
        if "category_name" in sampled_df.columns:
            for cat, count in sampled_df["category_name"].value_counts().items():
                counts_after[str(cat)] = int(count)

        if session.feature_matrix is not None:
            session.feature_matrix = None
            session.feature_names = None
            session.labels = None
            notes.append("DTM state was cleared; rerun build_dtm_tool.")

        summary: Dict[str, Any] = {
            "result_id": session.next_result_id("sample_data"),
            "initial_rows": initial_rows,
            "sampled_rows": len(sampled_df),
            "sampled_indices": sampled_indices,
            "counts_before": counts_before,
            "counts_after": counts_after,
        }

        if notes:
            summary["note"] = " ".join(notes)

        session.store_result(
            tool_name="sample_data_tool",
            args={"n": n, "random_state": random_state},
            summary=summary
        )

        return summary

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenizes text in session.dataframe into lowercased unigrams using nltk.word_tokenize.

        Stores list of unigrams for each document in session.dataframe['unigrams'].
        Handles missing or empty text gracefully by assigning an empty list.

        Returns:
            Dict containing result_id, n_documents, total_tokens, avg_tokens_per_doc,
            and sample_tokens_first_3.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe must be loaded and contain a 'text' column before tokenizing.")

        df = session.dataframe

        def _tokenize_text(text_val: Any) -> List[str]:
            if pd.isna(text_val) or text_val is None:
                return []
            text_str = str(text_val).strip()
            if not text_str:
                return []
            try:
                return nltk.word_tokenize(text_str.lower())
            except LookupError:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
                return nltk.word_tokenize(text_str.lower())

        df["unigrams"] = df["text"].apply(_tokenize_text)

        unigram_lists = df["unigrams"].tolist()
        n_documents = len(df)
        total_tokens = sum(len(tokens) for tokens in unigram_lists)
        avg_tokens_per_doc = float(round(total_tokens / n_documents, 4)) if n_documents > 0 else 0.0
        sample_tokens_first_3 = unigram_lists[:3]

        summary = {
            "result_id": session.next_result_id("tokenize"),
            "n_documents": n_documents,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": avg_tokens_per_doc,
            "sample_tokens_first_3": sample_tokens_first_3
        }

        session.store_result(
            tool_name="tokenize_tool",
            args={},
            summary=summary
        )

        return summary

    @tool
    def list_files_tool(
        subdirectory: str = ""
    ) -> Dict[str, Any]:
        """Lists directory names and file names in a specified subdirectory relative to the repository root.

        Ensures security by preventing traversal outside the repository root.

        Args:
            subdirectory: Subdirectory path relative to the repo root (default "").

        Returns:
            Dict containing result_id, subdirectory, directories (sorted list of strings),
            and files (sorted list of strings).
        """
        target_path = (REPO_ROOT / subdirectory).resolve()

        # Path traversal guard using is_relative_to
        try:
            is_relative = target_path.is_relative_to(REPO_ROOT)
        except AttributeError:
            # Fallback for Python versions < 3.9 if needed, though is_relative_to is standard in 3.9+
            try:
                target_path.relative_to(REPO_ROOT)
                is_relative = True
            except ValueError:
                is_relative = False

        if not is_relative:
            raise ValueError(f"Access denied: Path '{subdirectory}' resolves outside repository root.")

        if not target_path.exists():
            raise ValueError(f"Path '{subdirectory}' does not exist.")

        if not target_path.is_dir():
            raise ValueError(f"Path '{subdirectory}' is not a directory.")

        directories: List[str] = []
        files: List[str] = []

        for item in target_path.iterdir():
            if item.is_dir():
                directories.append(item.name)
            elif item.is_file():
                files.append(item.name)

        directories.sort()
        files.sort()

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files
        }

        session.store_result(
            tool_name="list_files_tool",
            args={"subdirectory": subdirectory},
            summary=summary
        )

        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Computes text length statistics overall and per category, and generates a box plot.

        Calculates character count statistics (.describe()) for document text without mutating session.dataframe.
        Creates a box plot of text_length grouped by category_name and sets it on session.pending_figure.

        Returns:
            Dict containing result_id, overall (describe stats dict), and per_category (dict of describe stats dicts).
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe must be loaded and contain a 'text' column before describing data.")

        df = session.dataframe
        text_length = df["text"].fillna("").apply(len)

        def _get_describe_dict(s: pd.Series) -> Dict[str, Any]:
            desc = s.describe()
            std_val = desc["std"]
            std_formatted = float(std_val) if pd.notna(std_val) else None

            return {
                "count": int(desc["count"]),
                "mean": float(desc["mean"]),
                "std": std_formatted,
                "min": float(desc["min"]),
                "25%": float(desc["25%"]),
                "50%": float(desc["50%"]),
                "75%": float(desc["75%"]),
                "max": float(desc["max"]),
            }

        overall_stats = _get_describe_dict(text_length)

        per_category_stats: Dict[str, Dict[str, Any]] = {}
        if "category_name" in df.columns:
            plot_df = pd.DataFrame({
                "category_name": df["category_name"],
                "text_length": text_length
            })
            for cat, group in plot_df.groupby("category_name"):
                per_category_stats[str(cat)] = _get_describe_dict(group["text_length"])

            # Plotting boxplot grouped by category_name
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=plot_df, x="category_name", y="text_length", ax=ax)
            ax.set_title("Text Length Distribution by Category")
            ax.set_xlabel("Category")
            ax.set_ylabel("Text Length (characters)")

            session.pending_figure = fig
        else:
            # Overall boxplot if category_name is absent
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.boxplot(y=text_length, ax=ax)
            ax.set_title("Text Length Distribution")
            ax.set_ylabel("Text Length (characters)")

            session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall": overall_stats,
            "per_category": per_category_stats
        }

        session.store_result(
            tool_name="describe_data_tool",
            args={},
            summary=summary
        )

        return summary

    @tool
    def inspect_data_tool(
        n_rows: int = 5
    ) -> Dict[str, Any]:
        """Provides a preview of up to n_rows from session.dataframe with truncated text.

        Does not mutate session.dataframe. Returns total row count, column names,
        and a list of row preview dicts with truncated text (first 100 characters).

        Args:
            n_rows: Number of top rows to preview (default 5). Must be >= 1.

        Returns:
            Dict containing result_id, total_rows, columns, and preview (list of dicts).
        """
        if session.dataframe is None:
            raise ValueError("session.dataframe must be loaded before inspecting data.")

        if n_rows < 1:
            raise ValueError("n_rows must be an integer >= 1.")

        df = session.dataframe
        total_rows = len(df)
        columns = [str(col) for col in df.columns]

        head_df = df.head(n_rows)
        preview: List[Dict[str, Any]] = []

        for idx, row in head_df.iterrows():
            cat_name = str(row["category_name"]) if "category_name" in df.columns and pd.notna(row["category_name"]) else None
            
            raw_text = row["text"] if "text" in df.columns else ""
            if pd.isna(raw_text) or raw_text is None:
                text_str = ""
            else:
                text_str = str(raw_text)
            
            truncated_text = text_str[:100]

            row_preview = {
                "index": int(idx),
                "category_name": cat_name,
                "text": truncated_text
            }
            preview.append(row_preview)

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "columns": columns,
            "preview": preview
        }

        session.store_result(
            tool_name="inspect_data_tool",
            args={"n_rows": n_rows},
            summary=summary
        )

        return summary

    return [
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        tokenize_tool,
        list_files_tool,
        describe_data_tool,
        inspect_data_tool,
    ]
