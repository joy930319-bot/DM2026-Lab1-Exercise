from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session: Any) -> List[Any]:
    """Factory function creating tools for Document-Term Matrix operations."""

    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0
    ) -> Dict[str, Any]:
        """Builds a document-term matrix (DTM) from session.dataframe text column.

        Calculates matrix vocabulary, sparsity percentage, stores the fitted vectorizer in
        session.artifacts['count_vectorizer'], and updates session feature matrix and labels.

        Args:
            ngram_range: List of [min_n, max_n] for n-gram extraction (default [1, 1]).
            min_df: Minimum document frequency cutoff (int for count, float for ratio).
            max_df: Maximum document frequency cutoff (int for count, float for ratio).

        Returns:
            Dict containing result_id, n_documents, vocabulary_size, non_zero,
            total_elements, and sparsity_pct.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            raise ValueError("session.dataframe must be loaded and contain a 'text' column before building DTM.")

        # Convert List[int] to tuple for sklearn CountVectorizer compatibility
        ngram_tuple = (ngram_range[0], ngram_range[1])

        # Initialize and fit CountVectorizer
        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            min_df=min_df,
            max_df=max_df
        )

        texts = session.dataframe["text"].fillna("").astype(str)
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = list(vectorizer.get_feature_names_out())

        # Update session state
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names

        # Always check and update labels when available (or handle missing gracefully if set_labels expects labels)
        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        # Store vectorizer artifact for downstream tools like cosine_similarity_tool
        session.artifacts["count_vectorizer"] = vectorizer

        # Sparsity calculation
        n_rows, n_cols = feature_matrix.shape
        total_elements = n_rows * n_cols
        non_zero = int(feature_matrix.nnz)
        sparsity_pct = 100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0

        result_id = session.next_result_id("dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "vocabulary_size": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4)
        }

        session.store_result(
            tool_name="build_dtm_tool",
            args={"ngram_range": ngram_range, "min_df": min_df, "max_df": max_df},
            summary=summary
        )

        return summary

    @tool
    def term_frequency_tool(
        top_n: int = 20
    ) -> Dict[str, Any]:
        """Aggregates term frequencies across all documents in session.feature_matrix.

        Sums counts per term across documents, sorts terms by frequency descending,
        and returns the top N terms while storing the full sorted report DataFrame
        for visualization tools.

        Args:
            top_n: Number of top frequent terms to include in the returned summary (default 20).

        Returns:
            Dict containing result_id, total_terms, total_count, and top_terms mapping term to frequency.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("session.feature_matrix and session.feature_names must be built before calculating term frequency.")

        # Aggregate counts per column across all documents
        counts = np.asarray(session.feature_matrix.sum(axis=0)).ravel()
        feature_names = session.feature_names

        # Create sorted DataFrame: first column 'term', second column 'frequency'
        full_report = pd.DataFrame({
            "term": feature_names,
            "frequency": [int(c) for c in counts]
        }).sort_values(by="frequency", ascending=False, kind="stable").reset_index(drop=True)

        # Slice top N terms for summary dict with explicit Python int conversion
        top_df = full_report.head(top_n)
        top_terms = {row["term"]: int(row["frequency"]) for _, row in top_df.iterrows()}

        result_id = session.next_result_id("tf")
        summary = {
            "result_id": result_id,
            "total_terms": len(feature_names),
            "total_count": int(counts.sum()),
            "top_terms": top_terms
        }

        session.store_result(
            tool_name="term_frequency_tool",
            args={"top_n": top_n},
            summary=summary,
            full_report=full_report
        )

        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20
    ) -> Dict[str, Any]:
        """Generates a raw positional heatmap slice of session.feature_matrix.

        Slices up to n_documents and n_terms from session.feature_matrix, builds
        an annotated heatmap set on session.pending_figure, and returns the slice data.

        Args:
            n_terms: Number of terms (columns) to include from session.feature_names (default 20).
            n_documents: Number of documents (rows) to include from session.dataframe (default 20).

        Returns:
            Dict containing result_id, requested and actual counts, sliced terms, doc_indices, and matrix.
        """
        if session.feature_matrix is None or session.feature_names is None or session.dataframe is None:
            raise ValueError("session.feature_matrix, session.feature_names, and session.dataframe must be initialized before generating heatmap.")

        # Positional slicing up to requested counts
        matrix_slice_sparse = session.feature_matrix[0:n_documents, 0:n_terms]
        matrix_slice = matrix_slice_sparse.toarray()

        actual_n_docs, actual_n_terms = matrix_slice.shape
        sliced_terms = session.feature_names[:actual_n_terms]
        doc_indices = [int(idx) for idx in session.dataframe.index[:actual_n_docs]]

        # Build heatmap figure
        fig, ax = plt.subplots(figsize=(max(6, actual_n_terms * 0.8), max(4, actual_n_docs * 0.5)))
        sns.heatmap(
            matrix_slice,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=sliced_terms,
            yticklabels=doc_indices,
            ax=ax
        )
        ax.set_title(f"DTM Heatmap ({actual_n_docs} Docs x {actual_n_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Document Index")
        plt.tight_layout()

        # Set pending figure for the chat UI/loader
        session.pending_figure = fig

        # Convert matrix slice to JSON-serializable list of lists with Python ints
        matrix_list = [[int(val) for val in row] for row in matrix_slice]

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "requested_n_terms": n_terms,
            "requested_n_documents": n_documents,
            "actual_n_terms": actual_n_terms,
            "actual_n_documents": actual_n_docs,
            "terms": sliced_terms,
            "doc_indices": doc_indices,
            "matrix": matrix_list
        }

        session.store_result(
            tool_name="dtm_heatmap_tool",
            args={"n_terms": n_terms, "n_documents": n_documents},
            summary=summary
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
