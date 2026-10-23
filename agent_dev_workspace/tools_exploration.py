"""Tools for dataset exploration and document comparison."""

from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity


def make_tools(session: Any) -> list:
    """Factory function returning tools for document exploration bound to session."""

    @tool
    def cosine_similarity_tool(doc_index_a: int, doc_index_b: int) -> Dict[str, Any]:
        """Compute the cosine similarity between two documents by their row indices in the document-term matrix.

        Args:
            doc_index_a: Row index of the first document (0-based).
            doc_index_b: Row index of the second document (0-based).

        Returns:
            Dict containing result_id, doc_index_a, doc_index_b, category_a, category_b, and similarity score.
        """
        if session.feature_matrix is None:
            raise ValueError(
                "No document-term matrix found in session. Please run build_dtm_tool first."
            )

        n_rows = session.feature_matrix.shape[0]
        if not (0 <= doc_index_a < n_rows):
            raise ValueError(
                f"doc_index_a ({doc_index_a}) is out of bounds for feature matrix with {n_rows} documents."
            )
        if not (0 <= doc_index_b < n_rows):
            raise ValueError(
                f"doc_index_b ({doc_index_b}) is out of bounds for feature matrix with {n_rows} documents."
            )

        row_a = session.feature_matrix[doc_index_a]
        row_b = session.feature_matrix[doc_index_b]

        # Handle empty documents (all-zero rows) to prevent NaN output
        if row_a.nnz == 0 or row_b.nnz == 0:
            sim_value = 0.0
        else:
            sim_matrix = cosine_similarity(row_a, row_b)
            sim_value = float(sim_matrix[0, 0])

        # Retrieve category names if available in session
        category_a = None
        category_b = None
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            category_a = str(session.dataframe.iloc[doc_index_a]["category_name"])
            category_b = str(session.dataframe.iloc[doc_index_b]["category_name"])
        elif session.labels is not None:
            category_a = str(session.labels[doc_index_a])
            category_b = str(session.labels[doc_index_b])

        result_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": result_id,
            "doc_index_a": doc_index_a,
            "doc_index_b": doc_index_b,
            "category_a": category_a,
            "category_b": category_b,
            "similarity": round(sim_value, 4),
        }

        session.store_result(
            tool_name="cosine_similarity_tool",
            args={"doc_index_a": doc_index_a, "doc_index_b": doc_index_b},
            summary=summary,
        )

        return summary

    @tool
    def feature_correlation_matrix_tool() -> Dict[str, Any]:
        """
        Compute feature correlation matrix for top 20 terms by variance and produce a heatmap.

        Returns:
            Dict containing result_id, terms (selected term names in variance order), and matrix (nested list of floats).
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError("No document-term matrix found. Run build_dtm_tool first.")

        matrix_dense = session.feature_matrix.toarray()
        feature_names = np.array(session.feature_names)

        variances = np.var(matrix_dense, axis=0)
        sorted_indices = np.argsort(-variances, kind="stable")
        top_indices = sorted_indices[:20]

        top_terms = feature_names[top_indices].tolist()
        sub_matrix = matrix_dense[:, top_indices]

        corr = np.corrcoef(sub_matrix, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", xticklabels=top_terms, yticklabels=top_terms, ax=ax, cmap="coolwarm")
        ax.set_title("Feature Correlation Matrix (Top Variance Terms)")
        plt.tight_layout()

        session.pending_figure = fig

        matrix_list = corr.tolist()

        result_id = session.next_result_id("feature_corr")
        summary = {
            "result_id": result_id,
            "terms": top_terms,
            "matrix": matrix_list
        }
        session.store_result("feature_correlation_matrix_tool", {}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
