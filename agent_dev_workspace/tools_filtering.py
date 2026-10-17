from typing import Dict, Any, List
import pandas as pd
import numpy as np
from langchain_core.tools import tool


def make_tools(session: Any) -> List[Any]:
    """Factory function creating tools for feature filtering operations."""

    @tool
    def variance_filter_tool(
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Flags low-variance terms in the document-term matrix for removal.

        Computes the variance across documents for each term in session.feature_matrix.
        Terms with variance less than the threshold are flagged for removal.

        Args:
            threshold: Minimum variance required to keep a term (default 0.0).

        Returns:
            Dict containing result_id, threshold, total_features,
            kept_features_count, removed_features_count, kept_terms, and removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "session.feature_matrix and session.feature_names must be available. "
                "Run build_dtm_tool first."
            )

        matrix = session.feature_matrix

        # Calculate column-wise variance directly on sparse matrix:
        # Var(X) = E[X^2] - (E[X])^2
        # matrix.power(2).mean(axis=0) computes mean of squared entries per column
        mean_sq = np.array(matrix.power(2).mean(axis=0)).ravel()
        mean = np.array(matrix.mean(axis=0)).ravel()
        variances = mean_sq - mean ** 2

        feature_names = session.feature_names

        # Create full report DataFrame sorted by variance descending
        # First column must be term name, second column must be variance value
        report_df = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        # Identify kept and removed terms based on threshold
        kept_mask = report_df["variance"] >= threshold
        kept_terms = report_df.loc[kept_mask, "term"].tolist()
        removed_terms = report_df.loc[~kept_mask, "term"].tolist()

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_features": len(feature_names),
            "kept_features_count": len(kept_terms),
            "removed_features_count": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms
        }

        # store_result accepts full_report DataFrame for downstream visualization
        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=report_df
        )

        return summary

    return [variance_filter_tool]
