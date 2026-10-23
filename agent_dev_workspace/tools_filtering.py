from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy.stats import rankdata
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

    @tool
    def pearson_filter_tool(
        target_class: str,
        top_n: int = 20
    ) -> Dict[str, Any]:
        """Computes one-vs-rest Pearson correlation between each term and target_class.

        Vectorized correlation computation across session.feature_matrix against binary target_class.
        Sorts results by absolute correlation value descending for full_report and summary.

        Args:
            target_class: Category name to correlate terms against.
            top_n: Number of top correlated terms to include in summary (default 20).

        Returns:
            Dict containing result_id, target_class, total_features, and top_terms with pearson_r.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "session.feature_matrix and session.feature_names must be available. "
                "Run build_dtm_tool first."
            )

        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values

        if labels is None:
            raise ValueError("Document labels must be available in session (or session.dataframe) to calculate class correlation.")

        if target_class not in np.unique(labels):
            raise ValueError(f"Target class '{target_class}' not found in labels: {np.unique(labels).tolist()}")

        # Binary one-vs-rest array
        y = (np.asarray(labels) == target_class).astype(float)
        y_mean = y.mean()
        y_std = y.std()

        if y_std == 0:
            raise ValueError("Target class indicator has zero std (either present in all rows or no rows).")

        # Vectorized Pearson correlation calculation:
        # cov(X, y) = E[X * y] - E[X] * E[y]
        # X is (N, M) matrix, y is (N,) vector
        # X.T.dot(y) gives sum(X_i * y) per column
        matrix = session.feature_matrix
        n_samples = matrix.shape[0]

        X_mean = np.asarray(matrix.mean(axis=0)).ravel()
        X_std = np.sqrt(np.maximum(0, np.asarray(matrix.power(2).mean(axis=0)).ravel() - X_mean ** 2))

        X_y_mean = np.asarray(matrix.T.dot(y)).ravel() / n_samples
        cov = X_y_mean - X_mean * y_mean

        # Avoid division by zero for constant terms
        with np.errstate(divide='ignore', invalid='ignore'):
            pearson_r = cov / (X_std * y_std)
            pearson_r = np.nan_to_num(pearson_r, nan=0.0, posinf=0.0, neginf=0.0)

        feature_names = session.feature_names

        full_report = pd.DataFrame({
            "term": feature_names,
            "pearson_r": pearson_r,
            "abs_r": np.abs(pearson_r)
        }).sort_values(by="abs_r", ascending=False, kind="stable").reset_index(drop=True)

        # Drop temporary abs_r column so full_report has term as 1st col, pearson_r as 2nd col
        full_report_export = full_report[["term", "pearson_r"]]

        top_df = full_report_export.head(top_n)
        top_terms = {row["term"]: float(round(row["pearson_r"], 4)) for _, row in top_df.iterrows()}

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "total_features": len(feature_names),
            "top_terms": top_terms
        }

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class, "top_n": top_n},
            summary=summary,
            full_report=full_report_export
        )

        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        top_n: int = 20
    ) -> Dict[str, Any]:
        """Computes one-vs-rest Spearman rank correlation between terms and target_class.

        Processes terms in chunks of 1000, ranking column values with average tie handling,
        and computes Pearson correlation against binary target_class.
        Sorts results by absolute correlation value descending.

        Args:
            target_class: Category name to correlate terms against.
            top_n: Number of top correlated terms to include in summary (default 20).

        Returns:
            Dict containing result_id, target_class, total_features, and top_terms with spearman_r.
        """
        if session.feature_matrix is None or session.feature_names is None:
            raise ValueError(
                "session.feature_matrix and session.feature_names must be available. "
                "Run build_dtm_tool first."
            )

        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values

        if labels is None:
            raise ValueError("Document labels must be available in session (or session.dataframe) to calculate class correlation.")

        if target_class not in np.unique(labels):
            raise ValueError(f"Target class '{target_class}' not found in labels: {np.unique(labels).tolist()}")

        y = (np.asarray(labels) == target_class).astype(float)
        y_mean = y.mean()
        y_std = y.std()

        if y_std == 0:
            raise ValueError("Target class indicator has zero std (either present in all rows or no rows).")

        matrix = session.feature_matrix
        n_samples, n_features = matrix.shape
        chunk_size = 1000
        spearman_r = np.zeros(n_features, dtype=float)

        for start in range(0, n_features, chunk_size):
            end = min(start + chunk_size, n_features)
            # Convert chunk to dense for rankdata computation
            X_chunk_dense = matrix[:, start:end].toarray()

            # Rank each column using average method for ties
            X_ranked = rankdata(X_chunk_dense, method="average", axis=0)

            X_mean = X_ranked.mean(axis=0)
            X_std = X_ranked.std(axis=0)

            cov = np.mean(X_ranked * y[:, None], axis=0) - X_mean * y_mean

            with np.errstate(divide='ignore', invalid='ignore'):
                r_chunk = cov / (X_std * y_std)
                r_chunk = np.nan_to_num(r_chunk, nan=0.0, posinf=0.0, neginf=0.0)

            spearman_r[start:end] = r_chunk

        feature_names = session.feature_names

        full_report = pd.DataFrame({
            "term": feature_names,
            "spearman_r": spearman_r,
            "abs_r": np.abs(spearman_r)
        }).sort_values(by="abs_r", ascending=False, kind="stable").reset_index(drop=True)

        full_report_export = full_report[["term", "spearman_r"]]

        top_df = full_report_export.head(top_n)
        top_terms = {row["term"]: float(round(row["spearman_r"], 4)) for _, row in top_df.iterrows()}

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "total_features": len(feature_names),
            "top_terms": top_terms
        }

        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class, "top_n": top_n},
            summary=summary,
            full_report=full_report_export
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
