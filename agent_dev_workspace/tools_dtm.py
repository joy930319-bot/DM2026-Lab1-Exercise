from typing import Dict, Any, List, Union
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

    return [build_dtm_tool]
