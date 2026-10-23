from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def make_tools(session: Any) -> list:
    """Factory function returning tools for dimensionality reduction and label processing bound to session."""

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """
        One-hot encode document category labels into a binary matrix.

        Returns:
            Dict containing result_id, classes, shape, counts_per_class, and first_rows.
        """
        labels = None
        if session.labels is not None:
            labels = session.labels
        elif session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].tolist()

        if labels is None or len(labels) == 0:
            raise ValueError("No label data found in session. Load dataset or set labels first.")

        classes = sorted(list(set(labels)))
        labels_arr = np.array(labels)

        # Build one-hot encoding matrix explicitly to avoid 1D output for 2 classes
        matrix = (labels_arr[:, None] == np.array(classes)).astype(int)

        session.artifacts["binarized_labels"] = matrix
        session.artifacts["binarized_classes"] = classes

        counts_per_class = {cls: int(count) for cls, count in zip(classes, matrix.sum(axis=0))}
        first_rows = matrix[:5].tolist()

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "classes": classes,
            "shape": list(matrix.shape),
            "counts_per_class": counts_per_class,
            "first_rows": first_rows
        }

        session.store_result(
            tool_name="binarize_labels_tool",
            args={},
            summary=summary
        )

        return summary

    @tool
    def reduce_dimensions_tool(
        method: str = "pca",
        n_components: int = 2,
        random_state: int = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15
    ) -> Dict[str, Any]:
        """
        Reduce dimensionality of the document-term matrix using PCA, t-SNE, or UMAP, and plot the result.

        Args:
            method: Reduction method ("pca", "tsne", or "umap").
            n_components: Number of dimensions (minimum 2).
            random_state: Random seed for reproducibility.
            perplexity: Perplexity parameter for t-SNE.
            n_neighbors: Number of neighbors parameter for UMAP.

        Returns:
            Dict containing result_id, method, n_components, n_documents, coordinates_preview, and optional explained_variance_ratio.
        """
        method_lower = method.lower()
        if method_lower not in ["pca", "tsne", "umap"]:
            raise ValueError(f"Invalid method '{method}'. Expected one of 'pca', 'tsne', 'umap'.")

        if n_components < 2:
            raise ValueError(f"n_components must be at least 2 for plotting, got {n_components}.")

        if session.feature_matrix is None:
            raise ValueError("No document-term matrix found. Run build_dtm_tool first.")

        matrix_dense = session.feature_matrix.toarray()
        n_docs = matrix_dense.shape[0]

        explained_variance_ratio = None

        if method_lower == "pca":
            pca = PCA(n_components=n_components, random_state=random_state)
            coords = pca.fit_transform(matrix_dense)
            explained_variance_ratio = pca.explained_variance_ratio_.tolist()
        elif method_lower == "tsne":
            if perplexity >= n_docs:
                raise ValueError(f"perplexity ({perplexity}) must be less than number of samples ({n_docs}).")
            tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
            coords = tsne.fit_transform(matrix_dense)
        elif method_lower == "umap":
            import umap
            reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
            coords = reducer.fit_transform(matrix_dense)

        session.artifacts["reduced_coordinates"] = coords

        # Get category labels for hue
        hue_labels = None
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            hue_labels = session.dataframe["category_name"]
        elif session.labels is not None:
            hue_labels = session.labels

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=hue_labels, ax=ax, palette="tab10")
        ax.set_title(f"Dimensionality Reduction ({method_lower.upper()})")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        plt.tight_layout()

        session.pending_figure = fig

        coords_list = coords.tolist()
        coords_preview = coords_list[:10]

        # Prepare DataFrame report for full_report
        col_names = [f"dim_{i+1}" for i in range(n_components)]
        report_df = pd.DataFrame(coords, columns=col_names)

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method_lower,
            "n_components": n_components,
            "n_documents": n_docs,
            "coordinates_preview": coords_preview
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        args = {
            "method": method,
            "n_components": n_components,
            "random_state": random_state,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors
        }

        session.store_result("reduce_dimensions_tool", args, summary, full_report=report_df)
        return summary

    return [binarize_labels_tool, reduce_dimensions_tool]
