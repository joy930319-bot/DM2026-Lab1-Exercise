import os
import tempfile
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def make_tools(session: Any) -> list:
    """Factory function returning tools for frequent pattern mining bound to session."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str,
        algorithm: str = "fpgrowth",
        min_sup: int = 0,
        k: int = 0
    ) -> Dict[str, Any]:
        """
        Mine frequent term patterns for a specific category using PAMI algorithms.

        Args:
            category_name: Target category name to mine.
            filtering_method: Vocabulary filtering method ("variance", "tfidf", or "term_frequency").
            algorithm: Pattern mining algorithm ("fpgrowth", "topk", or "maxfpgrowth").
            min_sup: Minimum support threshold (required for fpgrowth/maxfpgrowth).
            k: Top-K pattern limit (required for topk).

        Returns:
            Dict containing result_id, category_name, filtering_method, algorithm, kept_terms, n_patterns, and top 20 patterns.
        """
        if session.dataframe is None:
            raise ValueError("No dataset loaded. Run load_dataset_tool first.")

        filt_method = filtering_method.lower()
        if filt_method not in ["variance", "tfidf", "term_frequency"]:
            raise ValueError(f"Invalid filtering_method '{filtering_method}'. Expected one of 'variance', 'tfidf', 'term_frequency'.")

        algo = algorithm.lower()
        if algo not in ["fpgrowth", "topk", "maxfpgrowth"]:
            raise ValueError(f"Invalid algorithm '{algorithm}'. Expected one of 'fpgrowth', 'topk', 'maxfpgrowth'.")

        if algo in ["fpgrowth", "maxfpgrowth"] and min_sup < 1:
            raise ValueError(f"min_sup must be at least 1 for {algo}, got {min_sup}.")
        if algo == "topk" and k < 1:
            raise ValueError(f"k must be at least 1 for topk algorithm, got {k}.")

        # Filter dataframe to target category
        df = session.dataframe
        cat_df = df[df["category_name"] == category_name]
        if len(cat_df) == 0:
            raise ValueError(f"Category '{category_name}' not found or has no documents.")

        texts = cat_df["text"].fillna("").astype(str).tolist()

        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm_sparse = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
        except ValueError:
            # Handle empty vocabulary case
            dtm_sparse = None
            feature_names = np.array([])

        artifact_key = f"patterns_{category_name}_{filt_method}_{algo}"

        if dtm_sparse is None or len(feature_names) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filt_method,
                "algorithm": algo,
                "kept_terms": [],
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm, "min_sup": min_sup, "k": k}, summary)
            return summary

        counts_dense = dtm_sparse.toarray()
        counts_df = pd.DataFrame(counts_dense, columns=feature_names)

        # Apply filtering method matching text_filtering_helpers.py
        if filt_method == "variance":
            var_series = counts_df.var(axis=0, ddof=1)
            lower = np.percentile(var_series, 5)
            upper = np.percentile(var_series, 95)
            kept_terms = [str(t) for t in var_series[(var_series >= lower) & (var_series <= upper)].index.tolist()]

        elif filt_method == "term_frequency":
            tf_series = counts_df.sum(axis=0)
            lower = np.percentile(tf_series, 5)
            upper = np.percentile(tf_series, 95)
            kept_terms = [str(t) for t in tf_series[(tf_series >= lower) & (tf_series <= upper)].index.tolist()]

        elif filt_method == "tfidf":
            tfidf_mat = TfidfTransformer().fit_transform(dtm_sparse)
            mean_tfidf = np.array(tfidf_mat.mean(axis=0)).ravel()
            p20 = np.percentile(mean_tfidf, 20)
            kept_terms = [str(feature_names[i]) for i in range(len(feature_names)) if mean_tfidf[i] >= p20]

        if len(kept_terms) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filt_method,
                "algorithm": algo,
                "kept_terms": [],
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm, "min_sup": min_sup, "k": k}, summary)
            return summary

        # Build presence DataFrame for surviving terms
        presence_df = (counts_df[kept_terms] >= 1).astype(int)

        patterns = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_db_path = os.path.join(tmp_dir, "transactional.db")

            from PAMI.extras.convert import DF2DB
            converter = DF2DB.DF2DB(presence_df)
            converter.convert2TransactionalDatabase(tmp_db_path, ">=", 1)

            if algo == "fpgrowth":
                from PAMI.frequentPattern.basic import FPGrowth
                miner = FPGrowth.FPGrowth(iFile=tmp_db_path, minSup=min_sup, sep="\t")
            elif algo == "topk":
                from PAMI.frequentPattern.topk import FAE
                miner = FAE.FAE(iFile=tmp_db_path, k=k, sep="\t")
            elif algo == "maxfpgrowth":
                from PAMI.frequentPattern.maximal import MaxFPGrowth
                miner = MaxFPGrowth.MaxFPGrowth(iFile=tmp_db_path, minSup=min_sup, sep="\t")

            miner.mine()
            raw_patterns = miner.getPatterns()

            if raw_patterns:
                for item_key, sup_val in raw_patterns.items():
                    if isinstance(item_key, (list, tuple)):
                        items = sorted([str(x) for x in item_key])
                    elif isinstance(item_key, str):
                        items = sorted([str(x) for x in item_key.split()])
                    else:
                        items = [str(item_key)]

                    if isinstance(sup_val, (list, tuple)):
                        sup_int = int(sup_val[0])
                    else:
                        sup_int = int(sup_val)
                    patterns.append({"items": items, "support": sup_int})

        # Sort patterns by support descending
        patterns.sort(key=lambda x: x["support"], reverse=True)

        session.artifacts[artifact_key] = patterns

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filt_method,
            "algorithm": algo,
            "kept_terms": sorted(kept_terms),
            "n_patterns": len(patterns),
            "patterns": patterns[:20]
        }

        args = {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k
        }
        session.store_result("mine_patterns_tool", args, summary)
        return summary

    return [mine_patterns_tool]
