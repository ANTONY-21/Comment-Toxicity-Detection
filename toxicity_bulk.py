"""Prepare bulk toxicity predictions while preserving blank input rows."""
import numpy as np
import pandas as pd


def score_comments(frame, labels, threshold, predictor):
    result = frame.copy()
    valid = result['comment_text'].notna() & result['comment_text'].astype(str).str.strip().ne('')
    for label in labels:
        result[f'prob_{label}'] = np.nan
    result['is_toxic'] = pd.Series(pd.NA, index=result.index, dtype='boolean')
    if valid.any():
        scores = np.asarray(predictor(result.loc[valid, 'comment_text'].astype(str).tolist()))
        expected = (int(valid.sum()), len(labels))
        if scores.shape != expected:
            raise ValueError(f'Model returned {scores.shape}; expected {expected}.')
        for i, label in enumerate(labels):
            result.loc[valid, f'prob_{label}'] = scores[:, i].round(4)
        result.loc[valid, 'is_toxic'] = (scores >= threshold).any(axis=1)
    return result, int(valid.sum())
