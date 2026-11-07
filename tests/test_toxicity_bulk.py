import unittest
import numpy as np
import pandas as pd
from toxicity_bulk import score_comments


class BulkScoringTests(unittest.TestCase):
    def test_blanks_preserved_and_other_columns_unchanged(self):
        data = pd.DataFrame({'id': [3, 4, 5, 6], 'comment_text': ['hello', None, '  ', 'rude']})
        seen = []
        def predict(texts):
            seen.extend(texts)
            return np.array([[.05, .1], [.7, .05]])
        out, count = score_comments(data, ['toxic', 'threat'], .5, predict)
        self.assertEqual(seen, ['hello', 'rude'])
        self.assertEqual(count, 2)
        self.assertEqual(out['id'].tolist(), [3, 4, 5, 6])
        self.assertFalse(out.loc[0, 'is_toxic'])
        self.assertTrue(out.loc[3, 'is_toxic'])
        self.assertTrue(pd.isna(out.loc[1, 'is_toxic']))
        self.assertTrue(pd.isna(out.loc[2, 'prob_toxic']))

    def test_all_blank_never_calls_model(self):
        def unexpected(_): raise AssertionError('model must not run')
        out, count = score_comments(pd.DataFrame({'comment_text': [None, ' ']}), ['toxic'], .5, unexpected)
        self.assertEqual(count, 0)
        self.assertTrue(out['is_toxic'].isna().all())

    def test_mismatched_model_output_is_rejected(self):
        with self.assertRaises(ValueError):
            score_comments(pd.DataFrame({'comment_text': ['hello']}), ['toxic'], .5, lambda _: np.zeros((2, 1)))


if __name__ == '__main__': unittest.main()
