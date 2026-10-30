import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from evaluate import load_audio, windows, summary


class EvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (Path(__file__).parent / '.cache').mkdir(exist_ok=True)

    def test_windows_match_stream_context_and_drop_incomplete_tail(self):
        audio = np.arange(80000, dtype=np.float32)
        actual = list(windows(audio))
        self.assertEqual(7, len(actual))
        np.testing.assert_array_equal(actual[0], audio[:31200])
        np.testing.assert_array_equal(actual[-1], audio[46800:78000])
        self.assertEqual([], list(windows(audio[:31199])))

    def test_pcm_scale_preserves_volume(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / '.cache') as folder:
            path = Path(folder) / 'audio.wav'
            wavfile.write(path, 16000, np.array([-32768, 0, 16384], dtype=np.int16))
            np.testing.assert_allclose(load_audio(path), [-1, 0, .5])

    def test_resampling_preserves_duration(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / '.cache') as folder:
            path = Path(folder) / 'audio.wav'
            wavfile.write(path, 44100, np.zeros(220500, dtype=np.int16))
            self.assertEqual(80000, len(load_audio(path)))

    def test_recall_and_nominal_false_positives_use_correct_denominators(self):
        records = [
            dict(category='siren', peak_scores={'test': [0, .5, 0, 0, 0, 0]}),
            dict(category='siren', peak_scores={'test': [0, .49, 0, 0, 0, 0]}),
            dict(category='dog', peak_scores={'test': [0, .6, 0, 0, 0, 0]}),
        ]
        result = summary(records, 'test')
        self.assertEqual(.5, result['SIREN']['recall'])
        self.assertEqual(1, result['SIREN']['nominal_clip_fpr'])
        self.assertIsNone(result['DOORBELL']['recall'])


if __name__ == '__main__':
    unittest.main()
