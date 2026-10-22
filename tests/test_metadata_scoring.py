import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.meta_agent import MetaAgent
from models.document import MetadataExtraction
from orchestrator.scorers import compute_metadata_similarity


class MetadataScoringTests(unittest.TestCase):
    def test_identical_metadata_never_exceeds_one(self):
        for fields in (
            {"title": "Project summary"},
            {"author": "Analyst"},
            {"title": "Project summary", "author": "Analyst", "page_count": 2},
        ):
            with self.subTest(fields=fields):
                metadata = MetadataExtraction(**fields)
                report = compute_metadata_similarity(metadata, metadata)
                self.assertAlmostEqual(report.score, 1.0)
                self.assertAlmostEqual(MetaAgent().compute_metadata_similarity(metadata, metadata), 1.0)
                for match in report.matched_items:
                    self.assertLessEqual(match["similarity"], 1.0)

    def test_matching_title_and_different_page_counts_are_normalized(self):
        first = MetadataExtraction(title="Project summary", page_count=2)
        second = MetadataExtraction(title="Project summary", page_count=4)
        self.assertAlmostEqual(compute_metadata_similarity(first, second).score, 5 / 6)
        self.assertAlmostEqual(MetaAgent().compute_metadata_similarity(first, second), 5 / 6)

    def test_missing_metadata_returns_zero(self):
        self.assertEqual(compute_metadata_similarity(MetadataExtraction(), MetadataExtraction()).score, 0.0)
        self.assertEqual(MetaAgent().compute_metadata_similarity(MetadataExtraction(), MetadataExtraction()), 0.0)


if __name__ == "__main__":
    unittest.main()
