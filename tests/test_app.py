import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_FILE = PROJECT_ROOT / "app.py"
DATA_FILE = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


class CustomerChurnAppTests(unittest.TestCase):
    def create_app(self) -> AppTest:
        return AppTest.from_file(str(APP_FILE)).run(timeout=30)

    def test_pages_render(self):
        app = self.create_app()
        self.assertFalse(app.exception)

        pages = [
            "Estimate a customer",
            "Batch scoring",
            "Model performance",
            "Explore the data",
            "Overview",
        ]
        for page in pages:
            app.radio[0].set_value(page).run(timeout=30)
            self.assertFalse(app.exception, f"{page}: {app.exception}")

    def test_batch_scoring_adds_probability_results(self):
        source = pd.read_csv(DATA_FILE).head(2)
        app = self.create_app()
        app.radio[0].set_value("Batch scoring").run(timeout=30)
        app.file_uploader[0].upload(
            "customers.csv",
            source.to_csv(index=False).encode("utf-8"),
            "text/csv",
        ).run(timeout=30)

        self.assertFalse(app.exception)
        metric_labels = {metric.label for metric in app.metric}
        self.assertIn("Customers scored", metric_labels)
        self.assertIn("Average churn probability", metric_labels)

    def test_batch_scoring_reports_missing_columns(self):
        app = self.create_app()
        app.radio[0].set_value("Batch scoring").run(timeout=30)
        app.file_uploader[0].upload(
            "incomplete.csv",
            b"customerID\n123\n",
            "text/csv",
        ).run(timeout=30)

        self.assertTrue(
            any("missing required model inputs" in error.value for error in app.error)
        )

    def test_exploration_handles_empty_filter_results(self):
        app = self.create_app()
        app.radio[0].set_value("Explore the data").run(timeout=30)
        app.multiselect[0].set_value([]).run(timeout=30)

        self.assertTrue(
            any("No customers match these filters" in message.value for message in app.info)
        )


if __name__ == "__main__":
    unittest.main()
