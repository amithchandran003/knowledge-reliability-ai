"""Focused tests for V3 web-source registration."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from source_registry import main, register_web_source, validate_web_url


class SourceRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.registry_path = os.path.join(self.temporary_directory.name, "sources.json")
        with open(self.registry_path, "w", encoding="utf-8") as file:
            json.dump({"sources": []}, file)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_valid_url_is_canonicalized_without_fragment(self):
        self.assertEqual(validate_web_url("https://Example.COM/refunds#details"), "https://example.com/refunds")

    def test_duplicate_url_is_rejected(self):
        register_web_source("Refund Policy", "https://example.com/refunds", self.registry_path)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_web_source("Duplicate", "https://EXAMPLE.com/refunds#section", self.registry_path)

    def test_invalid_protocol_and_unsafe_hosts_are_rejected(self):
        for url in (
            "ftp://example.com/file", "https://localhost/policy", "http://127.0.0.1/policy",
            "http://192.168.1.10/policy", "http://169.254.1.10/policy",
            "https://user:password@example.com/policy", "https://example.com/policy?api_key=secret",
            "https://example.com/policy?token=secret",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_web_url(url)

    def test_list_and_show_commands(self):
        record = register_web_source("Company Refund Policy", "https://example.com/refunds", self.registry_path)
        output = io.StringIO()
        with redirect_stdout(output):
            main(["list"], self.registry_path)
        self.assertIn(record["source_id"], output.getvalue())
        self.assertIn("Company Refund Policy", output.getvalue())

        output = io.StringIO()
        with redirect_stdout(output):
            main(["show", record["source_id"]], self.registry_path)
        self.assertEqual(json.loads(output.getvalue())["source_url"], "https://example.com/refunds")


if __name__ == "__main__":
    unittest.main()
