"""Protect the Hugging Face Docker metadata and runtime wiring."""
import json
from pathlib import Path
import re
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class SpaceConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")
        cls.dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    def metadata(self):
        match = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|$)", self.readme, re.S)
        self.assertIsNotNone(match, "README.md must start with the Spaces YAML header")
        return yaml.safe_load(match.group(1))

    def entrypoint(self):
        match = re.search(r"ENTRYPOINT\s+(\[.*?\])", self.dockerfile, re.S)
        self.assertIsNotNone(match, "A Docker entry point is required")
        return json.loads(re.sub(r"\\\r?\n\s*", "", match.group(1)))

    def test_readme_declares_docker_space(self):
        metadata = self.metadata()
        for field in ("title", "emoji", "colorFrom", "colorTo", "sdk", "app_port", "pinned"):
            self.assertIn(field, metadata)
        self.assertEqual(metadata["sdk"], "docker")
        self.assertIs(type(metadata["app_port"]), int)
        self.assertIs(type(metadata["pinned"]), bool)

    def test_public_port_and_application_match_docker(self):
        port = self.metadata()["app_port"]
        command = self.entrypoint()
        self.assertIn(f"--server.port={port}", command)
        self.assertIn("--server.address=0.0.0.0", command)
        self.assertRegex(self.dockerfile, rf"(?m)^EXPOSE\s+{port}\s*$")
        self.assertIn(f"http://localhost:{port}/_stcore/health", self.dockerfile)
        self.assertTrue((ROOT / command[2]).is_file())

    def test_container_uses_owned_paths_for_uid_1000(self):
        self.assertIn("--uid 1000 appuser", self.dockerfile)
        self.assertIn("chown appuser:appuser /app", self.dockerfile)
        self.assertIn("COPY --chown=appuser:appuser src/", self.dockerfile)
        self.assertRegex(self.dockerfile, r"(?m)^USER appuser\s*$")

    def test_embedded_space_keeps_xsrf_with_cross_site_cookie_support(self):
        command = self.entrypoint()
        self.assertIn("--server.enableXsrfProtection=true",command)
        self.assertIn("--server.enableCORS=true",command)
        self.assertIn("--server.xsrfCookieSameSite=none",command)
        self.assertIn("--server.corsAllowedOrigins=https://huggingface.co",command)
        self.assertIn("--server.corsAllowedOrigins=https://syedmohaiminulhoque-agentic-doc-sim-streamlit.hf.space",command)
        self.assertIn("--browser.serverAddress=syedmohaiminulhoque-agentic-doc-sim-streamlit.hf.space",command)
        self.assertIn("--browser.serverPort=443",command)


if __name__ == "__main__":
    unittest.main()
