"""Test public-data boundaries and invariants without GIS dependencies or network access."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from scripts import site_map_data as data


class SiteMapDataTests(unittest.TestCase):
    def setUp(self):
        self.sites = json.loads((data.OUTPUT / "estuary-sites.geojson").read_text())

    def test_committed_data_and_provenance(self):
        data.check()

    def test_rejects_duplicate_or_missing_site(self):
        duplicate = copy.deepcopy(self.sites)
        duplicate["features"][-1] = duplicate["features"][0]
        with self.assertRaisesRegex(ValueError, "Duplicate site IDs"):
            data.validate(duplicate)
        self.sites["features"].pop()
        with self.assertRaisesRegex(ValueError, "exactly 66"):
            data.validate(self.sites)

    def test_rejects_coordinate_errors(self):
        for coordinates in ([35, -120], [-120, float("nan")], [-120], [True, 35]):
            with self.subTest(coordinates=coordinates):
                invalid = copy.deepcopy(self.sites)
                invalid["features"][0]["geometry"]["coordinates"] = coordinates
                with self.assertRaises(ValueError):
                    data.validate(invalid)

    def test_rejects_extra_properties(self):
        self.sites["features"][0]["properties"]["internal_path"] = "/private/example"
        with self.assertRaisesRegex(ValueError, "allowlist"):
            data.validate(self.sites)

    def test_rejects_changed_coordinate_even_with_updated_provenance(self):
        self.sites["features"][0]["geometry"]["coordinates"][0] += 0.001
        output = data.encode(self.sites)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "estuary-sites.geojson").write_bytes(output)
            (path / "provenance.json").write_bytes(data.encode(data.provenance(output)))
            with self.assertRaisesRegex(ValueError, "pinned release subset"):
                data.check(path)

    def test_rejects_unverified_archive_without_writing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = path / "source.zip"
            source.write_bytes(b"wrong archive")
            with self.assertRaisesRegex(ValueError, "Archive SHA256"):
                data.generate(source, path / "output")
            self.assertFalse((path / "output").exists())

    def test_rejects_malformed_geometry_and_identifiers(self):
        for field, value in (("geometry", None), ("id", True), ("properties", [])):
            with self.subTest(field=field):
                invalid = copy.deepcopy(self.sites)
                invalid["features"][0][field] = value
                with self.assertRaises(ValueError):
                    data.validate(invalid)


if __name__ == "__main__":
    unittest.main()
