import importlib.util
from pathlib import Path
import tempfile
import unittest

class ReleaseTests(unittest.TestCase):
    def test_private_artifacts_excluded_and_release_blockers_explicit(self):
        spec=importlib.util.spec_from_file_location("release_check",Path(__file__).resolve().parents[1]/"scripts/verify_release.py")
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"source.py").write_text("print('public')")
            (root/"private.pt").write_bytes(b"private")
            (root/".local_deps311").mkdir();(root/".local_deps311/secret.py").write_text("private")
            result=module.scan(root)
            self.assertEqual(set(result["public_source_sha256"]),{"source.py"})
            self.assertFalse(result["release_ready"])
            self.assertIn("license_unconfirmed",[i["kind"] for i in result["issues"]])
            self.assertIn("git_index_unavailable",[i["kind"] for i in result["issues"]])

if __name__=="__main__":unittest.main()
