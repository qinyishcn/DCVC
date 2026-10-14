"""Regression tests for process handling and user-video report generation."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import run_rd_comparison as rd


class ProcessTests(unittest.TestCase):
    def test_stdin_is_closed(self):
        result = rd.run([sys.executable, "-c", "import sys; print(len(sys.stdin.read()))"], timeout=5)
        self.assertEqual(result.strip(), b"0")

    def test_large_stdout_and_stderr_do_not_block(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "output.log"
            result = rd.run([sys.executable, "-c",
                             "import sys; sys.stdout.write('x'*2000000); sys.stderr.write('y'*2000000)"],
                            log, timeout=10)
            self.assertEqual(len(result), 2000000)
            self.assertEqual(log.stat().st_size, 2000000)

    def test_failed_command_preserves_diagnostic_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "error.log"
            with self.assertRaisesRegex(RuntimeError, "bad MP4"):
                rd.run([sys.executable, "-c", "import sys; sys.stderr.write('bad MP4'); sys.exit(3)"], log)
            self.assertEqual(log.read_text(), "bad MP4")

    def test_idle_command_is_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            start = time.monotonic()
            with self.assertRaisesRegex(TimeoutError, "no log/output activity"):
                rd.run([sys.executable, "-c", "import time; time.sleep(30)"],
                       Path(directory) / "idle.log", timeout=10, idle_timeout=0.3)
            self.assertLess(time.monotonic() - start, 8)

    def test_total_timeout_even_when_output_is_active(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TimeoutError, "command timeout"):
                rd.run([sys.executable, "-u", "-c",
                        "import time\nwhile True:\n print('still alive',flush=True)\n time.sleep(0.05)"],
                       Path(directory) / "active.log", timeout=0.4, idle_timeout=5)


class ReportTests(unittest.TestCase):
    def test_custom_video_without_overlap_or_444_curve(self):
        points = [{"codec": "DCVC-RT all-I", "quality": 16, "frames": 3,
                   "bitrate_kbps": 20, "psnr_rgb_pooled_db": 30, "psnr_rgb_mean_frame_db": 31},
                  {"codec": "H.264 4:2:0 inter", "quality": 30, "frames": 3,
                   "bitrate_kbps": 100, "psnr_rgb_pooled_db": 29, "psnr_rgb_mean_frame_db": 30}]
        env = {"source_name": "我的 视频.mp4", "reference_frames": 3, "width": 96, "height": 64,
               "fps": 30, "fps_rational": "30", "duration_seconds": 0.1,
               "source_sha256": "source", "reference_sha256": "reference",
               "dcvc_environment": {"checkpoint_sha256": "checkpoint"}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rd.make_report(points, env, output)
            rd.make_plots(points, output, env)
            report = (output / "README.md").read_text(encoding="utf-8")
            self.assertIn("我的 视频.mp4", report)
            self.assertIn("没有重叠", report)
            self.assertNotIn("Starigrad", report)
            self.assertNotIn("高于 DCVC", report)
            self.assertTrue((output / "rd_curve.png").is_file())

    def test_completed_mp4_commands_use_legacy_sync(self):
        # The integration runner saves its argv; check those when a test run is
        # available without requiring a GPU for the unit-only test suite.
        path = Path(__file__).parents[1] / "results/mp4_rd_compatibility_20261014/modern/commands.json"
        if not path.exists():
            self.skipTest("run the MP4 integration smoke test first")
        commands = json.loads(path.read_text(encoding="utf-8"))
        ffmpeg = [row["argv"] for row in commands if row["purpose"] not in
                  ("DCVC native all-I encode/decode", "DCVC environment check")]
        self.assertTrue(ffmpeg)
        for argv in ffmpeg:
            self.assertNotIn("-fps_mode", argv)
            self.assertIn("-nostdin", argv)
            self.assertIn("-vsync", argv)
            self.assertEqual(argv[argv.index("-vsync") + 1], "0")


if __name__ == "__main__":
    unittest.main()
