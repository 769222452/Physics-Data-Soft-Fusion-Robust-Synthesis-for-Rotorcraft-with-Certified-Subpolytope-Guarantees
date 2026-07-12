import csv
import contextlib
import hashlib
import io
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.linalg as la


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
TABLES = ROOT / "results" / "tables"
sys.path.insert(0, str(ROOT / "src"))

import time_domain_standard as model  # noqa: E402


def _scalar(array):
    return float(np.asarray(array).reshape(-1)[0])


def _display_tolerance(text):
    value = text.strip().lower()
    if "e" in value:
        mantissa, exponent = value.split("e")
        decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
        return 0.5 * 10.0 ** (int(exponent) - decimals) + 1e-14
    decimals = len(value.split(".")[1]) if "." in value else 0
    return 0.5 * 10.0 ** (-decimals) + 1e-14


class ReleasedArtifactTests(unittest.TestCase):
    def test_release_manifest_matches_files(self):
        manifest = ROOT / "results" / "manifest.sha256"
        entries = []
        for line in manifest.read_text(encoding="ascii").splitlines():
            digest, relative = line.split("  ", 1)
            entries.append(relative)
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(digest, actual, relative)
        expected = [
            path.relative_to(ROOT).as_posix()
            for base in (ROOT / "results", ROOT / "logs")
            for path in base.rglob("*")
            if path.is_file() and path.name != "manifest.sha256"
        ]
        self.assertEqual(sorted(entries), sorted(expected))

    def test_all_npz_files_are_pickle_free(self):
        for path in sorted(RAW.glob("*.npz")):
            with self.subTest(path=path.name):
                data = np.load(path, allow_pickle=False)
                for key in data.files:
                    self.assertNotEqual(data[key].dtype.kind, "O", key)

    def test_time_domain_tables_match_raw_archives(self):
        cases = (
            ("time_domain_standard_metrics.csv",
             "stage3_time_domain_results.npz"),
            ("time_domain_expanded_metrics.csv",
             "stage3_time_domain_results-ex.npz"),
        )
        prefixes = {
            "Proposed": "Proposed_",
            "Baseline B": "BaselineB_NoRelax__",
            "Nominal LQR": "NominalLQR_",
        }
        for table_name, raw_name in cases:
            raw = np.load(RAW / raw_name, allow_pickle=False)
            with (TABLES / table_name).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            for row in rows:
                prefix = prefixes[row["Controller"]]
                for field, text in row.items():
                    if field == "Controller":
                        continue
                    if field == "gamma":
                        if row["Controller"] == "Nominal LQR":
                            self.assertTrue(math.isnan(float(text)))
                            continue
                        key = "gamma_proposed" if row["Controller"] == "Proposed" else "gamma_baselineB"
                    else:
                        key = prefix + field
                    actual = _scalar(raw[key])
                    displayed = float(text)
                    self.assertLessEqual(
                        abs(displayed - actual),
                        _display_tolerance(text),
                        f"{table_name}: {row['Controller']} {field}",
                    )

    def test_fusion_summary_matches_monte_carlo_archive(self):
        raw = np.load(RAW / "fusion_ablation_monte_carlo_raw.npz",
                      allow_pickle=False)
        with (TABLES / "fusion_ablation_monte_carlo_summary.csv").open(
                newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        raw_names = {
            "NominalLQR": "NominalLQR",
            "P0(PriorOnly)": "P0_PriorOnly",
            "P1(NoRelax)": "P1_NoRelax",
            "F1a(HardStrict)": "F1a_HardStrict",
            "F1b(HardBudget)": "F1b_HardBudget",
            "S0(DataOnly)": "S0_DataOnly",
            "Proposed": "Proposed",
        }
        for row in rows:
            name = raw_names[row["Controller"]]
            success = raw[f"{name}_Success"].astype(bool)
            rmse_all = raw[f"{name}_RMSE"]
            valid = np.isfinite(rmse_all)
            rmse = rmse_all[valid]
            peak = raw[f"{name}_Peak"][valid]
            expected = {
                "SuccessRatePct": 100.0 * np.mean(success),
                "RMSEMean": np.mean(rmse),
                "RMSEMedian": np.median(rmse),
                "RMSEStd": np.std(rmse),
                "RMSE95": np.percentile(rmse, 95),
                "PeakMean": np.mean(peak),
                "PeakMax": np.max(peak),
                "EnergyMean": np.mean(raw[f"{name}_Energy"][valid]),
                "IncrementSatRateMean": np.mean(raw[f"{name}_IncrementSatRate"]),
                "AbsoluteSatRateMean": np.mean(raw[f"{name}_AbsoluteSatRate"]),
                "CommandPeakNormalizedMax": np.max(raw[f"{name}_CommandPeakNormalized"]),
                "AppliedPeakNormalizedMax": np.max(raw[f"{name}_AppliedPeakNormalized"]),
            }
            for field, actual in expected.items():
                self.assertAlmostEqual(float(row[field]), float(actual), places=12)

    def test_vertex_summary_matches_per_seed_archive(self):
        raw = np.load(RAW / "vertex_selection_ablation_raw.npz",
                      allow_pickle=False)
        with (TABLES / "vertex_selection_table8_summary.csv").open(
                newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 24)
        key_names = {
            "SFPS+ICE": "SFPS_ICE",
            "SFPS-init": "SFPS_init",
            "Random+ICE": "Random_ICE",
            "TopK+ICE": "TopK_ICE",
            "FPS-Param+ICE": "FPS_Param_ICE",
            "FPS-Delta+ICE": "FPS_Delta_ICE",
        }
        for row in rows:
            base = f"K{row['KBudget']}_{key_names[row['Strategy']]}"
            classes = raw[f"{base}_classification"]
            success = raw[f"{base}_synth_success"].astype(bool)
            valid = success
            values = {
                "SurrogatePassPct": 100.0 * np.mean(classes == "surrogate_pass"),
                "NearSurrogatePct": 100.0 * np.mean(classes == "near_surrogate"),
                "FailPct": 100.0 * np.mean(classes == "failed"),
                "MonteCarloSuccessMeanPct": np.mean(raw[f"{base}_mc_success_rate"]),
                "MonteCarloSuccessStdPct": np.std(raw[f"{base}_mc_success_rate"]),
            }
            conditional = {
                "GammaMeanSuccessful": raw[f"{base}_gamma"],
                "SDPRoundsMeanSuccessful": raw[f"{base}_n_rounds"],
                "RMSE95MeanSuccessful": raw[f"{base}_mc_rmse_95pct"],
            }
            for field, array in conditional.items():
                values[field] = np.mean(array[valid]) if np.any(valid) else np.nan
            violations = raw[f"{base}_max_violation"]
            values["MaxViolationSuccessful"] = (
                np.max(violations[valid]) if np.any(valid) else np.nan
            )
            for field, actual in values.items():
                reported = float(row[field])
                if np.isnan(actual):
                    self.assertTrue(np.isnan(reported), f"{base} {field}")
                else:
                    self.assertAlmostEqual(reported, float(actual), places=12)

    def test_data_qmi_scores_and_bounds_are_reproducible(self):
        names = (
            "standard_data_fusion_diagnostics.npz",
            "expanded_data_fusion_diagnostics.npz",
            "fusion_ablation_data_diagnostics.npz",
            "vertex_selection_data_diagnostics.npz",
        )
        identity = np.eye(32)
        for name in names:
            data = np.load(RAW / name, allow_pickle=False)
            psi = data["Psi"]
            self.assertLess(np.max(np.abs(psi - psi.T)), 1e-9)
            recalculated = []
            for delta in data["vertex_delta"]:
                multiplier = np.hstack((delta, identity))
                qmi = multiplier @ psi @ multiplier.T
                recalculated.append(la.eigvalsh(0.5 * (qmi + qmi.T))[-1])
            np.testing.assert_allclose(
                recalculated, data["raw_scores"], rtol=1e-10, atol=1e-7
            )
            self.assertEqual(int(np.sum(data["raw_scores"] <= 0.0)), 0)
            self.assertEqual(int(np.sum(data["processed_scores"] == 0.0)), 103)

            wc = data["W_c"]
            gaps = [
                la.eigvalsh(wc - matrix @ matrix.T)[0]
                for matrix in data["vertex_disturbance_injection"]
            ]
            self.assertGreaterEqual(min(gaps), -1e-10)

            true_successor = data["batch_X_phys"][:, 1:]
            recorded = data["batch_X_tp1_recorded_phys"]
            residual = np.zeros((32, recorded.shape[1]))
            inv_state_scale = np.diag(1.0 / data["state_scales"])
            residual[:12] = inv_state_scale @ (
                recorded[:12] - true_successor[:12]
            )
            self.assertGreaterEqual(
                la.eigvalsh(data["W_E"] - residual @ residual.T)[0],
                -1e-10,
            )

            regressors = np.vstack((data["batch_X_t"], data["batch_U_t"]))
            observations = np.vstack((data["batch_X_tp1"], data["batch_Z_t"]))
            generator_delta = np.block([
                [data["batch_A"], data["batch_B"]],
                [data["batch_Cc"], data["batch_Dc"]],
            ])
            aggregate = observations - generator_delta @ regressors
            length = regressors.shape[1]
            aggregate_bound = 2.0 * la.block_diag(
                length * data["W_c"], np.zeros((16, 16))
            ) + 2.0 * data["W_E"]
            self.assertGreaterEqual(
                la.eigvalsh(aggregate_bound - aggregate @ aggregate.T)[0],
                -1e-8,
            )

    def test_current_score_pipeline_reproduces_archived_scores(self):
        names = (
            "standard_data_fusion_diagnostics.npz",
            "expanded_data_fusion_diagnostics.npz",
            "fusion_ablation_data_diagnostics.npz",
            "vertex_selection_data_diagnostics.npz",
        )
        for name in names:
            data = np.load(RAW / name, allow_pickle=False)
            vertices = [{"s": float(value)} for value in data["raw_scores"]]
            with contextlib.redirect_stdout(io.StringIO()):
                processed, diagnostics = model.compute_si_from_vi(
                    vertices, L=data["batch_X_t"].shape[1]
                )
            np.testing.assert_allclose(
                [vertex["s"] for vertex in processed],
                data["processed_scores"],
                rtol=1e-12,
                atol=1e-12,
            )
            self.assertEqual(diagnostics["n_raw_consistent"], 0)
            self.assertEqual(diagnostics["n_zero_processed"], 103)

    def test_reported_time_domain_lmi_residuals_recompute(self):
        cases = (
            ("standard_data_fusion_diagnostics.npz",
             "stage3_time_domain_results.npz"),
            ("expanded_data_fusion_diagnostics.npz",
             "stage3_time_domain_results-ex.npz"),
        )
        for diagnostic_name, result_name in cases:
            diagnostic = np.load(RAW / diagnostic_name, allow_pickle=False)
            result = np.load(RAW / result_name, allow_pickle=False)
            scores = diagnostic["processed_scores"]
            deltas = diagnostic["vertex_delta"]
            injections = diagnostic["vertex_disturbance_injection"]
            for method in ("proposed", "baselineB"):
                q = result[f"{method}_Q"]
                y = result[f"{method}_Y"]
                beta = _scalar(result[f"{method}_beta"])
                gamma2 = _scalar(result[f"{method}_gamma2"])
                decay = _scalar(result[f"{method}_decay_rate"])
                mu = _scalar(result[f"{method}_mu"])
                violations = []
                certified = []
                output_residuals = []
                for index, (delta, injection) in enumerate(zip(deltas, injections)):
                    a = delta[:16, :16]
                    b = delta[:16, 16:]
                    c = delta[16:, :16]
                    d = delta[16:, 16:]
                    aq_by = a @ q + b @ y
                    cq_dy = c @ q + d @ y
                    matrix = np.block([
                        [-decay * q - beta * scores[index] * np.eye(16),
                         np.zeros((16, 6)), aq_by.T, cq_dy.T],
                        [np.zeros((6, 16)), -gamma2 * np.eye(6),
                         injection.T, np.zeros((6, 16))],
                        [aq_by, injection, -q, np.zeros((16, 16))],
                        [cq_dy, np.zeros((16, 6)), np.zeros((16, 16)),
                         -np.eye(16)],
                    ])
                    violation = la.eigvalsh(0.5 * (matrix + matrix.T))[-1]
                    violations.append(violation)
                    if scores[index] == 0.0:
                        certified.append(violation)
                    output = np.block([
                        [q, cq_dy.T], [cq_dy, mu * np.eye(16)]
                    ])
                    output_residuals.append(
                        -la.eigvalsh(0.5 * (output + output.T))[0]
                    )
                increment = np.block([
                    [q, y.T], [y, np.eye(4)]
                ])
                self.assertAlmostEqual(
                    max(violations), _scalar(result[f"{method}_v_sur"]),
                    places=9,
                )
                self.assertAlmostEqual(
                    max(certified), _scalar(result[f"{method}_v_cert"]),
                    places=9,
                )
                self.assertAlmostEqual(
                    max(output_residuals),
                    _scalar(result[f"{method}_output_lmi_signed_violation"]),
                    places=9,
                )
                self.assertAlmostEqual(
                    -la.eigvalsh(0.5 * (increment + increment.T))[0],
                    _scalar(result[f"{method}_increment_lmi_signed_violation"]),
                    places=9,
                )

    def test_saved_gain_factors_are_consistent(self):
        count = 0
        paths = list(RAW.glob("*_synthesis.npz"))
        paths.extend(RAW.glob("stage3_time_domain_results*.npz"))
        for path in sorted(paths):
            data = np.load(path, allow_pickle=False)
            for key in data.files:
                if not key.endswith("_Q"):
                    continue
                base = key[:-2]
                if f"{base}_Y" not in data.files or f"{base}_K" not in data.files:
                    continue
                q = data[key]
                y = data[f"{base}_Y"]
                gain = data[f"{base}_K"]
                self.assertGreater(la.eigvalsh(0.5 * (q + q.T))[0], 0.0)
                np.testing.assert_allclose(
                    la.solve(q.T, y.T).T, gain, rtol=1e-10, atol=1e-10
                )
                count += 1
        self.assertEqual(count, 19)


if __name__ == "__main__":
    unittest.main()
