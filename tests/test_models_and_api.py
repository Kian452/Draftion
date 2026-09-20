"""
Model versioning, which model gets served (trained / raw-stats fallback / demo), the
hot reload of a newer version, and the Ranked map pool - end to end through the API,
on small throwaway databases built with the synthetic generator (src/synthetic.py).
Nothing here touches data/.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib

from src import model as model_mod
from src.config import Settings
from src.db import active_ranked_maps, connect
from src.model import (
    DEFAULT_C, C_CANDIDATES, WinProbabilityModel, latest_version, list_versions, resolve_model_path,
    save_versioned, select_c, train_model,
)
from src.model_select import confidence, select_model
from src.synthetic import BATTLE_TYPE, MIN_RANK, generate_dataset

BRAWLERS = [f"BR{i:02d}" for i in range(16)]


class TempDirCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # sqlite may still hold a WAL file on Windows
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)


def make_settings(root: Path, **over) -> Settings:
    base = dict(
        db_path=str(root / "brawl.db"), brawlers_path=str(root / "brawlers.json"), maps_path=str(root / "maps.json"),
        model_path=None, model_dir=str(root / "models"), legacy_model_path=str(root / "legacy.joblib"), keep_models=5, ranked_type=BATTLE_TYPE, min_rank=MIN_RANK,
        min_train_rows=500, min_raw_battles=50, early_model_battles=10_000, draft_time_budget=0.3,
        model_check_seconds=0.0, map_window_days=7, map_min_battles=3, crawl_interval_hours=2, crawl_max_players=10,
        retrain_interval_hours=24, retrain_min_new_battles=100, backend="logreg",
    )
    base.update(over)
    return Settings(**base)


def write_roster(root: Path) -> None:
    (root / "brawlers.json").write_text(json.dumps([{"name": n} for n in BRAWLERS]), encoding="utf-8")


def trained(root: Path, data_source="real", n_battles=None, c=0.05) -> WinProbabilityModel:
    m, _ = train_model(str(root / "brawl.db"), BATTLE_TYPE, MIN_RANK, brawlers=BRAWLERS, data_source=data_source, c=c)
    if n_battles is not None:
        m.n_battles = n_battles
    return m


class Versioning(TempDirCase):
    def test_newest_version_wins_and_old_ones_are_pruned(self):
        d = self.root / "models"
        d.mkdir()
        for stamp in ("20200101T000000Z", "20200102T000000Z", "20200103T000000Z", "20200104T000000Z"):
            joblib.dump({"stamp": stamp}, d / f"win_model_{stamp}.joblib")
        self.assertEqual(latest_version(str(d)).name, "win_model_20200104T000000Z.joblib")

        new_path = save_versioned({"stamp": "now"}, str(d), keep=3)  # stamped with the real clock: newer than the 2020 ones
        versions = [p.name for p in list_versions(str(d))]
        self.assertEqual(len(versions), 3)                       # pruned down to `keep`
        self.assertIn(Path(new_path).name, versions)             # the fresh one survives
        self.assertNotIn("win_model_20200101T000000Z.joblib", versions)  # the oldest went first
        self.assertEqual(joblib.load(new_path), {"stamp": "now"})
        self.assertFalse(list(d.glob("*.tmp")))                  # no half-written leftovers

    def test_resolve_order_explicit_then_newest_version_then_legacy_then_none(self):
        d, legacy = self.root / "models", self.root / "old.joblib"
        self.assertIsNone(resolve_model_path(None, str(d), str(legacy)))
        legacy.write_bytes(b"x")
        self.assertEqual(resolve_model_path(None, str(d), str(legacy)), str(legacy))
        d.mkdir()
        (d / "win_model_20260101T000000Z.joblib").write_bytes(b"x")
        (d / "win_model_20260201T000000Z.joblib").write_bytes(b"x")
        self.assertTrue(resolve_model_path(None, str(d), str(legacy)).endswith("win_model_20260201T000000Z.joblib"))
        self.assertEqual(resolve_model_path("pinned.joblib", str(d), str(legacy)), "pinned.joblib")


class TrainingChoices(TempDirCase):
    def setUp(self):
        super().setUp()
        generate_dataset(str(self.root / "brawl.db"), BRAWLERS, n_battles=700)

    def test_the_regularization_is_chosen_by_cv_and_recorded(self):
        m, df = train_model(str(self.root / "brawl.db"), BATTLE_TYPE, MIN_RANK, brawlers=BRAWLERS, data_source="synthetic")
        self.assertIn(m.metrics["c"], C_CANDIDATES + (DEFAULT_C,))
        self.assertEqual(m.n_battles, len(df))
        for key in ("auc", "brier", "logloss", "n_test", "cv_logloss", "cv_raw_logloss"):
            self.assertIn(key, m.metrics)  # incl. the raw-stats fallback measured on the same CV folds
        self.assertTrue(m.trained_at)

    def test_a_pinned_c_is_used_as_is(self):
        m = trained(self.root, c=0.123)
        self.assertEqual(m.metrics["c"], 0.123)

    def test_cv_is_skipped_on_tiny_data(self):
        df = model_mod.build_training_table(str(self.root / "brawl.db"), BATTLE_TYPE, MIN_RANK).head(100)
        self.assertEqual(select_c(df), DEFAULT_C)

    def test_weights_are_rescaled_so_they_cannot_change_the_regularization(self):
        import numpy as np
        w = model_mod._unit_mean(np.array([1.0, 3.0, 5.0]))
        self.assertAlmostEqual(float(w.mean()), 1.0)
        self.assertEqual(list(model_mod._unit_mean(np.array([]))), [])

    def test_no_training_data_is_thrown_away_when_calibration_is_skipped(self):
        """The calibration slice is only cut off if it will actually be used: otherwise 15%
        of the battles would never take part in the fit."""
        df = model_mod.build_training_table(str(self.root / "brawl.db"), BATTLE_TYPE, MIN_RANK)
        fit_sizes = []
        real = model_mod._with_mirrored
        with mock.patch.object(model_mod, "_with_mirrored", lambda d: (fit_sizes.append(len(d)), real(d))[1]):
            model_mod.fit_pipeline(df, "logreg", 0.05, "sigmoid")
        self.assertEqual(fit_sizes[0], len(df))  # the fit saw everything

    def test_calibration_stays_off_until_the_slice_is_large_enough(self):
        from sklearn.calibration import CalibratedClassifierCV

        m = trained(self.root)  # ~700 battles -> a calibration slice of roughly 80, far below the minimum
        self.assertNotIsInstance(m.estimator, CalibratedClassifierCV)

    def test_the_model_stays_side_symmetric(self):
        m = trained(self.root)
        for x, y in ((BRAWLERS[0:3], BRAWLERS[3:6]), (BRAWLERS[6:9], BRAWLERS[9:12])):
            self.assertAlmostEqual(m.predict_proba(None, x, y) + m.predict_proba(None, y, x), 1.0, delta=0.02)


class Selection(TempDirCase):
    def setUp(self):
        super().setUp()
        write_roster(self.root)

    def with_data(self, n=700):
        generate_dataset(str(self.root / "brawl.db"), BRAWLERS, n_battles=n)

    def test_nothing_at_all(self):
        sel = select_model(make_settings(self.root))
        self.assertIsNone(sel.model)

    def test_only_data_gives_the_raw_stats_fallback(self):
        self.with_data()
        sel = select_model(make_settings(self.root))
        self.assertEqual(sel.model.kind, "raw-stats")
        self.assertEqual(confidence(sel.model, make_settings(self.root)), "low")

    def test_too_little_data_for_even_the_fallback(self):
        self.with_data(60)  # ~60 battles, below MIN_RAW_BATTLES once filtered for rank
        sel = select_model(make_settings(self.root, min_raw_battles=10_000))
        self.assertIsNone(sel.model)

    def test_a_real_model_with_enough_battles_is_served(self):
        self.with_data()
        save_versioned(trained(self.root, n_battles=5000), str(self.root / "models"))
        s = make_settings(self.root)
        sel = select_model(s)
        self.assertEqual(sel.model.kind, "trained")
        self.assertEqual(confidence(sel.model, s), "early")  # < 10,000 battles

    def test_a_trained_model_needs_min_train_rows_else_the_fallback_answers(self):
        self.with_data()
        save_versioned(trained(self.root, n_battles=200), str(self.root / "models"))
        self.assertEqual(select_model(make_settings(self.root)).model.kind, "raw-stats")

    def test_a_trained_model_that_loses_to_the_raw_stats_is_not_served_until_it_wins(self):
        self.with_data()
        models = self.root / "models"
        worse = trained(self.root, n_battles=5000)
        worse.metrics.update(cv_logloss=0.700, cv_raw_logloss=0.685)
        save_versioned(worse, str(models))
        self.assertEqual(select_model(make_settings(self.root)).model.kind, "raw-stats")

        better = trained(self.root, n_battles=9000)  # a later training run that now beats the fallback
        better.metrics.update(cv_logloss=0.680, cv_raw_logloss=0.685)
        joblib.dump(better, models / "win_model_29990101T000000Z.joblib")
        self.assertEqual(select_model(make_settings(self.root)).model.kind, "trained")

    def test_without_a_recorded_comparison_the_trained_model_gets_the_benefit_of_the_doubt(self):
        self.with_data()
        save_versioned(trained(self.root, n_battles=5000), str(self.root / "models"))  # pinned C: no CV numbers
        self.assertEqual(select_model(make_settings(self.root)).model.kind, "trained")

    def test_confidence_levels(self):
        s = make_settings(self.root)
        self.with_data()
        self.assertEqual(confidence(trained(self.root, "synthetic"), s), "demo")
        self.assertEqual(confidence(trained(self.root, n_battles=9_999), s), "early")
        self.assertEqual(confidence(trained(self.root, n_battles=10_000), s), "ok")
        self.assertIsNone(confidence(None, s))

    def test_the_synthetic_demo_model_gives_way_to_real_data_but_not_to_nothing(self):
        self.with_data()
        save_versioned(trained(self.root, "synthetic"), str(self.root / "models"))
        # no real data -> the demo model is all there is
        no_data = make_settings(self.root, db_path=str(self.root / "does_not_exist.db"))
        sel = select_model(no_data)
        self.assertEqual((sel.model.kind, sel.model.data_source), ("trained", "synthetic"))
        self.assertEqual(confidence(sel.model, no_data), "demo")
        # real data appears -> the raw-stats fallback replaces the demo
        self.assertEqual(select_model(make_settings(self.root)).model.kind, "raw-stats")

    def test_an_explicit_model_path_is_served_exactly(self):
        self.with_data()
        pinned = self.root / "pinned.joblib"
        trained(self.root, "synthetic").save(str(pinned))
        sel = select_model(make_settings(self.root, model_path=str(pinned)))
        self.assertEqual((sel.model.kind, sel.model.data_source), ("trained", "synthetic"))  # not replaced by raw stats
        with self.assertRaises(FileNotFoundError):
            select_model(make_settings(self.root, model_path=str(self.root / "missing.joblib")))


class ApiHotReload(TempDirCase):
    """The running server picks up newer models and reports what it serves."""

    def setUp(self):
        super().setUp()
        write_roster(self.root)
        (self.root / "maps.json").write_text("[]", encoding="utf-8")
        generate_dataset(str(self.root / "brawl.db"), BRAWLERS, n_battles=700)
        env = {
            "DB_PATH": str(self.root / "brawl.db"), "BRAWLERS_PATH": str(self.root / "brawlers.json"),
            "MAPS_PATH": str(self.root / "maps.json"), "WIN_MODEL_DIR": str(self.root / "models"),
            "LEGACY_MODEL_PATH": str(self.root / "legacy.joblib"), "MODEL_CHECK_SECONDS": "0", "MIN_TRAIN_ROWS": "500", "MIN_RAW_BATTLES": "50",
            "MIN_RANK": str(MIN_RANK), "RANKED_TYPE": BATTLE_TYPE, "DRAFT_TIME_BUDGET": "0.3",
        }
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("WIN_MODEL_PATH", None)

        from fastapi.testclient import TestClient

        from src import api
        self.client = TestClient(api.app)
        self.client.__enter__()
        self.addCleanup(lambda: self.client.__exit__(None, None, None))

    def health(self):
        return self.client.get("/health").json()

    def test_without_a_trained_model_the_fallback_answers_and_says_so(self):
        h = self.health()
        self.assertEqual((h["model_loaded"], h["model_kind"], h["model_confidence"]), (True, "raw-stats", "low"))
        self.assertGreater(h["model_training_battles"], 50)
        r = self.client.post("/recommend", json={})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertGreater(len(r.json()["recommendations"]), 0)
        self.assertEqual(self.client.post("/analyze", json={"picks_a": BRAWLERS[:3], "picks_b": BRAWLERS[3:6]}).status_code, 200)

    def test_a_new_version_is_served_without_a_restart_and_the_newest_wins(self):
        self.assertEqual(self.health()["model_kind"], "raw-stats")

        save_versioned(trained(self.root, n_battles=700), str(self.root / "models"))
        h = self.health()
        self.assertEqual((h["model_kind"], h["model_confidence"], h["model_training_battles"]), ("trained", "early", 700))
        self.assertIn("auc", h["model_metrics"])

        newer = trained(self.root, n_battles=1234)
        joblib.dump(newer, self.root / "models" / "win_model_29990101T000000Z.joblib")  # sorts after any real timestamp
        self.assertEqual(self.health()["model_training_battles"], 1234)

    def test_an_explicit_model_path_pins_the_model_against_new_versions(self):
        pinned = self.root / "pinned.joblib"
        trained(self.root, "synthetic").save(str(pinned))
        os.environ["WIN_MODEL_PATH"] = str(pinned)
        from src import api
        api.reload_model()
        self.assertEqual(self.health()["model_confidence"], "demo")
        save_versioned(trained(self.root, n_battles=9000), str(self.root / "models"))
        self.assertEqual(self.health()["model_confidence"], "demo")


def insert_battle(conn, n, battle_time, map_name, battle_type=BATTLE_TYPE):
    conn.execute(
        "INSERT INTO battles (battle_hash, battle_time, mode, map, type) VALUES (?, ?, ?, ?, ?)",
        (f"h{n}", battle_time, "gemGrab", map_name, battle_type),
    )


class RankedMapPool(TempDirCase):
    def build_db(self):
        db = str(self.root / "brawl.db")
        conn = connect(db)
        n = 0
        for i in range(7):  # seven maps played 5x in the last few days
            for k in range(5):
                n += 1
                insert_battle(conn, n, f"202609{15 + k % 3:02d}T1{k}0000.000Z", f"Live Map {i}")
        for k in range(2):  # a stray map: only 2 battles
            n += 1
            insert_battle(conn, n, "20260919T100000.000Z", "Stray Map")
        for k in range(6):  # a map from a previous season, 60 days back
            n += 1
            insert_battle(conn, n, "20260720T100000.000Z", "Old Map")
        for k in range(6):  # trophy games on a map the Ranked pool doesn't have
            n += 1
            insert_battle(conn, n, "20260919T100000.000Z", "Trophy Only Map", battle_type="ranked")
        conn.commit()
        return conn

    def test_the_pool_is_what_was_played_recently_relative_to_the_newest_battle(self):
        conn = self.build_db()
        pool = active_ranked_maps(conn, BATTLE_TYPE, window_days=7, min_battles=3)
        self.assertEqual(set(pool), {f"Live Map {i}" for i in range(7)})
        self.assertEqual(set(active_ranked_maps(conn, BATTLE_TYPE, window_days=7, min_battles=1)),
                         {f"Live Map {i}" for i in range(7)} | {"Stray Map"})
        self.assertIn("Old Map", active_ranked_maps(conn, BATTLE_TYPE, window_days=90, min_battles=3))
        self.assertEqual(active_ranked_maps(conn, "nothing", 7, 3), {})
        conn.close()

    def test_maps_endpoint_marks_the_active_pool(self):
        self.build_db().close()
        write_roster(self.root)
        names = [f"Live Map {i}" for i in range(7)] + ["Old Map", "Stray Map", "Never Played"]
        (self.root / "maps.json").write_text(
            json.dumps([{"name": n, "mode": "Gem Grab", "image": None} for n in names]), encoding="utf-8")
        env = {"DB_PATH": str(self.root / "brawl.db"), "BRAWLERS_PATH": str(self.root / "brawlers.json"),
               "MAPS_PATH": str(self.root / "maps.json"), "WIN_MODEL_DIR": str(self.root / "models"),
               "LEGACY_MODEL_PATH": str(self.root / "legacy.joblib"), "MAP_WINDOW_DAYS": "7", "MAP_MIN_BATTLES": "3", "RANKED_TYPE": BATTLE_TYPE}
        with mock.patch.dict(os.environ, env):
            from fastapi.testclient import TestClient

            from src import api
            with TestClient(api.app) as client:
                maps = {m["name"]: m["active"] for m in client.get("/maps").json()}
        self.assertEqual({n for n, a in maps.items() if a}, {f"Live Map {i}" for i in range(7)})
        self.assertFalse(any(maps[n] for n in ("Old Map", "Stray Map", "Never Played")))

    def test_a_thin_database_hides_nothing(self):
        conn = connect(str(self.root / "brawl.db"))
        for n in range(10):
            insert_battle(conn, n, "20260919T100000.000Z", "Only Map")
        conn.commit()
        conn.close()
        write_roster(self.root)
        (self.root / "maps.json").write_text(
            json.dumps([{"name": n, "mode": "Gem Grab", "image": None} for n in ("Only Map", "Other 1", "Other 2")]),
            encoding="utf-8")
        env = {"DB_PATH": str(self.root / "brawl.db"), "BRAWLERS_PATH": str(self.root / "brawlers.json"),
               "MAPS_PATH": str(self.root / "maps.json"), "WIN_MODEL_DIR": str(self.root / "models"),
               "LEGACY_MODEL_PATH": str(self.root / "legacy.joblib"), "RANKED_TYPE": BATTLE_TYPE}
        with mock.patch.dict(os.environ, env):
            from fastapi.testclient import TestClient

            from src import api
            with TestClient(api.app) as client:
                self.assertTrue(all(m["active"] for m in client.get("/maps").json()))


if __name__ == "__main__":
    unittest.main()
