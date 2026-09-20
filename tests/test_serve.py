"""
The hosted worker (src/serve.py): versioned retraining, skipping when there is no new
data, and the crawl -> retrain -> reload loop. The crawler itself is replaced by a stub,
so nothing here talks to the Brawl Stars API.
"""
from __future__ import annotations

import random
import threading
import time
import unittest
from datetime import datetime, timedelta
from unittest import mock

from src import serve
from src.db import connect, store_battle
from src.model import list_versions
from src.synthetic import _build_bias, generate_battle_entry, generate_dataset
from tests.test_models_and_api import BRAWLERS, TempDirCase, make_settings, write_roster


def add_battles(db_path, n, seed=99):
    """More synthetic battles on top of an existing database (generate_dataset always starts fresh)."""
    rng = random.Random(seed)
    bias = _build_bias(random.Random(42), BRAWLERS)
    conn = connect(db_path)
    start = datetime(2026, 6, 1)
    for i in range(n):
        entry, tag = generate_battle_entry(rng, BRAWLERS, bias, start + timedelta(minutes=i))
        store_battle(conn, entry, queried_tag=tag)
    conn.close()


class Retraining(TempDirCase):
    def setUp(self):
        super().setUp()
        write_roster(self.root)
        generate_dataset(str(self.root / "brawl.db"), BRAWLERS, n_battles=700)
        self.s = make_settings(self.root, min_train_rows=300, retrain_min_new_battles=100)

    def test_trains_a_new_version_reloads_and_then_waits_for_enough_new_battles(self):
        reloads = []
        self.assertTrue(serve.retrain_if_ready(self.s, lambda: reloads.append(1)))
        self.assertEqual(len(reloads), 1)
        self.assertEqual(len(list_versions(self.s.model_dir)), 1)

        # no new battles -> no new model, no reload
        self.assertFalse(serve.retrain_if_ready(self.s, lambda: reloads.append(1)))
        self.assertEqual((len(reloads), len(list_versions(self.s.model_dir))), (1, 1))

        # fewer than RETRAIN_MIN_NEW_BATTLES new -> still waiting
        add_battles(str(self.root / "brawl.db"), 40)
        self.assertFalse(serve.retrain_if_ready(self.s, lambda: reloads.append(1)))

        # enough new battles -> a second version (the first one stays, until KEEP_MODELS is exceeded)
        add_battles(str(self.root / "brawl.db"), 400, seed=100)
        time.sleep(1.1)  # version files are stamped to the second
        self.assertTrue(serve.retrain_if_ready(self.s, lambda: reloads.append(1)))
        self.assertEqual((len(reloads), len(list_versions(self.s.model_dir))), (2, 2))

    def test_does_not_train_below_min_train_rows(self):
        s = make_settings(self.root, min_train_rows=100_000)
        reloads = []
        self.assertFalse(serve.retrain_if_ready(s, lambda: reloads.append(1)))
        self.assertEqual((reloads, list_versions(s.model_dir)), ([], []))

    def test_old_versions_are_pruned_to_keep_models(self):
        s = make_settings(self.root, min_train_rows=300, retrain_min_new_battles=1, keep_models=2)
        for seed in (1, 2, 3):
            add_battles(str(self.root / "brawl.db"), 30, seed=seed)
            serve.retrain_if_ready(s, lambda: None)
            time.sleep(1.1)
        self.assertEqual(len(list_versions(s.model_dir)), 2)


class WorkerLoop(TempDirCase):
    def test_a_cycle_crawls_trains_and_reloads_and_survives_a_failing_crawl(self):
        write_roster(self.root)
        generate_dataset(str(self.root / "brawl.db"), BRAWLERS, n_battles=700)
        s = make_settings(self.root, min_train_rows=300, crawl_interval_hours=1)
        calls = {"crawl": 0, "reload": 0}

        def failing_crawl(_settings):
            calls["crawl"] += 1
            raise RuntimeError("403 - the key's IP doesn't match")  # e.g. the wrong key on Railway

        def reload_model():
            calls["reload"] += 1

        stop = threading.Event()
        with mock.patch.object(serve, "crawl_once", failing_crawl):
            t = threading.Thread(target=serve.worker_loop, args=(s, reload_model, stop), daemon=True)
            t.start()
            deadline = time.time() + 60
            while calls["reload"] < 2 and time.time() < deadline:  # once after the retrain, once after the cycle
                time.sleep(0.1)
            stop.set()
            t.join(timeout=10)

        self.assertFalse(t.is_alive())
        self.assertEqual(calls["crawl"], 1)                 # the crawl failed, the loop carried on
        self.assertGreaterEqual(calls["reload"], 2)         # retrained (-> reload) and re-selected after the cycle
        self.assertEqual(len(list_versions(s.model_dir)), 1)


if __name__ == "__main__":
    unittest.main()
