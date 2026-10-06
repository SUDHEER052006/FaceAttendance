"""Pick match.threshold from YOUR enrolled faces, not from a paper.

    ./run.sh tune

Builds two score distributions from the templates already in the database:

  genuine  - pairs of templates belonging to the same person
  impostor - pairs belonging to different people

Then reports the equal-error-rate threshold and, more usefully, the lowest
threshold that produces zero impostor matches on your data. For attendance you
almost always want the strict end: letting the wrong person check in is far
worse than asking someone to stand still for another second.

Needs at least 2 people with 2+ templates each to say anything meaningful.
"""
import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from app import config, db


def load():
    conn = db.init()
    rows = conn.execute(
        "SELECT t.person_id, t.embedding FROM templates t "
        "JOIN people p ON p.id = t.person_id WHERE p.active = 1").fetchall()
    conn.close()
    by_person = {}
    for r in rows:
        by_person.setdefault(r["person_id"], []).append(
            np.frombuffer(r["embedding"], dtype=np.float32))
    return by_person


def main():
    by_person = load()
    people = {pid: vecs for pid, vecs in by_person.items() if len(vecs) >= 1}
    multi = {pid: v for pid, v in people.items() if len(v) >= 2}

    print("=" * 62)
    print(" Threshold tuning")
    print("=" * 62)
    print(" people with templates : %d" % len(people))
    print(" people with 2+        : %d" % len(multi))
    print(" total templates       : %d" % sum(len(v) for v in people.values()))
    print("")

    if len(people) < 2 or not multi:
        print(" Not enough data yet. Enroll at least 2 people with several")
        print(" angles each (the Register page captures 7 by default), then")
        print(" run this again.")
        return 1

    genuine = []
    for vecs in multi.values():
        for a, b in combinations(vecs, 2):
            genuine.append(float(a @ b))

    impostor = []
    pids = list(people.keys())
    for pa, pb in combinations(pids, 2):
        A = np.stack(people[pa])
        B = np.stack(people[pb])
        impostor.extend((A @ B.T).ravel().tolist())

    genuine = np.array(genuine, dtype=np.float64)
    impostor = np.array(impostor, dtype=np.float64)

    print(" genuine  pairs: %5d   mean %.3f   min %.3f"
          % (genuine.size, genuine.mean(), genuine.min()))
    print(" impostor pairs: %5d   mean %.3f   max %.3f"
          % (impostor.size, impostor.mean(), impostor.max()))
    print("")

    grid = np.arange(0.10, 0.90, 0.005)
    far = np.array([(impostor >= t).mean() for t in grid])   # wrong person let in
    frr = np.array([(genuine < t).mean() for t in grid])     # right person rejected

    eer_i = int(np.argmin(np.abs(far - frr)))
    eer_t = grid[eer_i]

    zero_far = grid[far == 0.0]
    strict_t = float(zero_far[0]) if zero_far.size else None

    print(" %-10s %-10s %-10s" % ("threshold", "false-accept", "false-reject"))
    for t in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        i = int(np.argmin(np.abs(grid - t)))
        print(" %-10.2f %-9.2f%% %-9.2f%%" % (t, far[i] * 100, frr[i] * 100))
    print("")
    print(" equal-error threshold     : %.3f  (both errors %.2f%%)"
          % (eer_t, far[eer_i] * 100))
    if strict_t is not None:
        i = int(np.argmin(np.abs(grid - strict_t)))
        print(" lowest zero-false-accept  : %.3f  (rejects %.2f%% of genuine)"
              % (strict_t, frr[i] * 100))
        recommend = round(float(min(strict_t + 0.02, 0.75)), 2)
    else:
        print(" no threshold reaches zero false accepts on this data -")
        print(" your enrollments are probably too few or too similar.")
        recommend = round(float(eer_t + 0.05), 2)

    print("")
    print(" RECOMMENDED  match.threshold: %.2f" % recommend)
    print(" current      match.threshold: %.2f"
          % float(config.g("match.threshold", 0.40)))
    print("")
    print(" Set it on the Settings page, or in config.yaml, then restart the")
    print(" recognizer. Re-run this after every batch of new enrollments.")
    if genuine.min() < 0.25:
        print("")
        print(" WARNING: some same-person pairs score below 0.25. That usually")
        print("          means a bad enrollment sample (eyes closed, motion")
        print("          blur, heavy backlight). Re-capture those people.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
