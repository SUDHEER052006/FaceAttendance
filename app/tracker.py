"""Minimal IOU tracker.

A doorway does not need Kalman filtering or ByteTrack's full machinery. What it
needs is stable identity across ~10 frames so votes can accumulate, and a
memory of "this track is already logged" so one person walking past produces
one attendance row instead of forty.
"""
import itertools

from . import config

_ids = itertools.count(1)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / float(aw * ah + bw * bh - inter)


class Track:
    __slots__ = ("id", "box", "age", "misses", "votes", "resolved",
                 "person_id", "best_score", "liveness", "thumb")

    def __init__(self, box):
        self.id = next(_ids)
        self.box = box
        self.age = 0
        self.misses = 0
        self.votes = []          # recent (person_id, score) - person_id None = no match
        self.resolved = False    # already written to attendance
        self.person_id = None
        self.best_score = 0.0
        self.liveness = None
        self.thumb = None

    def push_vote(self, person_id, score):
        window = config.g("match.vote_window", 10)
        self.votes.append((person_id, score))
        if len(self.votes) > window:
            self.votes = self.votes[-window:]
        if person_id is not None and score > self.best_score:
            self.best_score = score

    def verdict(self):
        """Return (person_id, mean_score, vote_count) once enough frames agree."""
        need = config.g("match.votes_required", 4)
        tally = {}
        for pid, score in self.votes:
            if pid is None:
                continue
            hit = tally.setdefault(pid, [0, 0.0])
            hit[0] += 1
            hit[1] += score
        if not tally:
            return None, 0.0, 0
        pid, (count, total) = max(tally.items(), key=lambda kv: kv[1][0])
        if count < need:
            return None, total / count, count
        return pid, total / count, count

    def miss_ratio(self):
        """Fraction of recent votes that matched nobody - used to flag unknowns."""
        if not self.votes:
            return 0.0
        return sum(1 for pid, _ in self.votes if pid is None) / len(self.votes)


class Tracker:
    def __init__(self):
        self.tracks = []

    def update(self, boxes):
        """Associate detections to tracks greedily by IOU. Returns live tracks
        paired with their detection index (or None if the track coasted)."""
        thresh = config.g("tracker.iou_threshold", 0.3)
        max_age = config.g("tracker.max_age", 12)
        unmatched = set(range(len(boxes)))
        pairs = []

        for track in self.tracks:
            best_i, best_v = None, 0.0
            for i in unmatched:
                v = iou(track.box, boxes[i])
                if v > best_v:
                    best_i, best_v = i, v
            if best_i is not None and best_v >= thresh:
                unmatched.discard(best_i)
                track.box = boxes[best_i]
                track.age += 1
                track.misses = 0
                pairs.append((track, best_i))
            else:
                track.misses += 1
                pairs.append((track, None))

        for i in sorted(unmatched):
            track = Track(boxes[i])
            self.tracks.append(track)
            pairs.append((track, i))

        self.tracks = [t for t in self.tracks if t.misses <= max_age]
        alive = {id(t) for t in self.tracks}
        return [(t, i) for t, i in pairs if id(t) in alive]

    def reset(self):
        self.tracks = []
