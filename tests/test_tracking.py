import math
import pytest
from poseguard.config import Config
from poseguard.demo import synthetic_person
from poseguard.tracking import Tracker, TemporalDetector


def test_people_keep_identity_when_detection_order_changes():
    tracker = Tracker()
    initial = tracker.update([synthetic_person((200, 270)), synthetic_person((700, 270))], 0)
    ids = [t.id for t in initial]
    updated = tracker.update([synthetic_person((705, 270)), synthetic_person((205, 270))], 0.1)
    assert [t.id for t in updated] == ids
    assert updated[0].detection.bbox[0] < updated[1].detection.bbox[0]


def test_person_window_does_not_alert_on_another_person():
    cfg = Config(window_seconds=0.3, fall_hold_seconds=0.5)
    tracker, temporal = Tracker(), TemporalDetector(cfg)
    events = []
    for i in range(30):
        timestamp = i / 10
        fraction = min(max((i-9)/4, 0), 1)
        tracks = tracker.update([synthetic_person((250, 280+120*fraction), fraction*math.pi/2),
                                 synthetic_person((700, 280))], timestamp)
        for track in tracks:
            label = "standing" if track.id == 2 or i < 10 else "lying"
            event = temporal.update(track, {label: 1.0}, timestamp)
            if event:
                events.append((track.id, event))
    assert events == [(1, "fall")]
    assert not tracker.tracks[2].alert_active


def test_static_lying_and_single_frame_noise_do_not_alert():
    cfg = Config(window_seconds=0.3, fall_hold_seconds=0.5)
    tracker, temporal = Tracker(), TemporalDetector(cfg)
    for i in range(30):
        track = tracker.update([synthetic_person((250, 280), math.pi / 2)], i / 10)[0]
        assert temporal.update(track, {"lying": 1}, i / 10) is None
    tracker = Tracker()
    for i in range(30):
        track = tracker.update([synthetic_person((250, 280))], i / 10)[0]
        probs = {"falling" if i == 12 else "standing": 1}
        assert temporal.update(track, probs, i / 10) is None


def test_alert_recovers_and_does_not_repeat_every_frame():
    cfg = Config(window_seconds=0.2, fall_hold_seconds=0.4, recovery_seconds=0.5, min_observations=2)
    tracker, temporal = Tracker(), TemporalDetector(cfg)
    events = []
    for i in range(50):
        track = tracker.update([synthetic_person((250, 280))], i / 10)[0]
        label = "falling" if 10 <= i < 30 else "standing"
        event = temporal.update(track, {label: 1}, i / 10)
        if event:
            events.append(event)
    assert events == ["fall", "recovered"]


def test_track_expiry_and_reconnect_do_not_reuse_old_window():
    tracker = Tracker(ttl=0.5)
    old = tracker.update([synthetic_person((200, 270))], 0)[0]
    old.alert_active = True
    new = tracker.update([synthetic_person((200, 270))], 1)[0]
    assert new.id != old.id and not new.alert_active
    tracker.reset()
    another = tracker.update([synthetic_person((200, 270))], 0)[0]
    assert another.id > new.id and not another.observations


def test_tracker_rejects_backward_timestamps():
    tracker = Tracker()
    tracker.update([], 1)
    with pytest.raises(ValueError):
        tracker.update([], 0)



def test_seated_fall_alerts_once_and_static_lying_does_not():
    cfg = Config()
    track = Tracker().update([synthetic_person((250, 280))], 0)[0]
    detector, events = TemporalDetector(cfg), []
    for i in range(150):
        fraction = min(max((i-29)/12, 0), 1)
        track.detection = synthetic_person((250, 280+120*fraction), fraction*math.pi/2)
        probs = {"sitting": .8, "standing": .2} if i < 30 else {"lying": .9, "sitting": .1}
        event = detector.update(track, probs, i / 30)
        if event: events.append(event)
    assert events == ["fall"]


@pytest.mark.parametrize("motion_seconds,drop", [(0.4, 0), (5, 120), (0, 0)])
def test_pose_change_without_fall_motion_does_not_alert(motion_seconds, drop):
    cfg = Config()
    track = Tracker().update([synthetic_person((250, 280))], 0)[0]
    detector = TemporalDetector(cfg)
    for i in range(240):
        t = i/30
        fraction = min(max((t-1)/motion_seconds, 0), 1) if motion_seconds else float(t >= 1)
        track.detection = synthetic_person((250, 280+drop*fraction), fraction*math.pi/2)
        probs = {"sitting": .9} if t < 1 else {"lying": .9}
        assert detector.update(track, probs, t) is None


def test_low_visibility_erases_motion_evidence():
    cfg = Config()
    track = Tracker().update([synthetic_person((250, 280))], 0)[0]
    detector = TemporalDetector(cfg)
    detector.update(track, {"standing": 1}, 0)
    track.motion_evidence_at = 0
    track.detection.keypoints[:, 2] = 0
    detector.update(track, {"unknown": 1}, .1)
    assert track.motion_evidence_at is None and not track.motion


def test_impact_occlusion_keeps_motion_evidence_until_landing():
    cfg = Config(fall_hold_seconds=0.2, motion_evidence_hold_seconds=1.0)
    track = Tracker().update([synthetic_person((250, 280))], 0)[0]
    detector = TemporalDetector(cfg)
    event = None
    for i in range(45):
        t = i / 30
        fraction = min(max((t - .5) / .35, 0), 1)
        track.detection = synthetic_person((250, 280 + 120 * fraction), fraction * math.pi / 2)
        probs = {"standing": .9} if t < .85 else {"sitting": .8, "standing": .2}
        if .85 <= t < 1.0:
            track.detection.keypoints[:, 2] = .1
        event = detector.update(track, probs, t)
        if event == "fall":
            break
    assert event == "fall"


def test_slow_sit_down_is_not_a_motion_fall():
    cfg = Config(motion_fall_hold_seconds=0.2)
    track = Tracker().update([synthetic_person((250, 280))], 0)[0]
    detector = TemporalDetector(cfg)
    events = []
    for i in range(120):
        t = i / 30
        fraction = min(max((t - 1) / 2.0, 0), 1)
        track.detection = synthetic_person((250, 280 + 70 * fraction), fraction * .65)
        event = detector.update(track, {"sitting": .9} if t >= 1 else {"standing": .9}, t)
        if event: events.append(event)
    assert events == []


def test_fast_sit_with_a_tilted_torso_is_not_a_floor_landing():
    cfg = Config(motion_fall_hold_seconds=.2)
    track = Tracker().update([synthetic_person((250, 280))], 0)[0]
    detector = TemporalDetector(cfg)
    for i in range(75):
        t = i / 30
        fraction = min(max((t - .5) / .25, 0), 1)
        track.detection = synthetic_person((250, 280 + 110 * fraction), fraction * .9)
        assert detector.update(track, {"sitting": .9} if t >= .5 else {"standing": .9}, t) is None
    assert track.motion_strength >= cfg.min_fall_motion_strength
    assert not track.alert_active


def test_expired_motion_strength_cannot_promote_a_later_weak_transition():
    cfg = Config(motion_evidence_hold_seconds=.3)
    track = Tracker().update([synthetic_person((250, 280))], 0)[0]
    detector = TemporalDetector(cfg)
    detector.update(track, {"standing": .9}, 0)
    track.motion_evidence_at = 0
    track.motion_evidence_valid_until = .3
    track.motion_strength = 2
    detector.update(track, {"standing": .9}, .31)
    assert track.motion_evidence_at is None
    assert track.motion_strength == 0
    for i in range(1, 10):
        t = .31 + i / 30
        fraction = i / 9
        track.detection = synthetic_person((250, 280 + 8 * fraction), fraction * math.pi / 2)
        assert detector.update(track, {"lying": .9}, t) is None
    assert track.motion_strength < cfg.min_fall_motion_strength


def test_unknown_frame_cannot_start_candidate_from_old_sitting_scores():
    cfg = Config(window_seconds=.5, min_observations=2, motion_fall_hold_seconds=.1)
    track = Tracker().update([synthetic_person((250, 280))], 0)[0]
    detector = TemporalDetector(cfg)
    for i in range(5):
        assert detector.update(track, {"sitting": .9}, i / 30) is None
    track.motion_evidence_at = .1
    track.motion_evidence_valid_until = 1
    track.motion_strength = 1
    track.detection = synthetic_person((250, 280), math.pi / 2)
    track.detection.keypoints[:, 2] = .1
    assert detector.update(track, {"unknown": 1}, .2) is None
    assert track.candidate_since is None
    assert track.last_upright < .2


def test_unknown_upright_box_cancels_an_existing_landing_candidate():
    cfg = Config(window_seconds=.5, min_observations=2, motion_fall_hold_seconds=.25)
    track = Tracker().update([synthetic_person((250, 280))], 0)[0]
    detector = TemporalDetector(cfg)
    detector.update(track, {"standing": .9}, 0)
    detector.update(track, {"standing": .9}, .05)
    track.motion_evidence_at = .1
    track.motion_evidence_valid_until = 1
    track.motion_strength = 1
    track.detection = synthetic_person((250, 400), math.pi / 2)
    assert detector.update(track, {"lying": .9}, .1) is None
    assert track.candidate_since == .1
    track.detection = synthetic_person((250, 280))
    track.detection.keypoints[:, 2] = .1
    assert detector.update(track, {"unknown": 1}, .2) is None
    assert track.candidate_since is None
    assert not track.alert_active


def test_motion_landing_still_requires_minimum_observations():
    cfg = Config(min_observations=3, motion_fall_hold_seconds=.1)
    track = Tracker().update([synthetic_person((250, 400), math.pi / 2)], 0)[0]
    track.motion_evidence_at = 0
    track.motion_evidence_valid_until = 1
    track.motion_strength = 1
    detector = TemporalDetector(cfg)
    assert detector.update(track, {"lying": .9}, 0) is None
    assert detector.update(track, {"lying": .9}, .1) is None
    assert track.candidate_since is None
    assert detector.update(track, {"lying": .9}, .2) is None
    assert track.candidate_since == .2
    assert detector.update(track, {"lying": .9}, .31) == "fall"
