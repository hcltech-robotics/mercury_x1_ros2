"""NavStatus: the Navigate tab's stack rows, from fake RTAB-Map / odometry messages (no ROS needed)."""
from types import SimpleNamespace as NS

from mercury_x1_web import navstatus
from mercury_x1_web.navstatus import NavStatus

LOC = {"running": True, "mode": "localization", "map": "office"}
MAP = {"running": True, "mode": "mapping", "map": "office"}
IDLE = {"state": "idle"}


def info(loop=0, prox=0, words=120):
    return NS(stats_keys=["Keypoint/Current_frame/words", "Memory/Working_memory_size/"],
              stats_values=[float(words), 649.0], loop_closure_id=loop, proximity_detection_id=prox)


def odom(x, y=0.0):
    return NS(pose=NS(pose=NS(position=NS(x=x, y=y))))


def rows(ns, slam=LOC, nav2=(False, False), keepout=0):
    snap = ns.snapshot(slam, nav2[0], nav2[1], IDLE, keepout)
    return snap["localized"], {r["key"]: r for r in snap["rows"]}


def test_slam_off():
    loc, r = rows(NavStatus(), slam={"running": False})
    assert loc == "off" and r["slam"]["state"] == "off" and r["loc"]["state"] == "off"


def test_not_localized_until_a_match():
    ns = NavStatus()
    ns.on_info(info())
    loc, r = rows(ns)
    assert loc == "no" and r["loc"]["state"] == "bad" and "Robot is here" in r["loc"]["hint"]


def test_manual_pose_then_lidar_match():
    ns = NavStatus()
    ns.on_info(info())
    ns.manual_pose()
    assert rows(ns)[0] == "manual"
    ns.on_info(info(prox=12))
    loc, r = rows(ns)
    assert loc == "yes" and "lidar" in r["loc"]["text"]


def test_uncertain_after_driving_without_match():
    ns = NavStatus()
    ns.on_info(info(loop=5))
    assert rows(ns)[0] == "yes"
    for i in range(60):                       # 0.1 m steps: 5.9 m
        ns.on_odom(odom(i * 0.1))
    ns.on_info(info())
    loc, r = rows(ns)
    assert loc == "uncertain" and r["loc"]["state"] == "warn"
    ns.on_info(info(loop=7))                  # a new match clears it
    assert rows(ns)[0] == "yes"


def test_odometry_jump_ignored():
    ns = NavStatus()
    ns.on_info(info(loop=5))
    ns.on_odom(odom(0.0))
    ns.on_odom(odom(30.0))                    # odometry reset, not driving
    assert ns.dist_since_match == 0.0


def test_dark_camera_warns():
    ns = NavStatus()
    for _ in range(40):
        ns.on_scan(None)
        ns.on_odom(odom(0.0))
    ns.on_info(info(words=0))
    _, r = rows(ns)
    assert r["sensors"]["state"] == "warn" and "too dark" in r["sensors"]["hint"]


def test_stale_info_after_startup(monkeypatch):
    ns = NavStatus()
    ns.on_info(info(loop=1))
    t = navstatus.time.monotonic()
    monkeypatch.setattr(navstatus.time, "monotonic", lambda: t + 60.0)
    loc, r = rows(ns)
    assert r["slam"]["state"] == "bad" and loc == "unknown"


def test_nav2_ready_but_not_localized_warns():
    ns = NavStatus()
    ns.on_info(info())
    _, r = rows(ns, nav2=(True, True), keepout=2)
    assert r["nav2"]["state"] == "warn" and "not reliably localized" in r["nav2"]["text"]
    assert r["keepout"]["state"] == "ok" and "2 zones" in r["keepout"]["text"]


def test_mapping_mode():
    ns = NavStatus()
    ns.on_info(info())
    loc, r = rows(ns, slam=MAP)
    assert loc == "mapping" and r["loc"]["state"] == "ok"
