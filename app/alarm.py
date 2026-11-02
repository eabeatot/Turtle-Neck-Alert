"""순수 시간 기반 자세 알림 상태 머신."""
def step(state, posture, now, delay=10, repeat=30):
    s=dict(state or {})
    if posture != "bad":
        # good/unknown 모두 나쁜 자세 지속 타이머를 초기화한다.
        return {"bad_since": None, "last_alert": None, "alert": False}
    if s.get("bad_since") is None:
        s["bad_since"] = now
        s["last_alert"] = None
    elapsed=now-s["bad_since"]
    due=elapsed >= delay and (s.get("last_alert") is None or now-s["last_alert"] >= repeat)
    s["alert"] = bool(due)
    if due: s["last_alert"] = now
    return s
