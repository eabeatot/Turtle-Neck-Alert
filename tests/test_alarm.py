from app.alarm import step
def test_repeating_alerts_and_reset():
    s=step({},"bad",0,10,30); assert not s["alert"]
    s=step(s,"bad",9,10,30); assert not s["alert"]
    s=step(s,"bad",10,10,30); assert s["alert"]
    s=step(s,"bad",20,10,30); assert not s["alert"]
    s=step(s,"bad",40,10,30); assert s["alert"]
    s=step(s,"unknown",41,10,30); assert not s["alert"] and s["bad_since"] is None
    s=step(s,"bad",50,10,30); assert not s["alert"]
    s=step(s,"bad",60,10,30); assert s["alert"]
