# Known-answer check of AlpaSim's DelayBuffer as called by events/controller.py.
# Logic RE-TYPED (not imported) from alpasim src/runtime/alpasim_runtime/delay_buffer.py:15-49
# (identical at main@affc2eab and e2e_challenge@e121e591). Call pattern from
# events/controller.py:123-127: add(plan, step_start_us) then at(step_start_us), once per
# ControllerEvent tick (interval = control_timestep_us; event_loop.py:464-487).
from collections import deque
import math

class DelayBuffer:
    def __init__(self, delay_us):
        self.delay_us = delay_us
        self.queue = deque()
    def add(self, obj, ts):
        if self.queue and ts <= self.queue[-1][0]:
            raise ValueError("Timestamps must be in strictly ascending order")
        self.queue.append((ts, obj))
    def at(self, ts):
        while len(self.queue) > 1 and (ts - self.queue[1][0]) >= self.delay_us:
            self.queue.popleft()
        return self.queue[0][1] if self.queue else None

def applied(delay_ms, dt_ms=100, n_ticks=12, gt_ticks=0):
    buf = DelayBuffer(delay_ms * 1000)
    out = []
    for k in range(n_ticks):
        t = k * dt_ms * 1000
        label = f"GT{k}" if k < gt_ticks else f"P{k}"   # force-GT ticks feed GT references (controller.py:91-93,123)
        buf.add(label, t)
        out.append(buf.at(t))
    return out

def lag_ticks(delay_ms, dt_ms=100):
    # steady-state lag measured from the buffer, not from the formula
    seq = applied(delay_ms, dt_ms, n_ticks=40)
    last = seq[-1]
    return 39 - int(last[1:])

print("1) Steady-state lag at control step 100 ms (measured from re-typed buffer) vs ceil(d/dt):")
for d in [0, 1, 50, 99, 100, 101, 139, 171, 200, 201, 250, 300]:
    m = lag_ticks(d)
    f = math.ceil(d / 100)
    v = 15.0
    print(f"   delay {d:>3} ms -> {m} ticks late (ceil = {f}); applied staleness {m*100:>3} ms; "
          f"distance at 15 m/s: modeled v*d = {v*d/1000:.3f} m, applied v*ticks*dt = {v*m*0.1:.3f} m")

print("\n2) Applied-plan sequence, plans added at t = 0,100,...,500 ms (no force-GT):")
for d in [0, 100, 139, 171, 200]:
    print(f"   {d:>3} ms: {applied(d, n_ticks=6)}")

print("\n3) Warm-up with force-GT: 17 GT ticks (force_gt_duration_us 1.7 s, base_config.yaml:277), closed loop from tick 17:")
for d in [0, 139, 200, 300]:
    seq = applied(d, n_ticks=22, gt_ticks=17)
    print(f"   {d:>3} ms: ticks 17..21 apply {seq[17:22]}")

print("\n4) Same delays at control step 50 ms (policy and controller share dt, event_loop.py:464-487; camera clocks are separate, camera.py:196-210):")
for d in [100, 139, 171, 200]:
    print(f"   delay {d} ms -> {lag_ticks(d, dt_ms=50)} ticks of 50 ms = {lag_ticks(d, dt_ms=50)*50} ms applied staleness")
