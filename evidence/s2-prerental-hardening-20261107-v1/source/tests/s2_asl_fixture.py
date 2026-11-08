"""Invented samples encoded from the pinned .proto, without the decoder/serializer.

These bytes exercise actual ASL framing; they are synthetic, not simulator output.
"""

import struct


def varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def field(number, payload, wire=2):
    tag = varint((number << 3) | wire)
    return tag + (varint(len(payload)) if wire == 2 else b"") + payload


def fixed64(number, value):
    return field(number, struct.pack("<Q", value), 1)


def integer(number, value):
    return field(number, varint(value), 0)


def text_field(number, value):
    return field(number, value.encode("utf-8"))


def framed(payload):
    return struct.pack(">L", len(payload)) + payload


def pose(position=(1.0, 2.0, 3.0), quaternion=(1.0, 0.0, 0.0, 0.0)):
    vector = b"".join(field(i, struct.pack("<f", v), 5) for i, v in enumerate(position, 1))
    rotation = b"".join(
        field(i, struct.pack("<f", v), 5) for i, v in enumerate(quaternion, 1)
    )
    return field(1, vector) + field(2, rotation)


def metadata(*, session_uuid="rollout-1", scene_id="scene-1", n_steps=2, dt=100,
             start=1000):
    session = (
        text_field(1, session_uuid) + text_field(2, scene_id) + integer(3, 1)
        + integer(4, n_steps) + fixed64(5, start) + fixed64(6, dt)
    )
    return field(1, field(1, session))


def actor_event(timestamp, *, actor_id="EGO", position=(1.0, 2.0, 3.0),
                quaternion=(1.0, 0.0, 0.0, 0.0)):
    actor = text_field(1, actor_id) + field(2, pose(position, quaternion))
    return field(2, fixed64(1, timestamp) + field(2, actor))


def asl_bytes(*, session_uuid="rollout-1", scene_id="scene-1", n_steps=2, dt=100,
              timestamps=(1000, 1100, 1150, 1200), requested_steps=None):
    # requested_steps is deliberately not serialized: upstream ASL logs the effective count.
    del requested_steps
    entries = [metadata(session_uuid=session_uuid, scene_id=scene_id, n_steps=n_steps, dt=dt)]
    entries.extend(actor_event(timestamp) for timestamp in timestamps)
    return b"".join(map(framed, entries))
