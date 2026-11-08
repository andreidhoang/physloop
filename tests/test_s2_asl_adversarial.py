"""Independent literal-wire known answer, without the shared fixture encoder."""

import json

from gate.s2_asl import inspect_asl


def test_literal_wire_pose_and_all_torn_suffixes_preserve_unknown_completion():
    # Hand-coded from pinned logging.proto/common.proto and write_pb_stream:
    # four-byte big-endian frame lengths; SessionMetadata fields 1..6;
    # ActorPoses fixed64 timestamp; Pose vec xyz and quaternion wxyz floats.
    # The session declares two effective steps against three requested steps.
    metadata = bytes.fromhex(
        "000000200a1e0a1c0a01721201731801200229e803000000000000316400000000000000"
    )
    poses = bytes.fromhex(
        "0000003b123909e803000000000000122e0a0345474f12270a0f0d0000803f"
        "15000000401d0000404012140d0000803f15000000001d000000002500000000"
    )
    data = metadata + poses
    arguments = {
        "expected_session_uuid": "r", "expected_scene_id": "s", "requested_steps": 3,
    }
    result = inspect_asl(data, **arguments)
    assert result["inspection_valid"]
    assert result["effective_n_sim_steps"] == 2
    assert result["actual_steps"] is None
    assert result["pose_events"][0]["actors"][0]["pose"] == {
        "translation": [1.0, 2.0, 3.0],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    assert result["coordinate_frame"] == "local_to_aabb"
    for key in ("completion", "renderer_coverage", "black_frame_coverage", "map_coverage"):
        assert result[key] is None
    for key in ("runtime_qualified", "score_admitted", "cloud_launch_authorized"):
        assert result[key] is False
    json.dumps(result, allow_nan=False)

    assert len(poses) == 63
    for cut in range(1, len(poses)):
        torn = inspect_asl(data[:-cut], **arguments)
        assert not torn["inspection_valid"], cut
        assert torn["metadata"]["session_uuid"] == "r"
        json.dumps(torn, allow_nan=False)

    wrong = inspect_asl(data, **{**arguments, "expected_session_uuid": "wrong"})
    assert wrong["parsed"]
    assert not wrong["inspection_valid"]
    assert "session_identity_mismatch" in {issue["code"] for issue in wrong["issues"]}
