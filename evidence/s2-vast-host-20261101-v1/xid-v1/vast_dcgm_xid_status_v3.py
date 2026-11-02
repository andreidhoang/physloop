"""Read raw event-only field state; never convert an empty event history to zero."""

import json
import sys

sys.path.insert(0, "/usr/share/datacenter-gpu-manager-4/bindings/python3")
import dcgm_agent  # noqa: E402
import dcgm_fields  # noqa: E402
import dcgm_structs  # noqa: E402
import dcgmvalue  # noqa: E402

handle = group = fields = None
watched = False
dcgm_structs._dcgmInit()
dcgm_agent.dcgmInit()
try:
    handle = dcgm_agent.dcgmConnect("127.0.0.1")
    group = dcgm_agent.dcgmGroupCreate(
        handle, dcgm_structs.DCGM_GROUP_EMPTY, "physloop-xid-status"
    )
    dcgm_agent.dcgmGroupAddDevice(handle, group, 0)
    fields = dcgm_agent.dcgmFieldGroupCreate(handle, [230], "physloop-xid-only")
    dcgm_agent.dcgmWatchFields(handle, group, fields, 1000000, 60.0, 0)
    watched = True
    dcgm_agent.dcgmUpdateAllFields(handle, 1)
    value = dcgm_agent.dcgmEntityGetLatestValues(handle, dcgm_fields.DCGM_FE_GPU, 0, [230])[0]
    print(
        json.dumps(
            {
                "field_id": 230,
                "status": value.status,
                "timestamp": value.ts,
                "raw_int64": value.value.i64,
                "status_constants": {
                    key: getattr(dcgm_structs, key)
                    for key in (
                        "DCGM_ST_OK",
                        "DCGM_ST_NO_DATA",
                        "DCGM_ST_NOT_SUPPORTED",
                        "DCGM_ST_NOT_WATCHED",
                        "DCGM_ST_NO_PERMISSION",
                    )
                },
                "blank_constants": {
                    key: val
                    for key, val in vars(dcgmvalue).items()
                    if key.startswith("DCGM_INT64_") and isinstance(val, int)
                },
                "interpretation": "Raw field state only; event delivery remains unverified",
            },
            indent=2,
        )
    )
finally:
    if watched:
        dcgm_agent.dcgmUnwatchFields(handle, group, fields)
    if fields is not None:
        dcgm_agent.dcgmFieldGroupDestroy(handle, fields)
    if group is not None:
        dcgm_agent.dcgmGroupDestroy(handle, group)
    if handle is not None:
        dcgm_agent.dcgmDisconnect(handle)
    dcgm_agent.dcgmShutdown()
