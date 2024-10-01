"""
Synced with huggingface/pyspaces:spaces/zero/api.py
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any
from typing import Generator
from typing import Literal
from typing import NamedTuple
from typing import Optional
from typing import overload

import httpx
from pydantic import BaseModel
from typing_extensions import assert_never


AllowToken = str
NvidiaIndex = int # TODO: Migrate to GpuIndex (less confusing for MIG)
NvidiaUUID = str
CGroupPath = str
VisitorId = str
Score = float

AuthLevel = Literal['regular', 'pro']


AUTHENTICATED_HEADER = 'X-Authenticated'


class ScheduleResponse(BaseModel):
    idle: bool
    nvidiaIndex: int
    nvidiaUUID: str
    allowToken: str


class QuotaInfos(BaseModel):
    left: int
    wait: timedelta


class ReportUsageMonitoringParams(NamedTuple):
    nvidia_index: int
    visitor_id: str
    duration: timedelta


class QueueEvent(BaseModel):
    event: Literal['ping', 'failed', 'succeeded']
    data: Optional[ScheduleResponse] = None


def sse_parse(text: str):
    event, *data = text.strip().splitlines()
    assert event.startswith('event:')
    event = event[6:].strip()
    if event in ('ping', 'failed'):
        return QueueEvent(event=event)
    assert event == 'succeeded'
    (data,) = data
    assert data.startswith('data:')
    data = data[5:].strip()
    return QueueEvent(event=event, data=ScheduleResponse.parse_raw(data))


def sse_stream(res: httpx.Response) -> Generator[QueueEvent, Any, None]:
    for text in res.iter_text():
        if len(text) == 0:
            break # pragma: no cover
        try:
            yield sse_parse(text)
        except GeneratorExit:
            res.close()
            break


class APIClient:

    def __init__(self, client: httpx.Client):
        self.client = client

    def startup_report(self) -> httpx.codes:
        res = self.client.post('/startup-report')
        return httpx.codes(res.status_code)

    def schedule(
        self,
        cgroup_path: str,
        task_id: int = 0,
        token: str | None = None,
        duration_seconds: int | None = None,
        enable_queue: bool = True,
    ):
        params: dict[str, str | int | bool] = {
            'cgroupPath': cgroup_path,
            'taskId': task_id,
            'enableQueue': enable_queue,
        }
        if duration_seconds is not None:
            params['durationSeconds'] = duration_seconds
        if token is not None:
            params['token'] = token
        res = self.client.send(
            request=self.client.build_request(
                method='POST',
                url='/schedule',
                params=params,
            ),
            stream=True,
        )
        status = httpx.codes(res.status_code)
        auth: AuthLevel | None = res.headers.get(AUTHENTICATED_HEADER)
        if (status is not httpx.codes.OK and
            status is not httpx.codes.TOO_MANY_REQUESTS
        ):
            res.close()
            return status, auth
        if "text/event-stream" in res.headers['content-type']:
            return sse_stream(res), auth
        res.read()
        if status is httpx.codes.TOO_MANY_REQUESTS:
            return QuotaInfos(**res.json()), auth # pragma: no cover
        if status is httpx.codes.OK:
            return ScheduleResponse(**res.json()), auth
        assert_never(status)

    def allow(
        self,
        allow_token: str,
        pid: int,
    ):
        res = self.client.post('/allow', params={
            'allowToken': allow_token,
            'pid': pid,
        })
        return httpx.codes(res.status_code)

    def release(
        self,
        allow_token: str,
        fail: bool = False,
    ) -> httpx.codes:
        res = self.client.post('/release', params={
            'allowToken': allow_token,
            'fail': fail,
        })
        return httpx.codes(res.status_code)

    def get_queue_size(self) -> int:
        res = self.client.get('/queue-size')
        assert res.status_code == 200, res.status_code
        size = res.json()
        assert isinstance(size, int)
        return size
