import json
import logging
import time
from collections import deque
from threading import Thread

import numpy as np
import redis
from redis.exceptions import RedisError

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import BallPoseRedisCtrlCfg

logger = logging.getLogger(__name__)


@ctrl_registry.register
class BallPoseRedisCtrl(Controller):
    """Consumes {"kick_ball_pos_b": [x,y,z], "kick_target_pos_b": [x,y], "t": unix_time} JSON
    published to a Redis key by a separate perception process (scripts/dummy_ball_perception.py for
    sim testing; a real onboard detector on the robot later, publishing to the same key/shape).
    Structurally identical to TwistRedisCtrl (background poll thread, bounded buffer, non-blocking
    get_data()) so it plugs into the exact same pipeline/CtrlManager machinery.

    Unlike TwistRedisCtrl, staleness is checked explicitly: if nothing has been published recently
    (producer not running, crashed, or a real detector losing track of the ball), get_data() reports
    valid=False rather than silently replaying the last-known position forever -- a stale ball
    reading is exactly the kind of thing that should fall back to "no detection", not a frozen fake
    one, especially once this same channel is feeding a real robot."""

    cfg_ctrl: BallPoseRedisCtrlCfg

    def __init__(self, cfg_ctrl: BallPoseRedisCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)

        self.redis_host = self.cfg_ctrl.redis_host
        self.redis_port = self.cfg_ctrl.redis_port
        self.redis_db = self.cfg_ctrl.redis_db
        self.redis_key = self.cfg_ctrl.redis_key
        self.buffer_size = self.cfg_ctrl.buffer_size
        self.stale_after_s = self.cfg_ctrl.stale_after_s

        self.data_buffer = deque(maxlen=self.buffer_size)
        self._stop = False
        self.redis_thread = Thread(target=self.redis_worker, daemon=True)
        self.redis_thread.start()

        self.last_data = None

        # wait for first data -- same blocking pattern as TwistRedisCtrl, but with a periodic
        # reminder since the usual reason this hangs is simply "forgot to start the perception
        # script in the other terminal", not a real connection problem.
        waited = 0.0
        while self.last_data is None:
            self.get_data()
            time.sleep(0.01)
            waited += 0.01
            if waited > 0 and int(waited) % 2 == 0 and abs(waited - round(waited)) < 0.01:
                logger.warning(
                    f"[BallPoseRedisCtrl] Still waiting for first ball-pose reading on redis key "
                    f"'{self.redis_key}' ({waited:.0f}s) -- is the perception script running? "
                    f"e.g. python scripts/dummy_ball_perception.py --redis-key {self.redis_key}"
                )
        logger.info("[BallPoseRedisCtrl] Initialized with first data.")

    def reset(self):
        self.data_buffer.clear()

    def redis_worker(self):
        """worker: try connect, poll, auto-reconnect -- mirrors TwistRedisCtrl.redis_worker."""
        redis_client = None
        while not self._stop:
            if redis_client is None:
                redis_client = self._connect_redis()

            try:
                payload_json = redis_client.get(self.redis_key)  # pyright: ignore[reportOptionalMemberAccess]
            except RedisError as e:
                logger.warning(f"[Redis] Lost connection: {e}, retrying...")
                redis_client = None
                continue

            if payload_json is not None:
                try:
                    payload = json.loads(payload_json)  # pyright: ignore[reportArgumentType]
                    reading = {
                        "kick_ball_pos_b": np.asarray(payload["kick_ball_pos_b"], dtype=np.float32),
                        "kick_target_pos_b": np.asarray(payload["kick_target_pos_b"], dtype=np.float32),
                        "t": float(payload["t"]),
                    }
                except (ValueError, KeyError, TypeError) as e:
                    logger.warning(f"[BallPoseRedisCtrl] Malformed payload on '{self.redis_key}': {e}")
                else:
                    self.data_buffer.append(reading)

            time.sleep(0.01)

    def _connect_redis(self):
        while not self._stop:
            try:
                redis_client = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    db=self.redis_db,
                    socket_timeout=1,
                    socket_connect_timeout=1,
                )
                redis_client.ping()
                logger.info("[Redis] Connected")
                return redis_client
            except Exception as e:
                logger.error(f"[Redis] Connect failed: {e}, retrying...")
                time.sleep(0.5)

    def get_data(self):
        """Pop oldest buffered reading -> last_data (non-blocking); report valid=False if nothing
        has ever arrived, or the freshest thing we have is older than stale_after_s."""
        if len(self.data_buffer) > 0:
            self.last_data = self.data_buffer.popleft()

        data = self.last_data
        valid = data is not None and (time.time() - data["t"]) <= self.stale_after_s
        return {
            "kick_ball_pos_b": data["kick_ball_pos_b"] if data is not None else None,
            "kick_target_pos_b": data["kick_target_pos_b"] if data is not None else None,
            "valid": valid,
        }


if __name__ == "__main__":
    ball_redis_ctrl = BallPoseRedisCtrl(cfg_ctrl=BallPoseRedisCtrlCfg(), env=None)
    for _ in range(10000):
        print(ball_redis_ctrl.get_data())
        print("================================")
        time.sleep(0.3)
