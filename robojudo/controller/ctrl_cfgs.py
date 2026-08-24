from robojudo.config import ASSETS_DIR, Config


class CtrlCfg(Config):
    ctrl_type: str  # name of the controller class

    triggers: dict[str, str] = {}  # trigger conditions
    triggers_extra: dict[str, str] = {}  # extra trigger conditions


class KeyboardCtrlCfg(CtrlCfg):
    ctrl_type: str = "KeyboardCtrl"

    combination_init_buttons: list[str] = ["Key.ctrl_l"]
    """first button in combination, need to be held down to trigger other commands;"""

    triggers: dict[str, str] = {
        "Key.esc": "[SHUTDOWN]",
        # "Key.tab": "[POLICY_TOGGLE]",
        "`": "[SIM_REBORN]",
        "<": "[MOTION_FADE_IN]",  # note: with shift
        ">": "[MOTION_FADE_OUT]",  # note: with shift
        "|": "[MOTION_RESET]",  # note: with shift
        "{": "[MOTION_LOAD_PREV]",  # note: with shift
        "}": "[MOTION_LOAD_NEXT]",  # note: with shift
    }


class JoystickCtrlCfg(CtrlCfg):
    ctrl_type: str = "JoystickCtrl"

    combination_init_buttons: list[str] = ["LB", "RB"]
    """first button in combination, need to be held down to trigger other commands;"""

    # reference for button names in JoystickThread config
    triggers: dict[str, str] = {
        "A": "[SHUTDOWN]",
        "X": "[MOTION_FADE_IN]",
        "B": "[MOTION_FADE_OUT]",
        "Y": "[MOTION_RESET]",
        # "LB": "[MOTION_LOAD_PREV]",
        # "RB": "[MOTION_LOAD_NEXT]",
        # Note: combo keys supported: "LB+RB+A": "[TEST]",
    }


class UnitreeCtrlCfg(JoystickCtrlCfg):
    ctrl_type: str = "UnitreeCtrl"

    combination_init_buttons: list[str] = ["L1", "R1"]
    """first button in combination, need to be held down to trigger other commands;"""

    triggers: dict[str, str] = {
        "A": "[SHUTDOWN]",
        "X": "[MOTION_FADE_IN]",
        "B": "[MOTION_FADE_OUT]",
        "Y": "[MOTION_RESET]",
        # Note: combo keys supported: "L1+R1+A": "[TEST]",
    }


class MotionCtrlCfg(CtrlCfg):
    class PhcCfg(Config):
        robot_config_file: str
        robot_config: dict = {}  # PLACEHOLDER for phc robot config, to be parsed by config manager

        def model_post_init(self, context) -> None:
            import yaml

            from robojudo.config import THIRD_PARTY_DIR

            # parse phc configs
            phc_dir_path = THIRD_PARTY_DIR / "phc"
            phc_robot_config_file = self.robot_config_file
            phc_robot_config_file_path = phc_dir_path / "phc/data/cfg" / phc_robot_config_file
            if phc_robot_config_file_path.exists():
                phc_robot_config_dict = yaml.safe_load(phc_robot_config_file_path.open("r"))
                phc_robot_config_dict["asset"]["assetRoot"] = phc_dir_path.as_posix()
                phc_robot_config_dict["asset"]["assetFileName"] = (
                    phc_dir_path / phc_robot_config_dict["asset"]["assetFileName"]
                ).as_posix()
                # phc_robot_config_dict["asset"]["urdfFileName"] = (
                #     phc_dir_path / phc_robot_config_dict["asset"]["urdfFileName"]
                # ).as_posix()

                self.robot_config = phc_robot_config_dict

    ctrl_type: str = "MotionCtrl"

    motion_ctrl_gui: bool = True

    # ==== policy specific configs ====
    track_keypoints_names: list[str] = []
    phc: PhcCfg

    # ==== motion config ====
    robot: str
    motion_name: str = ""

    @property
    def motion_path(self) -> str:
        motion_path = ASSETS_DIR / f"motions/{self.robot}/phc/{self.motion_name}.pkl"
        return motion_path.as_posix()


class MotionH2HCtrlCfg(MotionCtrlCfg):
    ctrl_type: str = "MotionH2HCtrl"

    extra_motion_data: bool = False  # extra data for motion recognition


class MotionKungfuBotCtrlCfg(MotionCtrlCfg):
    ctrl_type: str = "MotionKungfuBotCtrl"

    future_max_steps: int = 95
    future_num_steps: int = 20

    anchor_index: int = 0  # root
    key_body_id: list[int]


class MotionTwistCtrlCfg(MotionCtrlCfg):
    ctrl_type: str = "MotionTwistCtrl"

    # ==== motion config ====
    robot: str


class BeyondMimicCtrlCfg(CtrlCfg):
    ctrl_type: str = "BeyondMimicCtrl"

    override_robot_anchor_pos: bool = False  # if True, drop pos fdb

    # ==== motion config ====
    robot: str
    motion_name: str

    @property
    def motion_path(self) -> str:
        motion_path = ASSETS_DIR / f"motions/{self.robot}/beyondmimic/{self.motion_name}.npz"
        return motion_path.as_posix()

    # ==== from beyondmimic ====
    class MotionCommandCfg(Config):
        """Configuration for the motion command."""

        anchor_body_name: str
        body_names: list[str]
        body_names_all: list[str]
        """from beyondmimic asset, used for indexing"""

    motion_cfg: MotionCommandCfg


class TwistRedisCtrlCfg(CtrlCfg):
    ctrl_type: str = "TwistRedisCtrl"

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_key: str = "action_mimic_g1"  # key to get command data from redis

    buffer_size: int = 5  # size of the data buffer to store recent commands


class BallPoseRedisCtrlCfg(CtrlCfg):
    """Live kick_ball_pos_b/kick_target_pos_b from an external perception process, same
    Redis-poll pattern as TwistRedisCtrl. The producer is swappable -- scripts/dummy_ball_perception.py
    for sim testing today, a real onboard detector publishing to the same key on the robot later --
    with no change needed on the consumer (UnifiedLocoKickPolicy) side. See ball_pose_redis_ctrl.py."""

    ctrl_type: str = "BallPoseRedisCtrl"

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_key: str = "ball_pose_g1"  # key to get live ball-pose data from redis

    buffer_size: int = 5  # size of the data buffer to store recent readings
    stale_after_s: float = 0.5  # treat a reading older than this as "no detection" (falls back to zero)


class BallPoseRos2CtrlCfg(CtrlCfg):
    """ROS2 sibling of BallPoseRedisCtrlCfg -- same live kick_ball_pos_b/kick_target_pos_b contract,
    carried over two topics instead of one Redis key. See ball_pose_ros2_ctrl.py for why it's two
    topics (perception vs. a held aim command) and the QoS rationale."""

    ctrl_type: str = "BallPoseRos2Ctrl"

    node_name: str = "robojudo_ball_pose_ctrl"
    ball_topic: str = "/ball_pose"  # geometry_msgs/PointStamped, robot heading frame (x-fwd, z-up)
    aim_topic: str = "/kick_aim"  # geometry_msgs/Vector3Stamped, .x/.y used as kick_target_pos_b

    # None -> inherit the process's ROS_DOMAIN_ID env var (rclpy's own default). Set explicitly only
    # if this process must join a domain independent of its environment.
    domain_id: int | None = None

    qos_depth: int = 5  # matches BallPoseRedisCtrlCfg.buffer_size
    qos_reliable: bool = False  # BEST_EFFORT by default -- see module docstring for why

    stale_after_s: float = 0.5  # treat a ball reading older than this as "no detection" (falls back to zero)
