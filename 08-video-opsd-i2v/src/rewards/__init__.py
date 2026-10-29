import src.rewards.aesthetic  # noqa: F401
import src.rewards.dynamic_degree  # noqa: F401
import src.rewards.motion_smoothness  # noqa: F401
import src.rewards.naturalness_penalty  # noqa: F401
import src.rewards.subject_consistency  # noqa: F401
from .registry import REWARD_REGISTRY, build_reward, register
from .combined import CombinedReward

__all__ = ["REWARD_REGISTRY", "build_reward", "register", "CombinedReward"]
