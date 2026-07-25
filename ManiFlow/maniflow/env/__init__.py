# Both RoboCasa and LIBERO are optional external simulators (see
# docs/setup_and_train_robocasa.md / docs/setup_and_train_libero.md) — a
# server may only have one of them installed, so import each independently
# and don't let a missing one break the other.
try:
    from .robocasa import RoboCasaEnv
except ImportError:
    RoboCasaEnv = None

try:
    from .libero import LiberoEnv
except ImportError:
    LiberoEnv = None
