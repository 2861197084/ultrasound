"""Independent M2P reproduction. Device selection belongs to run configuration."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")
