"""Independent M2P reproduction. Physical GPU 1 is the only visible device."""
import os

GPU_UUID = "GPU-3d4c8194-7991-0a63-c517-075c83c0b393"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_UUID
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")

