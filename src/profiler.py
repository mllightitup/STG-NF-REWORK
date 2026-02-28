import csv
import threading
import time
from pathlib import Path
import torch

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("WARNING: 'psutil' is not installed. CPU and RAM profiling will be unavailable. Run `uv add psutil` to fix.")

class SystemProfiler:
    """Background threaded profiler logging CPU, RAM, and GPU/VRAM usage to a CSV."""
    
    def __init__(self, out_path: Path, interval: float = 1.0):
        self.out_path = out_path
        self.interval = interval
        self.running = False
        self.thread = None
        if HAS_PSUTIL:
            self.process = psutil.Process()

    def _monitor(self) -> None:
        with open(self.out_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Time(s)", "CPU(%)", "RAM_rss(MB)", "RAM_vms(MB)", 
                "GPU_Allocated(MB)", "GPU_Reserved(MB)"
            ])
            
            start_time = time.time()
            while self.running:
                proc_cpu = 0.0
                ram_rss = 0.0
                ram_vms = 0.0
                
                if HAS_PSUTIL:
                    try:
                        proc_cpu = self.process.cpu_percent(interval=None)
                        ram_info = self.process.memory_info()
                        ram_rss = ram_info.rss / (1024**2)
                        ram_vms = ram_info.vms / (1024**2)
                    except Exception:
                        pass
                
                gpu_alloc = 0.0
                gpu_res = 0.0
                if torch.cuda.is_available():
                    # Memory in MB
                    gpu_alloc = torch.cuda.memory_allocated() / (1024**2)
                    gpu_res = torch.cuda.memory_reserved() / (1024**2)
                    
                writer.writerow([
                    f"{time.time() - start_time:.1f}",
                    f"{proc_cpu:.1f}",
                    f"{ram_rss:.1f}",
                    f"{ram_vms:.1f}",
                    f"{gpu_alloc:.1f}",
                    f"{gpu_res:.1f}"
                ])
                f.flush()
                time.sleep(self.interval)

    def __enter__(self) -> "SystemProfiler":
        if HAS_PSUTIL:
            self.process.cpu_percent(interval=None) # Prime the CPU metric
        self.running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        print(f"[profiler] System profiling started. Saving to {self.out_path.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.running = False
        if self.thread:
            self.thread.join()
        print(f"[profiler] System profiling stopped.")
