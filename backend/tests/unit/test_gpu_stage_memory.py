"""G11 (2026-10-21): peak GPU memory per stage, measured path (a fake `torch.cuda`).

The CPU path and the wiring into the real probe run are asserted by
`test_probe_monitor_pipeline.py::...::test_GPU_MEMORY_is_recorded_for_every_timed_stage`.

Controls (review record `0xcc/reviews/2026-10-21_mcp-gaps.md`, restored from saved bytes):
  C-RESET  delete `reset_peak_memory_stats` in `start` → test_each_stage_reports_ITS_OWN_peak red
  C-WIRE   delete `memory["recorder"].start(name)` in probe_monitor_run.stage → pipeline test red
"""
from types import SimpleNamespace

from src.services.gpu_stage_memory import MIB, StagePeakMemory, cuda_device_of


class FakeCuda:
    """A high-water mark that only `reset_peak_memory_stats` lowers, like the real allocator."""

    def __init__(self):
        self.peak = 0
        self.resets = 0

    def allocate(self, mib):
        self.peak = max(self.peak, mib * MIB)

    def reset_peak_memory_stats(self, device):
        self.resets += 1
        self.peak = 0

    def max_memory_allocated(self, device):
        return self.peak

    def max_memory_reserved(self, device):
        return self.peak + 64 * MIB

    def get_device_properties(self, device):
        return SimpleNamespace(uuid="247aa582-0d1b-e161-8156-983ed1fefc57")


DEVICE = SimpleNamespace(type="cuda", index=0)


def test_each_stage_reports_ITS_OWN_peak():
    cuda = FakeCuda()
    rec = StagePeakMemory(DEVICE, "GPU-from-row", cuda=cuda)
    rec.start("capture")
    cuda.allocate(9000)
    rec.stop("capture")
    rec.start("training")
    cuda.allocate(1200)
    rec.stop("training")
    out = rec.record()
    assert out["measured"] is True
    assert out["stages"]["capture"]["peak_allocated_mib"] == 9000.0
    assert out["stages"]["training"]["peak_allocated_mib"] == 1200.0, (
        "without a reset per stage a later stage inherits the earlier peak")
    assert out["stages"]["training"]["peak_reserved_mib"] == 1264.0
    assert out["peak_allocated_mib"] == 9000.0 and out["peak_reserved_mib"] == 9064.0
    assert cuda.resets == 2


def test_the_card_uuid_is_torchs_for_the_device():
    rec = StagePeakMemory(DEVICE, "GPU-from-row", cuda=FakeCuda())
    assert rec.record()["gpu_uuid"] == "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"


def test_an_unclosed_stage_is_closed_by_the_next_start():
    cuda = FakeCuda()
    rec = StagePeakMemory(DEVICE, None, cuda=cuda)
    rec.start("a")
    cuda.allocate(10)
    rec.start("b")
    assert rec.record()["stages"]["a"]["peak_allocated_mib"] == 10.0


def test_a_stop_without_its_start_records_nothing():
    rec = StagePeakMemory(DEVICE, None, cuda=FakeCuda())
    rec.stop("never_started")
    assert rec.record()["stages"] == {}


def test_cpu_is_unmeasured_never_zero():
    cuda = FakeCuda()
    rec = StagePeakMemory(None, "GPU-x", cuda=cuda)
    rec.start("a")
    rec.stop("a")
    out = rec.record()
    assert out["measured"] is False and out["gpu_uuid"] == "GPU-x"
    assert out["stages"]["a"] == {"peak_allocated_mib": None, "peak_reserved_mib": None}
    assert cuda.resets == 0


def test_cuda_device_of():
    class M:
        def __init__(self, dev):
            self.dev = dev

        def parameters(self):
            return iter([SimpleNamespace(device=self.dev)])

    assert cuda_device_of(M(DEVICE)) is DEVICE
    assert cuda_device_of(M(SimpleNamespace(type="cpu"))) is None
    assert cuda_device_of(object()) is None
