import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional

from .megadetector_video import MegaDetectorRunner, SpeciesNetRunner
from .pipeline import process_video


def _emit(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _emit_error(msg: str) -> None:
    sys.stderr.write(msg.rstrip() + "\n")
    sys.stderr.flush()


@dataclass
class Job:
    job_id: str
    video_path: str
    output_path: str


def main() -> int:
    md: MegaDetectorRunner | None = None
    species: SpeciesNetRunner | None = None
    ready = False

    # runtime parameters controlled by init
    confidence: float = float(os.environ.get("MEGADETECTOR_CONFIDENCE", "0.5"))
    frames_per_batch: int = int(os.environ.get("MEGADETECTOR_FRAMES_PER_BATCH", "8"))

    q: Deque[Job] = deque()
    shutting_down = False

    _emit({"type": "hello", "pid": os.getpid()})

    def handle_init(msg: Dict[str, Any]) -> None:
        nonlocal md, species, ready, confidence, frames_per_batch

        md_model_path = str(msg.get("md_model_path", "")).strip()
        cpu_cores = int(msg.get("cpu_cores", max(1, (os.cpu_count() or 4) - 1)))
        confidence = float(msg.get("confidence", confidence))
        frames_per_batch = int(msg.get("frames_per_batch", frames_per_batch))

        species_model_path = str(msg.get("species_model_path", "")).strip()
        species_labels_path = str(msg.get("species_labels_path", "")).strip()

        if not md_model_path:
            raise ValueError("init.md_model_path is required")
        if not os.path.exists(md_model_path):
            raise FileNotFoundError(f"MegaDetector model not found: {md_model_path}")

        _emit(
            {
                "type": "model_load_started",
                "model": "megadetector",
                "model_path": os.path.abspath(md_model_path),
                "cpu_cores": cpu_cores,
            }
        )
        t0 = time.perf_counter()
        md = MegaDetectorRunner(md_model_path, cpu_cores=cpu_cores)
        md_load_s = float(getattr(md, "load_seconds", time.perf_counter() - t0))
        _emit(
            {
                "type": "model_load_finished",
                "model": "megadetector",
                "model_path": os.path.abspath(md_model_path),
                "load_seconds": md_load_s,
                "ort_threads": getattr(md, "ort_threads", None),
            }
        )

        species = None
        species_info: Dict[str, Any] = {"enabled": False}
        if species_model_path:
            if not os.path.exists(species_model_path):
                raise FileNotFoundError(f"SpeciesNet model not found: {species_model_path}")
            if not species_labels_path:
                raise ValueError("init.species_labels_path is required when species_model_path is set")
            if not os.path.exists(species_labels_path):
                raise FileNotFoundError(f"SpeciesNet labels not found: {species_labels_path}")

            _emit(
                {
                    "type": "model_load_started",
                    "model": "speciesnet",
                    "model_path": os.path.abspath(species_model_path),
                    "labels_path": os.path.abspath(species_labels_path),
                }
            )
            t1 = time.perf_counter()
            species = SpeciesNetRunner(species_model_path, species_labels_path)
            sn_load_s = float(time.perf_counter() - t1)
            _emit(
                {
                    "type": "model_load_finished",
                    "model": "speciesnet",
                    "model_path": os.path.abspath(species_model_path),
                    "labels_path": os.path.abspath(species_labels_path),
                    "load_seconds": sn_load_s,
                }
            )
            species_info = {
                "enabled": True,
                "model_path": os.path.abspath(species_model_path),
                "labels_path": os.path.abspath(species_labels_path),
                "load_seconds": sn_load_s,
            }

        ready = True
        _emit(
            {
                "type": "ready",
                "md": {
                    "model_path": os.path.abspath(md_model_path),
                    "cpu_cores": cpu_cores,
                    "ort_threads": getattr(md, "ort_threads", None),
                    "load_seconds": md_load_s,
                },
                "speciesnet": species_info,
                "confidence": confidence,
                "frames_per_batch": frames_per_batch,
            }
        )

    def handle_enqueue(msg: Dict[str, Any]) -> None:
        nonlocal q
        job_id = str(msg.get("job_id", "")).strip()
        video_path = str(msg.get("video_path", "")).strip()
        output_path = str(msg.get("output_path", "")).strip()
        if not job_id:
            raise ValueError("enqueue.job_id is required")
        if not video_path:
            raise ValueError("enqueue.video_path is required")
        if not output_path:
            raise ValueError("enqueue.output_path is required")
        q.append(Job(job_id=job_id, video_path=video_path, output_path=output_path))
        _emit({"type": "enqueued", "job_id": job_id, "queue_len": len(q)})

    def pump_queue() -> None:
        nonlocal q
        if not ready or md is None:
            return
        if not q:
            return
        job = q.popleft()
        _emit(
            {
                "type": "job_started",
                "job_id": job.job_id,
                "video_path": job.video_path,
                "output_path": job.output_path,
            }
        )

        try:
            if not os.path.exists(job.video_path):
                raise FileNotFoundError(f"Video not found: {job.video_path}")

            def on_progress(evt: Dict[str, Any]) -> None:
                if evt.get("type") == "progress":
                    _emit(
                        {
                            "type": "job_progress",
                            "job_id": job.job_id,
                            "sampled_frames": int(evt.get("sampled_frames", 0)),
                        }
                    )
                elif evt.get("type") == "batch_done":
                    # Log batch timings to the terminal (stderr), keep stdout strictly JSON events.
                    _emit_error(
                        "batch_done "
                        f"job_id={job.job_id} "
                        f"batch={int(evt.get('batches', 0))} "
                        f"B={int(evt.get('batch_size', 0))} "
                        f"onnx={float(evt.get('onnx_infer_s', 0.0)):.3f}s "
                        f"post={float(evt.get('post_s', 0.0)):.3f}s "
                        f"mode={evt.get('onnx_mode', '')} "
                        f"frames_written_total={int(evt.get('frames_written_total', 0))}"
                    )

            t_job0 = time.perf_counter()
            process_video(
                video_path=job.video_path,
                output_path=job.output_path,
                md_runner=md,
                species_runner=species,
                confidence_threshold=confidence,
                frames_per_batch=frames_per_batch,
                sample_rate_hz=1.0,
                on_progress=on_progress,
            )
            _emit(
                {
                    "type": "job_finished",
                    "job_id": job.job_id,
                    "output_path": job.output_path,
                    "elapsed_seconds": float(time.perf_counter() - t_job0),
                }
            )
        except Exception as e:
            _emit({"type": "job_failed", "job_id": job.job_id, "error": str(e)})

    # Simple single-threaded event loop:
    # - read one line if available (blocking)
    # - after processing that message, try to pump one job
    #
    # This keeps implementation small; GUI can enqueue all jobs and then worker will drain.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            pump_queue()
            continue
        try:
            msg = json.loads(line)
        except Exception as e:
            _emit({"type": "protocol_error", "error": f"invalid_json: {e}"})
            continue

        mtype = msg.get("type")
        try:
            if mtype == "init":
                handle_init(msg)
            elif mtype == "enqueue":
                if not ready:
                    raise RuntimeError("worker not ready; send init first")
                handle_enqueue(msg)
            elif mtype == "shutdown":
                shutting_down = True
                _emit({"type": "shutdown_ack"})
                break
            else:
                _emit({"type": "protocol_error", "error": f"unknown_type: {mtype}"})
        except Exception as e:
            _emit({"type": "protocol_error", "error": str(e), "in_reply_to": mtype})

        pump_queue()

        if shutting_down:
            break

    # Drain remaining jobs if stdin closes unexpectedly? Keep it simple: stop.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

