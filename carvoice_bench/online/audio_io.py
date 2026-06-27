"""Optional local playback and microphone recording for online tests."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class OnlineAudioIO:
    """Play test audio and record microphone input when optional deps exist."""

    def play(self, audio_path: str | Path) -> dict:
        try:
            from pydub import AudioSegment
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "Audio playback requires pydub and sounddevice. Install "
                '`pip install -e ".[audio,online]"`.'
            ) from exc

        segment = AudioSegment.from_file(str(audio_path))
        samples = segment.get_array_of_samples()
        import numpy as np

        audio = np.array(samples, dtype=np.float32)
        if segment.channels > 1:
            audio = audio.reshape((-1, segment.channels))
        audio /= float(1 << (8 * segment.sample_width - 1))
        sd.play(audio, samplerate=segment.frame_rate)
        sd.wait()
        logger.info("played audio: %s", audio_path)
        return {"played": True, "audio_path": str(audio_path), "sample_rate": segment.frame_rate}

    def capture_case(
        self,
        playback_path: str | Path,
        mic_output_path: str | Path | None = None,
        seconds: float = 0.0,
        sample_rate: int = 16000,
        video_output_path: str | Path | None = None,
        camera_index: int = 0,
        video_fps: float = 20.0,
    ) -> dict:
        """Play audio while optionally recording microphone audio and PC camera video."""
        segment, playback_audio = self._load_playback(
            playback_path,
            target_sample_rate=sample_rate if mic_output_path else None,
        )
        duration_sec = max(float(seconds or 0.0), segment.duration_seconds, 0.1)
        capture_result = {
            "playback": {},
            "recording": {},
            "video": {},
            "duration_sec": round(duration_sec, 3),
        }

        video_thread = None
        video_result: dict = {}
        stop_video = threading.Event()
        if video_output_path:
            video_thread = threading.Thread(
                target=_record_video_worker,
                args=(Path(video_output_path), duration_sec, camera_index, video_fps, stop_video, video_result),
                daemon=True,
            )
            video_thread.start()

        mic_audio = None
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "Playback/recording requires sounddevice. Install "
                '`pip install -e ".[audio,online]"`.'
            ) from exc

        if mic_output_path:
            try:
                import soundfile as sf
            except ImportError as exc:
                raise RuntimeError("Microphone recording requires soundfile.") from exc
            mic_output = Path(mic_output_path)
            mic_output.parent.mkdir(parents=True, exist_ok=True)
            frames = int(duration_sec * sample_rate)
            playback_for_recording = _pad_audio(playback_audio, frames)
            mic_audio = sd.playrec(
                playback_for_recording,
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            sf.write(str(mic_output), mic_audio, sample_rate)
            capture_result["recording"] = {
                "recorded": True,
                "audio_path": str(mic_output),
                "duration_sec": round(duration_sec, 3),
                "sample_rate": sample_rate,
            }
        else:
            sd.play(playback_audio, samplerate=segment.frame_rate)
            sd.wait()

        if video_thread is not None:
            video_thread.join(timeout=duration_sec + 3)
            stop_video.set()
            capture_result["video"] = video_result

        capture_result["playback"] = {
            "played": True,
            "audio_path": str(playback_path),
            "sample_rate": segment.frame_rate,
            "duration_sec": round(segment.duration_seconds, 3),
        }
        logger.info("captured online case playback=%s mic=%s video=%s", playback_path, mic_output_path, video_output_path)
        return capture_result

    def record(self, output_path: str | Path, seconds: float, sample_rate: int = 16000) -> dict:
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError(
                "Microphone recording requires sounddevice and soundfile. Install "
                '`pip install -e ".[audio,online]"`.'
            ) from exc

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames = int(max(seconds, 0.1) * sample_rate)
        audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        sf.write(str(output_path), audio, sample_rate)
        logger.info("recorded microphone audio: %s", output_path)
        return {
            "recorded": True,
            "audio_path": str(output_path),
            "duration_sec": round(frames / sample_rate, 3),
            "sample_rate": sample_rate,
        }

    def _load_playback(self, audio_path: str | Path, target_sample_rate: int | None = None):
        try:
            from pydub import AudioSegment
        except ImportError as exc:
            raise RuntimeError(
                "Audio playback requires pydub. Install "
                '`pip install -e ".[audio,online]"`.'
            ) from exc

        segment = AudioSegment.from_file(str(audio_path))
        if target_sample_rate:
            segment = segment.set_frame_rate(target_sample_rate)
        samples = segment.get_array_of_samples()
        import numpy as np

        audio = np.array(samples, dtype=np.float32)
        if segment.channels > 1:
            audio = audio.reshape((-1, segment.channels))
        audio /= float(1 << (8 * segment.sample_width - 1))
        return segment, audio


def _pad_audio(audio, frames: int):
    import numpy as np

    if len(audio) >= frames:
        return audio[:frames]
    pad_shape = (frames - len(audio),) if audio.ndim == 1 else (frames - len(audio), audio.shape[1])
    padding = np.zeros(pad_shape, dtype=audio.dtype)
    return np.concatenate([audio, padding], axis=0)


def _record_video_worker(
    output_path: Path,
    seconds: float,
    camera_index: int,
    fps: float,
    stop_event: threading.Event,
    result: dict,
) -> None:
    try:
        import cv2
    except ImportError:
        result.update({"captured": False, "reason": "opencv-python is not installed"})
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        result.update({"captured": False, "reason": f"camera {camera_index} could not be opened"})
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = float(fps or 20.0)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    start = time.perf_counter()
    frames = 0
    try:
        while not stop_event.is_set() and (time.perf_counter() - start) < seconds:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            frames += 1
            time.sleep(max(0.0, 1.0 / fps))
    finally:
        writer.release()
        cap.release()

    result.update({
        "captured": frames > 0,
        "video_path": str(output_path) if frames > 0 else "",
        "duration_sec": round(time.perf_counter() - start, 3),
        "frames": frames,
        "camera_index": camera_index,
        "fps": fps,
        "resolution": f"{width}x{height}",
    })
