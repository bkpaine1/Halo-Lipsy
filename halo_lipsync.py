"""
Halo-Lipsy: Native AMD Unified Memory Lip Sync for ComfyUI

Created by: Brent & Claude Code (Anthropic Claude Opus 4.5)
License: MIT
Version: 1.4.0

Built for AMD APUs with unified memory (Strix Halo, etc.) but works everywhere.
No subprocesses, no ghost files, no venv escapes. Just lip sync that works.
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

# Audio processing
import librosa
from scipy import signal

# ComfyUI
import folder_paths
import comfy.utils


# ============================================================================
# AUDIO PROCESSING
# ============================================================================

class AudioProcessor:
    """Mel spectrogram extraction for Wav2Lip"""

    def __init__(self):
        self.num_mels = 80
        self.sample_rate = 16000
        self.n_fft = 800
        self.hop_size = 200
        self.win_size = 800
        self.fmin = 55
        self.fmax = 7600
        self.preemphasis = 0.97
        self.ref_level_db = 20
        self.min_level_db = -100
        self.max_abs_value = 4.0
        self._mel_basis = None

    def load_wav(self, path_or_array, sr=None):
        if isinstance(path_or_array, (str, Path)):
            return librosa.core.load(str(path_or_array), sr=sr or self.sample_rate)[0]
        return path_or_array

    def _preemphasis(self, wav):
        return signal.lfilter([1, -self.preemphasis], [1], wav)

    def _stft(self, y):
        return librosa.stft(y=y, n_fft=self.n_fft, hop_length=self.hop_size,
                           win_length=self.win_size)

    def _build_mel_basis(self):
        return librosa.filters.mel(sr=self.sample_rate, n_fft=self.n_fft,
                                   n_mels=self.num_mels, fmin=self.fmin, fmax=self.fmax)

    def _linear_to_mel(self, spectrogram):
        if self._mel_basis is None:
            self._mel_basis = self._build_mel_basis()
        return np.dot(self._mel_basis, spectrogram)

    def _amp_to_db(self, x):
        min_level = np.exp(self.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def _normalize(self, S):
        return np.clip((2 * self.max_abs_value) *
                      ((S - self.min_level_db) / (-self.min_level_db)) - self.max_abs_value,
                      -self.max_abs_value, self.max_abs_value)

    def melspectrogram(self, wav):
        D = self._stft(self._preemphasis(wav))
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.ref_level_db
        return self._normalize(S)


# ============================================================================
# WAV2LIP MODEL
# ============================================================================

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)


class Wav2LipModel(nn.Module):
    """Wav2Lip neural network"""

    def __init__(self):
        super().__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3)),
            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                         Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                         Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),
            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                         Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                         Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                         Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),
            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                         Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                         Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),
            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                         Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                         Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),
            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                         Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),
            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
                         Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),
        ])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),
            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),
                         Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),
            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                         Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                         Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),
            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                         Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                         Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True)),
            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                         Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                         Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),
            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                         Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                         Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),
            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                         Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                         Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),
        ])

        self.output_block = nn.Sequential(
            Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, audio_sequences, face_sequences):
        B = audio_sequences.size(0)
        input_dim_size = len(face_sequences.size())

        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences)

        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            x = torch.cat((x, feats[-1]), dim=1)
            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)
            outputs = torch.stack(x, dim=2)
        else:
            outputs = x

        return outputs


# ============================================================================
# FACE DETECTION
# Priority: S3FD > MediaPipe > Haar (fallback)
# ============================================================================

class MediaPipeFaceDetector:
    """MediaPipe-based face detection - works well on AI-generated faces"""

    def __init__(self):
        import mediapipe as mp
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=1,  # Full-range model (better for various distances)
            min_detection_confidence=0.3
        )

    def detect_faces(self, images):
        results = []
        for img in images:
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

            # MediaPipe expects RGB
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            h, w = img.shape[:2]
            mp_results = self.detector.process(img)

            if mp_results.detections:
                # Take the highest confidence detection
                best = max(mp_results.detections, key=lambda d: d.score[0])
                bbox = best.location_data.relative_bounding_box

                # Convert relative coords to absolute
                x1 = max(0, int(bbox.xmin * w))
                y1 = max(0, int(bbox.ymin * h))
                x2 = min(w, int((bbox.xmin + bbox.width) * w))
                y2 = min(h, int((bbox.ymin + bbox.height) * h))

                # Expand bbox - minimal forehead (wastes 96px resolution), focus on mouth
                bw, bh = x2 - x1, y2 - y1
                expand_w = int(bw * 0.10)
                expand_h_top = int(bh * 0.15)  # Less forehead = more mouth pixels in 96x96
                expand_h_bot = int(bh * 0.10)

                x1 = max(0, x1 - expand_w)
                y1 = max(0, y1 - expand_h_top)
                x2 = min(w, x2 + expand_w)
                y2 = min(h, y2 + expand_h_bot)

                results.append((x1, y1, x2, y2))
            else:
                results.append(None)

        return results


class CPUFaceDetector:
    """CPU-only face detection using OpenCV Haar cascades (last resort fallback)"""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def detect_faces(self, images):
        results = []
        for img in images:
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)

            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img

            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                pad = int(0.15 * max(w, h))
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(img.shape[1], x + w + pad)
                y2 = min(img.shape[0], y + h + pad)
                results.append((x1, y1, x2, y2))
            else:
                results.append(None)

        return results


def get_face_detector():
    """Get face detector - tries S3FD > MediaPipe > OpenCV Haar"""
    # Try S3FD first (best quality)
    try:
        wav2lip_paths = [
            Path(__file__).parent / "Wav2Lip",
            Path(__file__).parent.parent / "ComfyUI_wav2lip" / "Wav2Lip",
        ]
        for wav2lip_path in wav2lip_paths:
            if wav2lip_path.exists() and str(wav2lip_path) not in sys.path:
                sys.path.insert(0, str(wav2lip_path))

        from face_detection import FaceAlignment, LandmarksType
        detector = FaceAlignment(LandmarksType._2D, flip_input=False, device='cpu')
        print("[Halo-Lipsy] Face detector: S3FD")
        return detector, 's3fd'
    except Exception:
        pass

    # Try MediaPipe (good for AI-generated faces)
    try:
        detector = MediaPipeFaceDetector()
        print("[Halo-Lipsy] Face detector: MediaPipe")
        return detector, 'mediapipe'
    except Exception:
        pass

    # Fall back to Haar cascades
    print("[Halo-Lipsy] Face detector: OpenCV Haar (least reliable)")
    return CPUFaceDetector(), 'opencv'


# ============================================================================
# HALO-LIPSY NODE
# ============================================================================

class HaloLipsy:
    """
    Halo-Lipsy: Native AMD Unified Memory Lip Sync

    No subprocesses. No ghost files. No venv escapes.
    Face detection on CPU, Wav2Lip on GPU (ROCm translates CUDA).
    Safe tensor casting for unified memory compatibility.
    """

    _model_cache = {}
    _face_detector = None
    _face_detector_type = None
    _audio_processor = None

    @classmethod
    def INPUT_TYPES(cls):
        checkpoint_paths = []
        search_paths = [
            Path(folder_paths.models_dir) / "wav2lip",
            Path(folder_paths.models_dir) / "Wav2Lip",
            Path(__file__).parent / "checkpoints",
            Path(__file__).parent.parent / "ComfyUI_wav2lip" / "Wav2Lip" / "checkpoints",
        ]

        for search_path in search_paths:
            if search_path.exists():
                for f in search_path.glob("*.pth"):
                    checkpoint_paths.append(str(f))

        checkpoint_options = ["auto"] + checkpoint_paths if checkpoint_paths else ["auto"]

        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
            },
            "optional": {
                "checkpoint": (checkpoint_options, {"default": "auto"}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.01,
                    "tooltip": "Video FPS - must match your input video for correct sync"}),
                "mode": (["sequential", "repetitive"], {"default": "sequential"}),
                "trim_to_audio": ("BOOLEAN", {"default": True,
                    "tooltip": "ON = trim video to audio length, OFF = keep full video (no lip sync after audio ends)"}),
                "face_detect_batch": ("INT", {"default": 4, "min": 1, "max": 32,
                    "tooltip": "Batch size for face detection (CPU)"}),
                "face_detect_interval": ("INT", {"default": 1, "min": 1, "max": 10,
                    "tooltip": "Detect face every Nth frame (interpolate between). Higher = faster, lower = more accurate"}),
                "inference_batch": ("INT", {"default": 64, "min": 1, "max": 256,
                    "tooltip": "Batch size for Wav2Lip inference (GPU)"}),
                "face_padding": ("INT", {"default": 5, "min": 0, "max": 50,
                    "tooltip": "Padding around detected face in pixels (less = more mouth resolution)"}),
                "sync_offset": ("INT", {"default": 0, "min": -10, "max": 10,
                    "tooltip": "Audio sync offset in frames (negative = audio earlier)"}),
                "mel_step_multiplier": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                    "tooltip": "Lip sync timing (>1 = faster mouth)"}),
                "smooth_box_frames": ("INT", {"default": 5, "min": 1, "max": 15,
                    "tooltip": "Frames for smoothing face box movement"}),
                "temporal_smooth": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.5, "step": 0.05,
                    "tooltip": "Temporal smoothing (0 = off, 0.2 = blend 20% of previous frame mouth)"}),
                "force_cpu": ("BOOLEAN", {"default": False,
                    "tooltip": "Run all inference on CPU (no VRAM)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "lipsync"
    CATEGORY = "Halo-Lipsy"

    def _find_checkpoint(self, checkpoint_path):
        if checkpoint_path != "auto" and os.path.exists(checkpoint_path):
            return checkpoint_path

        search_paths = [
            Path(__file__).parent / "checkpoints" / "wav2lip_gan.pth",
            Path(__file__).parent / "checkpoints" / "wav2lip.pth",
            Path(__file__).parent.parent / "ComfyUI_wav2lip" / "Wav2Lip" / "checkpoints" / "wav2lip_gan.pth",
            Path(folder_paths.models_dir) / "wav2lip" / "wav2lip_gan.pth",
        ]

        for path in search_paths:
            if path.exists():
                return str(path)

        raise FileNotFoundError(
            "Wav2Lip checkpoint not found. Please download wav2lip_gan.pth from:\n"
            "https://github.com/Rudrabha/Wav2Lip\n\n"
            "Place it in one of these locations:\n"
            f"  - {Path(__file__).parent}/checkpoints/\n"
            f"  - {folder_paths.models_dir}/wav2lip/"
        )

    def _load_model(self, checkpoint_path, force_cpu=False):
        use_fp16 = not force_cpu and torch.cuda.is_available()
        cache_key = f"{checkpoint_path}_{'cpu' if force_cpu else 'cuda'}_{'fp16' if use_fp16 else 'fp32'}"

        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        print(f"[Halo-Lipsy] Loading model: {Path(checkpoint_path).name}")

        model = Wav2LipModel()
        device = torch.device('cpu' if force_cpu else 'cuda')

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v

        model.load_state_dict(new_state_dict)
        model = model.to(device)

        if use_fp16:
            model = model.half()
            print("[Halo-Lipsy] Using FP16 inference (faster on GPU)")

        model.eval()

        self._model_cache[cache_key] = model
        return model

    def _get_face_detector(self):
        if self._face_detector is None:
            self._face_detector, self._face_detector_type = get_face_detector()
        return self._face_detector, self._face_detector_type

    def _get_audio_processor(self):
        if self._audio_processor is None:
            self._audio_processor = AudioProcessor()
        return self._audio_processor

    def _detect_faces(self, images, batch_size, interval=1):
        detector, detector_type = self._get_face_detector()
        num_frames = len(images)

        if interval <= 1:
            # Detect every frame
            sample_indices = list(range(num_frames))
        else:
            # Detect every Nth frame
            sample_indices = list(range(0, num_frames, interval))
            if sample_indices[-1] != num_frames - 1:
                sample_indices.append(num_frames - 1)

        sample_images = [images[i] for i in sample_indices]
        sample_results = []

        for i in range(0, len(sample_images), batch_size):
            batch = sample_images[i:i + batch_size]
            if detector_type == 's3fd':
                batch_np = np.array(batch)
                if batch_np.max() <= 1.0:
                    batch_np = (batch_np * 255).astype(np.uint8)
                results = detector.get_detections_for_batch(batch_np)
            else:
                results = detector.detect_faces(batch)
            sample_results.extend(results)

        # Interpolate between sampled detections for skipped frames
        all_results = [None] * num_frames
        for idx, sample_idx in enumerate(sample_indices):
            all_results[sample_idx] = sample_results[idx]

        if interval > 1:
            # Linear interpolation between detected frames
            for i in range(len(sample_indices) - 1):
                start_idx = sample_indices[i]
                end_idx = sample_indices[i + 1]
                start_box = all_results[start_idx]
                end_box = all_results[end_idx]

                if start_box is not None and end_box is not None:
                    for j in range(start_idx + 1, end_idx):
                        t = (j - start_idx) / (end_idx - start_idx)
                        interp = tuple(int(s + (e - s) * t) for s, e in zip(start_box, end_box))
                        all_results[j] = interp

        detected = sum(1 for r in all_results if r is not None)
        if interval > 1:
            print(f"[Halo-Lipsy] Face detection: {detected}/{num_frames} frames (sampled every {interval} frames)")
        else:
            print(f"[Halo-Lipsy] Face detection: {detected}/{num_frames} frames have faces")
        return all_results

    def _prepare_audio(self, audio_dict):
        waveform = audio_dict["waveform"]
        sample_rate = audio_dict["sample_rate"]

        if isinstance(waveform, torch.Tensor):
            waveform = waveform.float().cpu().numpy()

        if waveform.ndim == 3:
            waveform = waveform.squeeze(0)
        if waveform.ndim == 2:
            if waveform.shape[0] <= 2:
                waveform = waveform.mean(axis=0)
            else:
                waveform = waveform.squeeze()

        if sample_rate != 16000:
            import torchaudio
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(torch.from_numpy(waveform).float()).numpy()

        if np.abs(waveform).max() > 0:
            waveform = waveform / np.abs(waveform).max()

        return waveform.astype(np.float32)

    def _get_mel_chunks(self, audio, fps=30.0, mel_step_multiplier=1.0):
        processor = self._get_audio_processor()
        mel = processor.melspectrogram(audio)

        mel_step_size = 16
        # Use actual FPS instead of hardcoded 30
        mel_idx_multiplier = (80.0 / fps) * mel_step_multiplier

        mel_chunks = []
        i = 0
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > mel.shape[1]:
                mel_chunks.append(mel[:, mel.shape[1] - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx:start_idx + mel_step_size])
            i += 1

        return mel_chunks

    def _smooth_boxes(self, boxes, window_size=5):
        smoothed = []
        for i in range(len(boxes)):
            if boxes[i] is None:
                smoothed.append(None)
                continue

            # Check motion: compare to previous frame
            curr = np.array(boxes[i], dtype=np.float32)
            box_size = max(1, (curr[1] - curr[0] + curr[3] - curr[2]) / 2)

            # Adaptive window: shrink when face is moving
            effective_window = window_size
            if i > 0 and boxes[i - 1] is not None:
                prev = np.array(boxes[i - 1], dtype=np.float32)
                shift = np.sum(np.abs(curr - prev))
                motion_ratio = shift / box_size
                if motion_ratio > 0.1:
                    # Significant motion - use raw detection (no smoothing)
                    smoothed.append(boxes[i])
                    continue
                elif motion_ratio > 0.03:
                    # Moderate motion - reduce window
                    effective_window = max(1, window_size // 3)

            start = max(0, i - effective_window // 2)
            end = min(len(boxes), i + effective_window // 2 + 1)
            valid_boxes = [b for b in boxes[start:end] if b is not None]

            if valid_boxes:
                avg_box = np.mean(valid_boxes, axis=0).astype(int)
                smoothed.append(tuple(avg_box))
            else:
                smoothed.append(boxes[i])

        return smoothed

    def _pad_to_square(self, img):
        """Pad image to square preserving aspect ratio"""
        h, w = img.shape[:2]
        if h == w:
            return img, 0, 0
        size = max(h, w)
        pad_h = (size - h) // 2
        pad_w = (size - w) // 2
        if len(img.shape) == 3:
            padded = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
        else:
            padded = np.zeros((size, size), dtype=img.dtype)
        padded[pad_h:pad_h + h, pad_w:pad_w + w] = img
        return padded, pad_h, pad_w

    def _unpad_from_square(self, img, orig_h, orig_w, pad_h, pad_w):
        """Remove square padding to restore original aspect ratio"""
        return img[pad_h:pad_h + orig_h, pad_w:pad_w + orig_w]

    def _gradient_mouth_mask(self, height, width):
        """Ultra-tight lips-only mask - minimal Wav2Lip area to avoid 96x96 artifacts"""
        mask = np.zeros((height, width), dtype=np.float32)

        # Mouth center: 72% down the face, horizontally centered
        cy = int(height * 0.72)
        cx = width // 2

        # Just the lips - tight ellipse
        rx = int(width * 0.25)   # Lip width
        ry = int(height * 0.09)  # Lip height only

        # Create elliptical distance field
        yy, xx = np.ogrid[:height, :width]
        dist = ((xx - cx).astype(np.float32) / max(rx, 1)) ** 2 + \
               ((yy - cy).astype(np.float32) / max(ry, 1)) ** 2

        # Tight core with fast feather falloff
        inner = 0.4
        outer = 0.7
        mask[dist <= inner] = 1.0
        feather_zone = (dist > inner) & (dist <= outer)
        t = (dist[feather_zone] - inner) / (outer - inner)
        mask[feather_zone] = 0.5 * (1.0 + np.cos(np.pi * t))

        # Minimal anti-alias blur
        ksize = max(3, int(min(height, width) * 0.03) | 1)
        mask = cv2.GaussianBlur(mask, (ksize, ksize), ksize / 4.0)

        return np.stack([mask] * 3, axis=-1)

    def _color_match(self, source, target):
        """Match color/brightness of source to target using only the mouth region stats"""
        src = source.astype(np.float32)
        tgt = target.astype(np.float32)
        h = src.shape[0]

        # Only use the mouth region (60-90% of face height) for color stats
        # This prevents forehead/eye colors from skewing the mouth match
        mouth_top = int(h * 0.55)
        mouth_bot = int(h * 0.90)

        for c in range(3):
            src_region = src[mouth_top:mouth_bot, :, c]
            tgt_region = tgt[mouth_top:mouth_bot, :, c]
            src_mean, src_std = src_region.mean(), src_region.std() + 1e-6
            tgt_mean, tgt_std = tgt_region.mean(), tgt_region.std() + 1e-6
            src[:, :, c] = (src[:, :, c] - src_mean) * (tgt_std / src_std) + tgt_mean

        return np.clip(src, 0, 255).astype(np.uint8)

    def _sharpen(self, img, strength=0.3):
        """Mild unsharp mask to recover detail lost in upscale"""
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2)
        sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def lipsync(self, images, audio, checkpoint="auto", fps=30.0, mode="sequential",
                trim_to_audio=True, face_detect_batch=4, face_detect_interval=1,
                inference_batch=64, face_padding=5,
                sync_offset=0, mel_step_multiplier=1.0,
                smooth_box_frames=5, temporal_smooth=0.2, force_cpu=False):

        device_str = "CPU" if force_cpu else "GPU (FP16)"
        print(f"[Halo-Lipsy] v2.4 Processing {len(images)} frames @ {fps}fps on {device_str}")

        checkpoint_path = self._find_checkpoint(checkpoint)
        model = self._load_model(checkpoint_path, force_cpu=force_cpu)
        device = next(model.parameters()).device
        use_fp16 = next(model.parameters()).dtype == torch.float16

        if isinstance(images, torch.Tensor):
            images_np = images.float().cpu().numpy()
        else:
            images_np = np.array(images)

        if images_np.max() <= 1.0:
            images_uint8 = (images_np * 255).astype(np.uint8)
        else:
            images_uint8 = images_np.astype(np.uint8)

        # Prepare audio
        audio_wav = self._prepare_audio(audio)

        # Silence detection
        audio_energy = np.abs(audio_wav).mean()
        if audio_energy < 1e-5:
            print("[Halo-Lipsy] Audio is silent - skipping lip sync")
            output_images = torch.from_numpy(images_np.astype(np.float32))
            if output_images.max() > 1.0:
                output_images = output_images / 255.0
            return (output_images, audio)

        mel_chunks = self._get_mel_chunks(audio_wav, fps=fps, mel_step_multiplier=mel_step_multiplier)
        print(f"[Halo-Lipsy] {len(mel_chunks)} mel chunks @ {fps}fps")

        print("[Halo-Lipsy] Detecting faces (CPU)...")
        face_detections = self._detect_faces(list(images_uint8), face_detect_batch, interval=face_detect_interval)
        face_detections = self._smooth_boxes(face_detections, window_size=smooth_box_frames)

        valid_detections = [d for d in face_detections if d is not None]
        if len(valid_detections) == 0:
            print("[Halo-Lipsy] WARNING: No faces detected in any frame!")
            print("[Halo-Lipsy] Returning original video unchanged")
            output_images = torch.from_numpy(images_np.astype(np.float32))
            if output_images.max() > 1.0:
                output_images = output_images / 255.0
            return (output_images, audio)

        # Fill gaps: propagate nearest valid detection to frames with None
        last_valid = None
        for i in range(len(face_detections)):
            if face_detections[i] is not None:
                last_valid = face_detections[i]
            elif last_valid is not None:
                face_detections[i] = last_valid
        if face_detections[0] is None:
            first_valid = next(d for d in face_detections if d is not None)
            for i in range(len(face_detections)):
                if face_detections[i] is None:
                    face_detections[i] = first_valid
                else:
                    break

        img_size = 96
        frame_count = len(images_uint8)
        mel_count = len(mel_chunks)

        out_images = []
        all_data = []
        passthrough_frames = []

        if trim_to_audio:
            output_count = mel_count
            print(f"[Halo-Lipsy] Trimming to audio length ({mel_count} frames)")
        else:
            output_count = mel_count
            extra_frames = max(0, frame_count - mel_count)
            if extra_frames > 0:
                print(f"[Halo-Lipsy] Lip sync for {mel_count} frames, then {extra_frames} pass through")
            else:
                print(f"[Halo-Lipsy] Audio covers all {frame_count} frames")

        repeat_frames = mel_count / frame_count if frame_count > 0 else 1

        for mel_idx in range(output_count):
            adjusted_mel_idx = max(0, mel_idx + sync_offset)

            if mode == "sequential":
                frame_idx = min(int(adjusted_mel_idx / repeat_frames), frame_count - 1)
            else:
                frame_idx = adjusted_mel_idx % frame_count

            mel = mel_chunks[mel_idx]
            detection = face_detections[frame_idx]
            if detection is None:
                out_images.append(images_uint8[frame_idx].copy())
                continue

            # Check for silence in this mel chunk
            if np.abs(mel).mean() < 0.01:
                out_images.append(images_uint8[frame_idx].copy())
                continue

            x1, y1, x2, y2 = detection
            h, w = images_uint8[frame_idx].shape[:2]
            y1 = max(0, y1 - face_padding)
            y2 = min(h, y2 + face_padding)
            x1 = max(0, x1 - face_padding)
            x2 = min(w, x2 + face_padding)

            face_h, face_w = y2 - y1, x2 - x1
            if face_h < 20 or face_w < 20:
                out_images.append(images_uint8[frame_idx].copy())
                continue

            face = images_uint8[frame_idx][y1:y2, x1:x2]

            # Pad to square to preserve aspect ratio
            face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            face_square, pad_h, pad_w = self._pad_to_square(face_bgr)
            face_resized = cv2.resize(face_square, (img_size, img_size), interpolation=cv2.INTER_AREA)

            all_data.append({
                'face': face_resized,
                'mel': mel,
                'frame_idx': frame_idx,
                'coords': (y1, y2, x1, x2),
                'orig_h': face_h,
                'orig_w': face_w,
                'pad_h': pad_h,
                'pad_w': pad_w,
                'square_size': face_square.shape[0],
                'out_idx': len(out_images)
            })
            out_images.append(None)

        if not trim_to_audio and frame_count > mel_count:
            for frame_idx in range(mel_count, frame_count):
                passthrough_frames.append(images_uint8[frame_idx].copy())

        print(f"[Halo-Lipsy] Processing {len(all_data)} frames through Wav2Lip...")

        # ComfyUI progress bar
        pbar = comfy.utils.ProgressBar(len(all_data))
        prev_mouth = None  # For temporal smoothing
        prev_coords = None  # Track face movement

        for batch_start in tqdm(range(0, len(all_data), inference_batch), desc="Wav2Lip"):
            batch_end = min(batch_start + inference_batch, len(all_data))
            batch_data = all_data[batch_start:batch_end]

            img_batch = []
            mel_batch = []

            for item in batch_data:
                face = item['face']
                mel = item['mel']

                img_masked = face.copy()
                img_masked[img_size//2:] = 0

                img_concat = np.concatenate((img_masked, face), axis=2) / 255.0

                img_batch.append(img_concat)
                mel_batch.append(mel.reshape(mel.shape[0], mel.shape[1], 1))

            img_batch = np.array(img_batch)
            mel_batch = np.array(mel_batch)

            img_tensor = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_tensor = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

            if use_fp16:
                img_tensor = img_tensor.half()
                mel_tensor = mel_tensor.half()

            with torch.no_grad():
                pred = model(mel_tensor, img_tensor)

            pred = pred.float().cpu().numpy().transpose(0, 2, 3, 1) * 255.0

            for i, item in enumerate(batch_data):
                p = pred[i].clip(0, 255).astype(np.uint8)
                frame = images_uint8[item['frame_idx']].copy()
                y1, y2, x1, x2 = item['coords']
                target_w, target_h = x2 - x1, y2 - y1

                # Convert BGR→RGB
                p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)

                # Unpad from square: upscale to square size, then crop
                square_size = item['square_size']
                p_square = cv2.resize(p, (square_size, square_size), interpolation=cv2.INTER_LANCZOS4)
                p_face = self._unpad_from_square(p_square, item['orig_h'], item['orig_w'], item['pad_h'], item['pad_w'])

                # Resize to exact target if needed (rounding differences)
                if p_face.shape[0] != target_h or p_face.shape[1] != target_w:
                    p_face = cv2.resize(p_face, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

                # --- Lips-only replacement ---
                original_face = frame[y1:y2, x1:x2]
                mouth_mask = self._gradient_mouth_mask(target_h, target_w)

                # Color match prediction to original
                p_matched = self._color_match(p_face, original_face)

                # Sharpen to compensate for 96x96 upscale blur
                p_sharpened = self._sharpen(p_matched, strength=0.4)

                # Blend: only lips from Wav2Lip, everything else untouched
                blended_face = (p_sharpened.astype(np.float32) * mouth_mask +
                               original_face.astype(np.float32) * (1.0 - mouth_mask))
                blended_face = blended_face.clip(0, 255).astype(np.uint8)

                # Temporal smoothing - skip when face has moved significantly
                if temporal_smooth > 0 and prev_mouth is not None and prev_coords is not None:
                    py1, py2, px1, px2 = prev_coords
                    box_shift = abs(y1 - py1) + abs(x1 - px1) + abs(y2 - py2) + abs(x2 - px2)
                    box_size = max(1, (y2 - y1 + x2 - x1) // 2)
                    # Only smooth if movement is < 5% of box size
                    if box_shift < box_size * 0.05 and prev_mouth.shape == blended_face.shape:
                        blended_face = cv2.addWeighted(
                            blended_face, 1.0 - temporal_smooth,
                            prev_mouth, temporal_smooth, 0)
                prev_mouth = blended_face.copy()
                prev_coords = (y1, y2, x1, x2)

                # Place blended result directly - mouth mask already feathers to original
                frame[y1:y2, x1:x2] = blended_face

                out_images[item['out_idx']] = frame
                pbar.update(1)

        # Append pass-through frames
        if passthrough_frames:
            out_images.extend(passthrough_frames)
            synced = len(out_images) - len(passthrough_frames)
            print(f"[Halo-Lipsy] Done! {len(out_images)} total frames ({synced} synced + {len(passthrough_frames)} pass-through)")
        else:
            print(f"[Halo-Lipsy] Done! {len(out_images)} output frames")

        # Safety: fill any None placeholders
        final_images = []
        for idx, img in enumerate(out_images):
            if img is None:
                frame_idx = min(idx, frame_count - 1)
                final_images.append(images_uint8[frame_idx].copy())
            else:
                final_images.append(img)

        out_tensor_list = []
        for img in final_images:
            img_float = img.astype(np.float32) / 255.0
            out_tensor_list.append(torch.from_numpy(img_float))

        output_images = torch.stack(out_tensor_list, dim=0)

        return (output_images, audio)


# ============================================================================
# NODE REGISTRATION
# ============================================================================

__version__ = "2.4.0"
__author__ = "Brent & Claude Code"

NODE_CLASS_MAPPINGS = {
    "HaloLipsy": HaloLipsy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HaloLipsy": f"Halo-Lipsy v{__version__}",
}

# Startup message
print(f"[Halo-Lipsy] v{__version__} Loaded - AMD unified memory lip sync by Brent & Claude Code")
