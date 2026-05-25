#!/usr/bin/env python3
"""Test dual-head ONNX model in Whissle gateway (batch + streaming).

Usage:
    # 1. Start the gateway with the exported model:
    cd decoder_onnx
    ASR_MODEL_DIR=/path/to/export ASR_DEVICE=cpu python -m src.server --port 8001

    # 2. Run tests:
    python scripts/asr/meta-asr/test_onnx_dual_head.py --server http://localhost:8001

    # 3. With a specific test audio file:
    python scripts/asr/meta-asr/test_onnx_dual_head.py --server http://localhost:8001 --audio test.wav
"""

import argparse
import asyncio
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import requests


MISC_CATEGORIES = ['AGE', 'BEHAVIOR', 'EMOTION', 'EVAL', 'GENDER', 'ROLE']


def generate_test_audio(duration_sec=3.0, sample_rate=16000):
    """Generate a simple sine wave WAV for testing."""
    import io
    import wave

    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), dtype=np.float32)
    # Mix of frequencies to simulate speech-like energy
    audio = 0.3 * np.sin(2 * np.pi * 200 * t) + 0.2 * np.sin(2 * np.pi * 400 * t)
    audio += 0.1 * np.random.randn(len(t)).astype(np.float32)
    pcm = (audio * 16000).clip(-32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    buf.seek(0)
    return buf.read(), pcm.tobytes()


def test_health(server_url):
    """Test server health endpoint."""
    print("--- Health Check ---")
    r = requests.get(f"{server_url}/")
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    print(f"  OK: {r.json()}")


def test_status(server_url):
    """Test server status and verify tag classifier is loaded."""
    print("\n--- Status ---")
    r = requests.get(f"{server_url}/status")
    assert r.status_code == 200, f"Status failed: {r.status_code}"
    data = r.json()
    print(f"  Status: {data['status']}")
    print(f"  Models: {list(data.get('models', {}).keys())}")
    print(f"  Device: {data.get('device')}")
    return data


def test_batch_transcribe(server_url, audio_bytes, filename="test.wav"):
    """Test batch /transcribe endpoint with metadata extraction."""
    print("\n--- Batch Transcribe ---")
    files = {'file': (filename, audio_bytes, 'audio/wav')}
    data = {'metadata_prob': 'true', 'top_k': '5', 'word_timestamps': 'true'}

    start = time.time()
    r = requests.post(f"{server_url}/transcribe", files=files, data=data)
    elapsed = time.time() - start

    assert r.status_code == 200, f"Transcribe failed: {r.status_code} — {r.text}"
    result = r.json()

    print(f"  Transcript: {result.get('transcript', '')[:100]}")
    print(f"  Raw output: {result.get('raw_output', '')[:100]}")
    print(f"  Inference time: {result.get('inference_time', elapsed)}")

    metadata = result.get('metadata', {})
    print(f"  Metadata:")
    for key, val in sorted(metadata.items()):
        print(f"    {key}: {val}")

    # Check for tag classifier categories
    misc_keys = [k for k in metadata.keys() if k.upper() in MISC_CATEGORIES
                 or k in ('behavior', 'eval', 'role', 'age', 'gender', 'emotion')]
    if misc_keys:
        print(f"  Tag classifier predictions: {misc_keys}")
    else:
        print("  WARNING: No tag classifier predictions in metadata")

    entities = result.get('entities', [])
    if entities:
        print(f"  Entities: {entities[:3]}")

    metadata_probs = result.get('metadata_probs', {})
    if metadata_probs:
        for cat, probs in sorted(metadata_probs.items()):
            if isinstance(probs, list) and probs:
                top = probs[0]
                print(f"  {cat} probs: {top.get('token')}={top.get('probability', 0):.3f}")

    return result


def test_batch_transcribe_clean(server_url, audio_bytes, filename="test.wav"):
    """Test /transcribe/clean endpoint."""
    print("\n--- Clean Transcribe ---")
    files = {'file': (filename, audio_bytes, 'audio/wav')}
    r = requests.post(f"{server_url}/transcribe/clean", files=files)
    assert r.status_code == 200, f"Clean transcribe failed: {r.status_code}"
    result = r.json()
    print(f"  Clean transcript: {result.get('transcript', '')[:100]}")
    return result


async def test_streaming(server_url, pcm_bytes, sample_rate=16000):
    """Test WebSocket streaming endpoint."""
    print("\n--- Streaming Transcribe ---")
    try:
        import websockets
    except ImportError:
        print("  SKIPPED: websockets not installed (pip install websockets)")
        return None

    ws_url = server_url.replace('http://', 'ws://').replace('https://', 'wss://')
    ws_url = f"{ws_url}/stream"

    segments = []
    async with websockets.connect(ws_url) as ws:
        # Send config
        config = {
            'type': 'config',
            'language': 'en',
            'use_lm': True,
            'metadata_prob': True,
            'top_k': 5,
            'word_timestamps': True,
            'sample_rate': sample_rate,
        }
        await ws.send(json.dumps(config))
        print(f"  Config sent")

        # Send PCM in chunks (simulating real-time streaming)
        chunk_size = sample_rate * 2  # 1 second of 16-bit mono PCM
        total_sent = 0
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i:i + chunk_size]
            await ws.send(chunk)
            total_sent += len(chunk)

        print(f"  Sent {total_sent} bytes of PCM")

        # Send end signal
        await ws.send(json.dumps({'type': 'end'}))

        # Receive segments
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                data = json.loads(msg)
                if data.get('type') == 'end':
                    break
                if data.get('type') == 'transcript':
                    segments.append(data)
                    text = data.get('text', '')
                    meta = data.get('metadata', {})
                    is_final = data.get('is_final', False)
                    print(f"  Segment: text='{text[:60]}' final={is_final} meta={meta}")
            except asyncio.TimeoutError:
                print("  Timeout waiting for segments")
                break

    print(f"  Total segments received: {len(segments)}")

    # Check for tag classifier metadata in streaming segments
    for seg in segments:
        meta = seg.get('metadata', {})
        misc_keys = [k for k in meta.keys() if k in ('behavior', 'eval', 'role')]
        if misc_keys:
            print(f"  Tag classifier in streaming: {misc_keys}")
            break
    else:
        if segments:
            print("  WARNING: No tag classifier predictions in streaming segments")

    return segments


def main():
    parser = argparse.ArgumentParser(description='Test dual-head ONNX model')
    parser.add_argument('--server', default='http://localhost:8001',
                        help='Gateway server URL')
    parser.add_argument('--audio', help='Path to test audio WAV file')
    parser.add_argument('--skip-streaming', action='store_true',
                        help='Skip WebSocket streaming test')
    args = parser.parse_args()

    # Load or generate test audio
    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"ERROR: Audio file not found: {audio_path}")
            sys.exit(1)
        wav_bytes = audio_path.read_bytes()
        # Extract PCM for streaming (skip WAV header)
        import wave
        import io
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
            pcm_bytes = wf.readframes(wf.getnframes())
            sr = wf.getframerate()
        print(f"Using audio: {audio_path} ({len(wav_bytes)} bytes)")
    else:
        wav_bytes, pcm_bytes = generate_test_audio()
        sr = 16000
        print(f"Using generated test audio ({len(wav_bytes)} bytes)")

    # Run tests
    test_health(args.server)
    test_status(args.server)
    test_batch_transcribe(args.server, wav_bytes)
    test_batch_transcribe_clean(args.server, wav_bytes)

    if not args.skip_streaming:
        asyncio.run(test_streaming(args.server, pcm_bytes, sample_rate=sr))

    print("\n=== All tests completed ===")


if __name__ == '__main__':
    main()
