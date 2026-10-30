import struct

from parakeet_lite.audio import calculate_audio_level


def test_audio_level_silence():
    assert calculate_audio_level(struct.pack("<4h", 0, 0, 0, 0)) == 0


def test_audio_level_is_normalized():
    level = calculate_audio_level(struct.pack("<2h", 16384, -16384))
    assert 0.49 < level < 0.51


def test_empty_audio_level():
    assert calculate_audio_level(b"") == 0
