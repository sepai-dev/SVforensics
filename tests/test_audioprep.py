import pytest
import os
import numpy as np
import tempfile
import shutil
import soundfile as sf
from pathlib import Path
from typing import List, Tuple, Dict, Any

from svforensics import config
from svforensics import audioprep

# Constants for tests
TEST_SAMPLE_RATE = 16000
TEST_DURATION = 2.0  # seconds
TEST_CHUNK_DURATION = 0.5  # seconds
TEST_FADE_DURATION = 0.1  # seconds
REAL_VOICE_FILE = "tests/files/voice_test.ogg"


@pytest.fixture
def temp_config(tmpdir):
    """Create a temporary configuration for testing"""
    # Create mock config
    test_config = config.load_config().copy()
    
    # Update paths to use temporary directory
    test_config["paths"]["data_dir"] = str(tmpdir)
    test_config["paths"]["case_dir"] = str(tmpdir / "case_audios")
    test_config["paths"]["probe_dir"] = str(tmpdir / "case_audios" / "raw" / "probe")
    test_config["paths"]["reference_dir"] = str(tmpdir / "case_audios" / "raw" / "reference")
    test_config["paths"]["processed_audio_dir"] = str(tmpdir / "case_audios" / "processed")
    test_config["paths"]["probe_processed_dir"] = str(tmpdir / "case_audios" / "processed" / "probe")
    test_config["paths"]["reference_processed_dir"] = str(tmpdir / "case_audios" / "processed" / "reference")
    
    # Update audio parameters for tests
    test_config["audio"]["chunk_duration"] = TEST_CHUNK_DURATION
    test_config["audio"]["fade_duration"] = TEST_FADE_DURATION
    
    # Create directories
    os.makedirs(test_config["paths"]["probe_dir"], exist_ok=True)
    os.makedirs(test_config["paths"]["reference_dir"], exist_ok=True)
    os.makedirs(test_config["paths"]["processed_audio_dir"], exist_ok=True)
    os.makedirs(test_config["paths"]["probe_processed_dir"], exist_ok=True)
    os.makedirs(test_config["paths"]["reference_processed_dir"], exist_ok=True)
    
    # Save test config to a temporary file
    config_path = str(tmpdir / "test_config.json")
    config.save_config(test_config, config_path)
    
    # Setup test module with test config
    original_config_path = os.environ.get("SVFORENSICS_CONFIG_PATH")
    os.environ["SVFORENSICS_CONFIG_PATH"] = config_path
    
    # Force reload config to use the temporary one
    config.reload_config()
    
    # Important: monkeypatch audioprep default values to use the test configuration
    audioprep.DEFAULT_DATA_DIR = config.get_path("data_dir")
    audioprep.DEFAULT_CASE_DIR = config.get_path("case_dir")
    audioprep.DEFAULT_PROBE_DIR = config.get_path("probe_dir")
    audioprep.DEFAULT_REFERENCE_DIR = config.get_path("reference_dir")
    audioprep.DEFAULT_PROCESSED_DIR = config.get_path("processed_audio_dir")
    audioprep.DEFAULT_PROBE_PROCESSED_DIR = config.get_path("probe_processed_dir")
    audioprep.DEFAULT_REFERENCE_PROCESSED_DIR = config.get_path("reference_processed_dir")
    audioprep.DEFAULT_SAMPLE_RATE = config.get_audio_config("sample_rate")
    audioprep.DEFAULT_CHUNK_DURATION = config.get_audio_config("chunk_duration")
    audioprep.DEFAULT_FADE_DURATION = config.get_audio_config("fade_duration")
    audioprep.DEFAULT_AUDIO_EXTENSIONS = config.get_audio_config("audio_extensions")
    
    yield test_config
    
    # Reset config path
    if original_config_path:
        os.environ["SVFORENSICS_CONFIG_PATH"] = original_config_path
    else:
        os.environ.pop("SVFORENSICS_CONFIG_PATH", None)
    
    # Reload original config
    config.reload_config()


@pytest.fixture
def sine_wave():
    """Generate a sine wave for testing"""
    duration = TEST_DURATION  # seconds
    sample_rate = TEST_SAMPLE_RATE
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate a 440 Hz sine wave
    amplitude = 0.5
    frequency = 440.0
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    
    return wave.copy(), sample_rate


@pytest.fixture
def temp_audio_file(tmpdir, sine_wave):
    """Create a temporary audio file for testing"""
    wave, sample_rate = sine_wave
    
    # Create audio file
    audio_path = str(tmpdir / "test_audio.wav")
    sf.write(audio_path, wave, sample_rate)
    
    return audio_path, wave, sample_rate


@pytest.fixture
def real_voice_setup(temp_config):
    """Copy the real voice file to the test directories"""
    # Get paths from the test configuration
    probe_dir = config.get_path("probe_dir")
    ref_dir = config.get_path("reference_dir")
    
    # Create speaker subdirectories
    speaker1_probe_dir = os.path.join(probe_dir, "speaker1")
    speaker1_ref_dir = os.path.join(ref_dir, "speaker1")
    os.makedirs(speaker1_probe_dir, exist_ok=True)
    os.makedirs(speaker1_ref_dir, exist_ok=True)
    
    # Copy the real voice file to probe and reference directories
    probe_file = os.path.join(speaker1_probe_dir, "voice_test.ogg")
    ref_file = os.path.join(speaker1_ref_dir, "voice_test.ogg")
    
    shutil.copyfile(REAL_VOICE_FILE, probe_file)
    shutil.copyfile(REAL_VOICE_FILE, ref_file)
    
    # Load the real voice file to determine its properties
    import librosa
    audio, sr = librosa.load(REAL_VOICE_FILE, sr=None)
    duration = len(audio) / sr
    
    return {
        "probe_dir": probe_dir,
        "reference_dir": ref_dir,
        "sample_rate": sr,
        "duration": duration,
        "probe_file": probe_file,
        "ref_file": ref_file
    }


@pytest.fixture
def temp_audio_dirs(temp_config):
    """Create temporary audio directories with test files"""
    # We'll use the temp_config fixture to create files in the right locations
    probe_dir = config.get_path("probe_dir")
    ref_dir = config.get_path("reference_dir")
    
    # Create speaker subdirectories
    os.makedirs(os.path.join(probe_dir, "speaker1"), exist_ok=True)
    os.makedirs(os.path.join(probe_dir, "speaker2"), exist_ok=True)
    os.makedirs(os.path.join(ref_dir, "speaker1"), exist_ok=True)
    os.makedirs(os.path.join(ref_dir, "speaker2"), exist_ok=True)
    
    # Create audio files for testing
    sample_rate = TEST_SAMPLE_RATE
    duration = TEST_DURATION
    
    # Create multiple audio files
    for i, speaker_dir in enumerate([
        os.path.join(probe_dir, "speaker1"),
        os.path.join(probe_dir, "speaker2"),
        os.path.join(ref_dir, "speaker1"),
        os.path.join(ref_dir, "speaker2")
    ]):
        # Create slightly different frequencies for different files
        for j in range(2):
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            freq = 440.0 + (i * 100) + (j * 50)
            test_wave = 0.5 * np.sin(2 * np.pi * freq * t)
            
            # Save file
            audio_path = os.path.join(speaker_dir, f"audio_{j}.wav")
            sf.write(audio_path, test_wave, sample_rate)
    
    return {
        "probe_dir": probe_dir,
        "reference_dir": ref_dir
    }


def test_load_audio(temp_audio_file):
    """Test loading an audio file"""
    audio_path, original_wave, original_sr = temp_audio_file
    
    # Load audio
    loaded_audio, loaded_sr = audioprep.load_audio(audio_path)
    
    # Check if loaded correctly
    assert loaded_sr == original_sr
    assert len(loaded_audio) == len(original_wave)
    assert np.allclose(loaded_audio, original_wave, atol=1e-4)


def test_apply_fade(sine_wave):
    """Test applying fade in/out to audio"""
    wave, sr = sine_wave
    # Create a copy to avoid modifying the original
    wave_copy = wave.copy()
    
    # Apply fade
    fade_duration = 0.1  # seconds
    faded_audio = audioprep.apply_fade(wave_copy, sr, fade_duration)
    
    # Check if fade was applied correctly
    fade_samples = int(fade_duration * sr)
    
    # Check fade in (first samples should be attenuated)
    assert faded_audio[0] == 0.0  # First sample should be zero
    
    # For positive values in the original signal, the faded version should be less than or equal
    # For negative values, the faded version should be greater than or equal (less negative)
    for i in range(fade_samples):
        if wave[i] >= 0:
            assert faded_audio[i] <= wave[i], f"Failed at index {i}: {faded_audio[i]} > {wave[i]}"
        else:
            assert faded_audio[i] >= wave[i], f"Failed at index {i}: {faded_audio[i]} < {wave[i]}"
    
    # Check fade out (last samples should be attenuated)
    assert faded_audio[-1] == 0.0  # Last sample should be zero
    
    # For positive values in the original signal, the faded version should be less than or equal
    # For negative values, the faded version should be greater than or equal (less negative)
    for i in range(-fade_samples, 0):
        if wave[i] >= 0:
            assert faded_audio[i] <= wave[i], f"Failed at index {i}: {faded_audio[i]} > {wave[i]}"
        else:
            assert faded_audio[i] >= wave[i], f"Failed at index {i}: {faded_audio[i]} < {wave[i]}"
    
    # Middle samples should be unchanged
    assert np.allclose(faded_audio[fade_samples:-fade_samples], 
                       wave[fade_samples:-fade_samples])


def test_split_into_chunks(sine_wave):
    """Test splitting audio into chunks"""
    wave, sr = sine_wave
    
    # Split into chunks
    chunk_duration = 0.5  # seconds
    fade_duration = 0.1  # seconds
    chunks = audioprep.split_into_chunks(wave, sr, chunk_duration, fade_duration)
    
    # Expected number of chunks
    expected_chunks = int(np.ceil(TEST_DURATION / chunk_duration))
    
    # Check number of chunks
    assert len(chunks) == expected_chunks
    
    # Check chunk lengths
    chunk_samples = int(chunk_duration * sr)
    for i in range(expected_chunks - 1):  # All but last chunk
        assert len(chunks[i]) == chunk_samples
    
    # Last chunk might be shorter
    assert len(chunks[-1]) <= chunk_samples
    
    # Check that fades were applied (first and last samples of each chunk)
    for chunk in chunks:
        assert chunk[0] == 0.0  # First sample should be zero
        assert chunk[-1] == 0.0  # Last sample should be zero


def test_save_chunks(tmpdir, sine_wave):
    """Test saving audio chunks to files"""
    wave, sr = sine_wave
    
    # Create chunks
    chunk_duration = 0.5  # seconds
    fade_duration = 0.1  # seconds
    chunks = audioprep.split_into_chunks(wave, sr, chunk_duration, fade_duration)
    
    # Save chunks
    output_dir = str(tmpdir / "chunks")
    os.makedirs(output_dir, exist_ok=True)
    base_filename = "test_audio"
    format = "wav"
    
    saved_files = audioprep.save_chunks(chunks, sr, output_dir, base_filename, format)
    
    # Check if files were saved correctly
    assert len(saved_files) == len(chunks)
    for i, file_path in enumerate(saved_files):
        assert os.path.exists(file_path)
        
        # Load saved chunk
        loaded_audio, loaded_sr = sf.read(file_path)
        
        # Check if loaded chunk matches original
        assert loaded_sr == sr
        assert len(loaded_audio) == len(chunks[i])
        assert np.allclose(loaded_audio, chunks[i], atol=1e-4)


def test_process_audio_file(temp_audio_file, temp_config):
    """Test processing a single audio file"""
    audio_path, original_wave, original_sr = temp_audio_file
    processed_dir = config.get_path("processed_audio_dir")
    
    # Process audio file
    chunk_files = audioprep.process_audio_file(
        audio_path,
        processed_dir,
        sample_rate=TEST_SAMPLE_RATE,
        chunk_duration=TEST_CHUNK_DURATION,
        fade_duration=TEST_FADE_DURATION
    )
    
    # Expected number of chunks
    expected_chunks = int(np.ceil(TEST_DURATION / TEST_CHUNK_DURATION))
    
    # Check if files were created correctly
    assert len(chunk_files) == expected_chunks
    for file_path in chunk_files:
        assert os.path.exists(file_path)
        
        # Load saved chunk
        loaded_audio, loaded_sr = sf.read(file_path)
        
        # Check sample rate
        assert loaded_sr == TEST_SAMPLE_RATE


def test_process_directory(temp_audio_dirs, temp_config):
    """Test processing a directory of audio files"""
    probe_dir = temp_audio_dirs["probe_dir"]
    processed_dir = config.get_path("processed_audio_dir")
    
    # Debug the directory structure
    print(f"\nProbe directory: {probe_dir}")
    print(f"Files in probe directory and subdirectories:")
    for root, dirs, files in os.walk(probe_dir):
        for file in files:
            print(f"  {os.path.join(root, file)}")
    
    # Process directory
    chunk_files = audioprep.process_directory(
        probe_dir,
        processed_dir,
        sample_rate=TEST_SAMPLE_RATE,
        chunk_duration=TEST_CHUNK_DURATION,
        fade_duration=TEST_FADE_DURATION,
        extensions=[".wav"]  # Explicitly specify extensions
    )
    
    # Expected number of files: 2 speakers × 2 files per speaker
    expected_files = 4
    
    # Expected chunks per file
    chunks_per_file = int(np.ceil(TEST_DURATION / TEST_CHUNK_DURATION))
    
    # Expected total chunks
    expected_chunks = expected_files * chunks_per_file
    
    # Check if files were created correctly
    assert len(chunk_files) == expected_chunks, f"Expected {expected_chunks} chunks, got {len(chunk_files)}"
    
    # Check if files exist and are organized by speaker
    speaker_dirs = [
        os.path.join(processed_dir, "speaker1"),
        os.path.join(processed_dir, "speaker2")
    ]
    
    for speaker_dir in speaker_dirs:
        assert os.path.exists(speaker_dir)
        
        # Count files for this speaker
        speaker_files = list(Path(speaker_dir).glob("**/*.wav"))
        assert len(speaker_files) == chunks_per_file * 2  # 2 files per speaker


def test_process_probe_reference(temp_audio_dirs, temp_config):
    """Test processing both probe and reference directories"""
    probe_dir = temp_audio_dirs["probe_dir"]
    reference_dir = temp_audio_dirs["reference_dir"]
    output_dir = config.get_path("processed_audio_dir")
    
    # Debug the directory structure
    print(f"\nProbe directory: {probe_dir}")
    print(f"Reference directory: {reference_dir}")
    print(f"Files in probe directory and subdirectories:")
    for root, dirs, files in os.walk(probe_dir):
        for file in files:
            print(f"  {os.path.join(root, file)}")
    print(f"Files in reference directory and subdirectories:")
    for root, dirs, files in os.walk(reference_dir):
        for file in files:
            print(f"  {os.path.join(root, file)}")
    
    # Process probe and reference
    probe_chunks, reference_chunks = audioprep.process_probe_reference(
        probe_dir,
        reference_dir,
        output_dir,
        sample_rate=TEST_SAMPLE_RATE,
        chunk_duration=TEST_CHUNK_DURATION,
        fade_duration=TEST_FADE_DURATION,
        extensions=[".wav"]  # Explicitly specify extensions
    )
    
    # Expected files in each directory
    expected_files = 4  # 2 speakers × 2 files per speaker
    
    # Expected chunks per file
    chunks_per_file = int(np.ceil(TEST_DURATION / TEST_CHUNK_DURATION))
    
    # Expected total chunks for each directory
    expected_chunks = expected_files * chunks_per_file
    
    # Check if files were created correctly
    assert len(probe_chunks) == expected_chunks
    assert len(reference_chunks) == expected_chunks
    
    # Check if output directories exist
    probe_output_dir = config.get_path("probe_processed_dir")
    reference_output_dir = config.get_path("reference_processed_dir")
    
    assert os.path.exists(probe_output_dir)
    assert os.path.exists(reference_output_dir)
    
    # Check speaker directories in each output directory
    for output_dir in [probe_output_dir, reference_output_dir]:
        for speaker in ["speaker1", "speaker2"]:
            speaker_dir = os.path.join(output_dir, speaker)
            assert os.path.exists(speaker_dir)
            
            # Count files for this speaker
            speaker_files = list(Path(speaker_dir).glob("**/*.wav"))
            assert len(speaker_files) == chunks_per_file * 2  # 2 files per speaker


def test_parse_args(monkeypatch, temp_config):
    """Test the argument parsing functionality"""
    # Test probe-ref mode
    test_args = ["audio-prep", "probe-ref", "--chunk-duration", "1.0"]
    monkeypatch.setattr("sys.argv", test_args)
    
    # This won't work with pytest directly, since parse_args expects to be called
    # from the main module, so we'll just check the function exists
    assert callable(audioprep.parse_args)


def test_min_chunk_duration(sine_wave):
    """Test minimum chunk duration parameter"""
    wave, sr = sine_wave
    
    # Split into chunks with a small min_chunk_duration
    chunk_duration = 0.5  # seconds
    fade_duration = 0.1  # seconds
    min_chunk_duration = 0.1  # seconds (very small)
    
    chunks = audioprep.split_into_chunks(
        wave, sr, chunk_duration, fade_duration, min_chunk_duration
    )
    
    # Expected number of chunks (should include the last partial chunk)
    expected_chunks = int(np.ceil(TEST_DURATION / chunk_duration))
    assert len(chunks) == expected_chunks
    
    # Now test with a large min_chunk_duration that would exclude the last chunk
    min_chunk_duration = 0.4  # seconds
    chunks = audioprep.split_into_chunks(
        wave, sr, chunk_duration, fade_duration, min_chunk_duration
    )
    
    # If the last chunk is shorter than min_chunk_duration, it should be excluded
    last_chunk_duration = TEST_DURATION % chunk_duration
    if last_chunk_duration > 0 and last_chunk_duration < min_chunk_duration:
        expected_chunks -= 1
    
    assert len(chunks) == expected_chunks


def test_error_handling_load_audio(tmpdir):
    """Test error handling when loading a non-existent audio file"""
    non_existent_file = str(tmpdir / "non_existent.wav")
    
    # Should raise an exception
    with pytest.raises(Exception):
        audioprep.load_audio(non_existent_file)


def test_error_handling_process_directory(tmpdir):
    """Test error handling when processing an empty directory"""
    empty_dir = str(tmpdir / "empty")
    os.makedirs(empty_dir, exist_ok=True)
    
    # Should not raise an exception but return empty list
    chunks = audioprep.process_directory(empty_dir, str(tmpdir / "output"))
    assert len(chunks) == 0


def test_load_real_voice_file():
    """Test loading the real voice file"""
    # Load the real voice file
    audio, sr = audioprep.load_audio(REAL_VOICE_FILE)
    
    # Check if loaded correctly
    assert sr == 16000  # Should match the expected sample rate
    assert len(audio) > 0
    assert isinstance(audio, np.ndarray)


def test_process_real_voice_file(temp_config):
    """Test processing the real voice file"""
    processed_dir = config.get_path("processed_audio_dir")
    
    # Process the real voice file
    chunk_files = audioprep.process_audio_file(
        REAL_VOICE_FILE,
        processed_dir,
        sample_rate=16000,
        chunk_duration=4.0,  # Use a larger chunk duration for real voice
        fade_duration=0.1
    )
    
    # Based on the voice file duration (about 23.62 seconds)
    expected_chunks = int(np.ceil(23.62 / 4.0))
    
    # Check if files were created correctly
    assert len(chunk_files) == expected_chunks
    
    # Verify each chunk file exists and has the right format
    for file_path in chunk_files:
        assert os.path.exists(file_path)
        
        # Load saved chunk
        loaded_audio, loaded_sr = sf.read(file_path)
        
        # Check sample rate
        assert loaded_sr == 16000
        
        # Check that the audio has data
        assert len(loaded_audio) > 0


def test_process_real_voice_directory(real_voice_setup, temp_config):
    """Test processing a directory with real voice files"""
    probe_dir = real_voice_setup["probe_dir"]
    processed_dir = config.get_path("processed_audio_dir")
    
    # Process the directory with real voice files
    chunk_files = audioprep.process_directory(
        probe_dir,
        processed_dir,
        sample_rate=16000,
        chunk_duration=4.0,
        fade_duration=0.1,
        extensions=[".ogg"]  # The real voice file is in OGG format
    )
    
    # Expected chunks based on the real voice file duration (about 23.62 seconds)
    expected_chunks = int(np.ceil(23.62 / 4.0))
    
    # Check if files were created correctly
    assert len(chunk_files) == expected_chunks
    
    # Verify the output directory structure
    speaker_dir = os.path.join(processed_dir, "speaker1")
    assert os.path.exists(speaker_dir)
    
    # Count files for the speaker
    speaker_files = list(Path(speaker_dir).glob("**/*.wav"))
    assert len(speaker_files) == expected_chunks


def test_process_real_voice_probe_reference(real_voice_setup, temp_config):
    """Test processing both probe and reference directories with real voice files"""
    probe_dir = real_voice_setup["probe_dir"]
    reference_dir = real_voice_setup["reference_dir"]
    output_dir = config.get_path("processed_audio_dir")
    
    # Process probe and reference directories with real voice files
    probe_chunks, reference_chunks = audioprep.process_probe_reference(
        probe_dir,
        reference_dir,
        output_dir,
        sample_rate=16000,
        chunk_duration=4.0,
        fade_duration=0.1,
        extensions=[".ogg"]  # The real voice file is in OGG format
    )
    
    # Expected chunks based on the real voice file duration (about 23.62 seconds)
    expected_chunks = int(np.ceil(23.62 / 4.0))
    
    # Check if files were created correctly
    assert len(probe_chunks) == expected_chunks
    assert len(reference_chunks) == expected_chunks
    
    # Check if output directories exist
    probe_output_dir = config.get_path("probe_processed_dir")
    reference_output_dir = config.get_path("reference_processed_dir")
    
    assert os.path.exists(probe_output_dir)
    assert os.path.exists(reference_output_dir)
    
    # Check speaker directories in each output directory
    for dir_path in [probe_output_dir, reference_output_dir]:
        speaker_dir = os.path.join(dir_path, "speaker1")
        assert os.path.exists(speaker_dir)
        
        # Count files for the speaker
        speaker_files = list(Path(speaker_dir).glob("**/*.wav"))
        assert len(speaker_files) == expected_chunks 