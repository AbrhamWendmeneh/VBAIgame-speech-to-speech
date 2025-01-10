import asyncio
import pyaudio
import numpy as np
from pydub import AudioSegment
from io import BytesIO
from pydub.playback import play

CHANNELS = 1
SAMPLE_RATE = 16000

class AudioPlayerAsync:
    """
    A delightful audio player that adjusts the pitch for a feminine speech style.
    """

    def __init__(self, chunk_size=1024, pitch_factor=1.2):
        """
        Initializes the audio player with pitch adjustment.

        Args:
            chunk_size (int): The size of each audio chunk for smooth processing.
            pitch_factor (float): The factor by which to raise the pitch (greater than 1 increases pitch).
        """
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frame_count = 0
        self.chunk_size = chunk_size
        self.max_frames = 300
        self.pitch_factor = pitch_factor  

    def reset_frame_count(self):
        """Resets the frame count to start fresh."""
        self.frame_count = 0

    def _adjust_pitch(self, audio_segment):
        """
        Adjusts the pitch of the audio segment to make it sound more feminine.

        Args:
            audio_segment (AudioSegment): The audio to modify.

        Returns:
            AudioSegment: The pitch-modified audio.
        """
        # Adjust pitch by changing playback speed
        return audio_segment._spawn(audio_segment.raw_data, overrides={
            "frame_rate": int(audio_segment.frame_rate * self.pitch_factor)
        }).set_frame_rate(SAMPLE_RATE)

    def add_data(self, data: bytes) -> None:
        """
        Adds new audio data to be played with a feminine touch.

        Args:
            data (bytes): The raw audio data to play.
        """
        if not data:
            return
        
        # Convert raw bytes into an AudioSegment
        audio_segment = AudioSegment.from_file(BytesIO(data), format="raw", frame_rate=SAMPLE_RATE, channels=CHANNELS, sample_width=2)
        
        # Adjust the pitch to sound more feminine
        pitched_audio = self._adjust_pitch(audio_segment)
        
        # Convert the adjusted audio back to a numpy array
        numpy_array = np.array(pitched_audio.get_array_of_samples(), dtype=np.int16)
        self._play_data(numpy_array)

    def _play_data(self, data):
        """
        Streams the audio data to the speakers, chunk by chunk.

        Args:
            data (np.ndarray): The prepared audio samples.
        """
        if self.stream is None:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                output=True
            )

        chunk_size = self.chunk_size
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]

            # Stop playback if we exceed the frame limit
            if self.frame_count > self.max_frames:
                return

            self.stream.write(chunk.tobytes())
            self.frame_count += 1

    def close(self):
        """Closes the audio player gracefully."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
