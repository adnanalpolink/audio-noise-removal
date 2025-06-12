import streamlit as st
import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf
import io
import tempfile
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt

def load_audio(uploaded_file):
    """Load audio file and return audio data and sample rate"""
    try:
        # Create a temporary file to save the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load audio using librosa with error handling
        try:
            audio_data, sample_rate = librosa.load(tmp_file_path, sr=None)
            
            # Ensure audio is mono (convert stereo to mono if needed)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=0)
            
            # Normalize audio data
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
        except Exception as e:
            st.error(f"Error with librosa, trying alternative method: {str(e)}")
            # Fallback to soundfile
            audio_data, sample_rate = sf.read(tmp_file_path)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return audio_data, sample_rate
        
    except Exception as e:
        st.error(f"Error loading audio file: {str(e)}")
        return None, None

def remove_noise(audio_data, sample_rate, stationary=True, prop_decrease=1.0):
    """Remove noise from audio using noisereduce library"""
    try:
        # Apply noise reduction
        reduced_noise = nr.reduce_noise(
            y=audio_data, 
            sr=sample_rate,
            stationary=stationary,
            prop_decrease=prop_decrease
        )
        return reduced_noise
    except Exception as e:
        st.error(f"Error removing noise: {str(e)}")
        return None

def create_audio_plot(audio_data, sample_rate, title):
    """Create a waveform plot of the audio"""
    fig, ax = plt.subplots(figsize=(12, 4))
    time = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
    ax.plot(time, audio_data)
    ax.set_title(title)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    return fig

def audio_to_bytes(audio_data, sample_rate):
    """Convert audio data to bytes for download and playback"""
    try:
        # Normalize audio to prevent clipping
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
        
        # Ensure audio is in the right format (float32)
        audio_data = audio_data.astype(np.float32)
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        st.error(f"Error converting audio: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Audio Noise Removal Tool",
        page_icon="🎵",
        layout="wide"
    )
    
    st.title("🎵 Audio Noise Removal Tool")
    st.markdown("Upload an audio file and remove background noise using advanced algorithms.")
    
    # Sidebar for parameters
    st.sidebar.header("Noise Reduction Settings")
    
    # Noise reduction parameters
    stationary_noise = st.sidebar.checkbox(
        "Stationary Noise", 
        value=True,
        help="Check if the noise is consistent throughout the recording"
    )
    
    prop_decrease = st.sidebar.slider(
        "Noise Reduction Strength",
        min_value=0.1,
        max_value=1.0,
        value=0.8,
        step=0.1,
        help="How much to reduce the noise (higher = more aggressive)"
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
        help="Supported formats: WAV, MP3, FLAC, M4A, OGG"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.info(f"📁 Uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
        
        # Load audio
        with st.spinner("Loading audio file..."):
            audio_data, sample_rate = load_audio(uploaded_file)
        
        if audio_data is not None:
            # Display original audio info
            duration = len(audio_data) / sample_rate
            st.success(f"✅ Audio loaded successfully! Duration: {duration:.2f} seconds, Sample Rate: {sample_rate} Hz")
            
            # Create two columns for before/after comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔊 Original Audio")
                
                # Play original audio
                try:
                    original_bytes = audio_to_bytes(audio_data, sample_rate)
                    if original_bytes:
                        st.audio(original_bytes, format='audio/wav')
                    else:
                        st.error("Could not generate audio for playback")
                except Exception as e:
                    st.error(f"Error playing original audio: {str(e)}")
                
                # Plot original waveform
                fig_original = create_audio_plot(audio_data, sample_rate, "Original Audio Waveform")
                st.pyplot(fig_original)
                plt.close()
            
            # Process audio when button is clicked
            if st.button("🎯 Remove Noise", type="primary"):
                with col2:
                    st.subheader("🔇 Noise Reduced Audio")
                    
                    with st.spinner("Removing noise... This may take a moment."):
                        cleaned_audio = remove_noise(
                            audio_data, 
                            sample_rate, 
                            stationary=stationary_noise,
                            prop_decrease=prop_decrease
                        )
                    
                    if cleaned_audio is not None:
                        st.success("✅ Noise removal completed!")
                        
                        # Play cleaned audio
                        try:
                            cleaned_bytes = audio_to_bytes(cleaned_audio, sample_rate)
                            if cleaned_bytes:
                                st.audio(cleaned_bytes, format='audio/wav')
                                
                                # Download button
                                st.download_button(
                                    label="📥 Download Cleaned Audio",
                                    data=cleaned_bytes,
                                    file_name=f"cleaned_{uploaded_file.name.split('.')[0]}.wav",
                                    mime="audio/wav"
                                )
                            else:
                                st.error("Could not generate cleaned audio for playback")
                        except Exception as e:
                            st.error(f"Error playing cleaned audio: {str(e)}")
                            # Still try to provide download even if playback fails
                            try:
                                cleaned_bytes = audio_to_bytes(cleaned_audio, sample_rate)
                                if cleaned_bytes:
                                    st.download_button(
                                        label="📥 Download Cleaned Audio (Playback Error)",
                                        data=cleaned_bytes,
                                        file_name=f"cleaned_{uploaded_file.name.split('.')[0]}.wav",
                                        mime="audio/wav"
                                    )
                            except:
                                st.error("Could not generate audio file")
                        
                        # Plot cleaned waveform
                        try:
                            fig_cleaned = create_audio_plot(cleaned_audio, sample_rate, "Noise Reduced Audio Waveform")
                            st.pyplot(fig_cleaned)
                            plt.close()
                        except Exception as e:
                            st.error(f"Error plotting cleaned audio: {str(e)}")
                        
                        # Audio statistics
                        st.subheader("📊 Audio Statistics")
                        stats_col1, stats_col2 = st.columns(2)
                        
                        with stats_col1:
                            st.metric("Original RMS", f"{np.sqrt(np.mean(audio_data**2)):.4f}")
                            st.metric("Original Peak", f"{np.max(np.abs(audio_data)):.4f}")
                        
                        with stats_col2:
                            st.metric("Cleaned RMS", f"{np.sqrt(np.mean(cleaned_audio**2)):.4f}")
                            st.metric("Cleaned Peak", f"{np.max(np.abs(cleaned_audio)):.4f}")
                    else:
                        st.error("Failed to process audio. Please try with different settings.")
    
    # Instructions
    st.markdown("---")
    st.subheader("📋 How to Use")
    st.markdown("""
    1. **Upload** your audio file using the file uploader above
    2. **Adjust** noise reduction settings in the sidebar:
       - **Stationary Noise**: Check if background noise is consistent
       - **Reduction Strength**: Higher values = more aggressive noise removal
    3. **Click** "Remove Noise" to process your audio
    4. **Listen** to the results and **download** the cleaned audio file
    
    **Tips for best results:**
    - Use WAV format for highest quality
    - For consistent background noise (like fan noise), enable "Stationary Noise"
    - Start with moderate reduction strength (0.6-0.8) to avoid over-processing
    - Very noisy recordings may need higher reduction strength
    """)
    
    # Technical info
    with st.expander("🔧 Technical Information"):
        st.markdown("""
        This tool uses advanced noise reduction algorithms:
        - **Spectral Gating**: Identifies and reduces noise based on spectral analysis
        - **Stationary vs Non-stationary**: Different algorithms for different noise types
        - **Adaptive Processing**: Preserves speech quality while removing noise
        
        **Supported Audio Formats:**
        - WAV (recommended for best quality)
        - MP3, FLAC, M4A, OGG
        
        **Limitations:**
        - Processing time increases with file size
        - Very heavily distorted audio may not improve significantly
        - Some speech artifacts may occur with very aggressive settings
        """)

if __name__ == "__main__":
    main()
