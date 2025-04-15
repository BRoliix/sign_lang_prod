import os
import streamlit as st
import time
import gc
import torch
import pandas as pd
from PIL import Image
from sign_translator import SignLanguageRAG
from video_preprocessor import SignLanguagePreprocessor

# Clear memory
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

# Page configuration
st.set_page_config(
    page_title="Sign Language Translation System",
    page_icon="👋",
    layout="wide"
)

# CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .subheader {
        font-size: 1.5rem;
        color: #424242;
    }
    .info-text {
        font-size: 1rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #E8F5E9;
        border-radius: 0.5rem;
    }
    .context-box {
        padding: 1rem;
        background-color: #F5F5F5;
        border-radius: 0.5rem;
    }
    .warning-box {
        padding: 1rem;
        background-color: #FFF8E1;
        border-radius: 0.5rem;
    }
    .feature-box {
        padding: 1rem;
        background-color: #E3F2FD;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .tab-content {
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Memory-optimized configuration defaults
DEFAULT_FRAMES = 8
DEFAULT_FPS = 8
DEFAULT_MOTION = 80
DEFAULT_NOISE = 0.1

# Initialize the preprocessor and translator once
@st.cache_resource
def load_translator():
    try:
        return SignLanguageRAG(
            data_dir="./data",
            chroma_dir="./chroma_db"
        )
    except Exception as e:
        st.error(f"Failed to initialize translator: {str(e)}")
        st.exception(e)
        return None

@st.cache_resource
def load_preprocessor():
    try:
        return SignLanguagePreprocessor(
            data_dir="./data"
        )
    except Exception as e:
        st.error(f"Failed to initialize preprocessor: {str(e)}")
        st.exception(e)
        return None

# App title
st.markdown("<h1 class='main-header'>Sign Language Translation System</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>Translate text to sign language videos using RAG and diffusion models with landmark preprocessing</p>", unsafe_allow_html=True)

# Main tabs
tab1, tab2 = st.tabs(["Translation", "Video Preprocessing"])

# ---- TRANSLATION TAB ----
with tab1:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    
    # Memory usage warning for Mac users
    st.markdown("""
    <div class="warning-box">
        <strong>Memory Optimization Notice:</strong> This application uses StableVideoDiffusion which requires significant memory. 
        The default settings have been optimized for Mac computers. Increasing frame count or resolution may cause out-of-memory errors.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with configuration options
    st.sidebar.header("Translation Configuration")
    data_dir = st.sidebar.text_input("Data Directory", value="./data")
    chroma_dir = st.sidebar.text_input("Chroma DB Directory", value="./chroma_db")
    
    # Memory-critical parameters
    st.sidebar.subheader("Video Generation Settings")
    fps = st.sidebar.slider("Video FPS", min_value=5, max_value=30, value=DEFAULT_FPS)
    num_frames = st.sidebar.slider("Number of Frames", min_value=6, max_value=12, value=DEFAULT_FRAMES, 
                                help="Higher values require more memory")
    resolution = st.sidebar.radio("Video Resolution", ["Low (192x192)", "Medium (256x256)"], index=0,
                               help="Lower resolution uses less memory")
    
    context_k = st.sidebar.slider("Number of contexts to retrieve", min_value=1, max_value=5, value=3)
    advanced_options = st.sidebar.expander("Advanced Options")
    with advanced_options:
        motion_intensity = st.slider("Motion Intensity", min_value=1, max_value=255, value=DEFAULT_MOTION)
        noise_strength = st.slider("Noise Strength", min_value=0.0, max_value=0.5, value=DEFAULT_NOISE, step=0.01)
        use_cpu = st.checkbox("Force CPU mode", value=False, 
                            help="Very slow but may help with memory errors")
        use_landmarks = st.checkbox("Use landmark-based pose generation", value=True,
                                  help="Generates better starting poses when landmarks are available")
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h2 class='subheader'>Input</h2>", unsafe_allow_html=True)
        
        # Text input area
        text_input = st.text_area(
            "Enter text to translate to sign language",
            value="Hello, how are you?",
            height=100
        )
        
        # File output name
        output_filename = st.text_input(
            "Output video filename",
            value="output_video.mp4"
        )
    
        # Action buttons
        button_col1, button_col2 = st.columns(2)
        with button_col1:
            translate_button = st.button("Translate", use_container_width=True)
        with button_col2:
            clear_button = st.button("Clear Results", use_container_width=True)
    
    # Results area
    with col2:
        st.markdown("<h2 class='subheader'>Results</h2>", unsafe_allow_html=True)
        result_placeholder = st.empty()
    
    # Process forced CPU mode
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    # Translation process
    if translate_button:
        with st.spinner("Initializing translation system..."):
            try:
                # Force garbage collection
                gc.collect()
                
                # If using MPS, clear cache
                if torch.backends.mps.is_available() and not use_cpu:
                    torch.mps.empty_cache()
                
                # Load translator
                translator = load_translator()
                
                if translator is None:
                    st.error("Failed to initialize the translation system.")
                else:
                    # Status message
                    start_time = time.time()
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process steps
                    status_text.text("Step 1/5: Retrieving context...")
                    progress_bar.progress(10)
                    
                    # Update progress
                    status_text.text("Step 2/5: Generating sign language prompt...")
                    progress_bar.progress(30)
                    
                    # Update memory settings based on resolution choice
                    height = width = 192 if resolution == "Low (192x192)" else 256
                    
                    # Execute translation
                    status_text.text("Step 3/5: Generating video (this may take a while)...")
                    progress_bar.progress(50)
                    
                    result = translator.translate(
                        text=text_input,
                        output_file=output_filename,
                        fps=fps,
                        num_frames=num_frames,
                        motion_bucket_id=motion_intensity,
                        noise_aug_strength=noise_strength
                    )
                    
                    # Update progress
                    status_text.text("Step 5/5: Finalizing results...")
                    progress_bar.progress(100)
                    
                    # Clear progress indicators
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Show success message
                    elapsed_time = time.time() - start_time
                    
                    with result_placeholder.container():
                        st.markdown(f"""
                        <div class='success-box'>
                            <h3>Translation Complete!</h3>
                            <p>Processed in {elapsed_time:.2f} seconds</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display video
                        st.subheader("Sign Language Video")
                        video_path = result["video_path"]
                        if os.path.exists(video_path):
                            st.video(video_path)
                        else:
                            st.error(f"Video file not found at: {video_path}")
                        
                        # Display enhanced prompt
                        st.subheader("Generated Prompt")
                        st.text_area("", result["prompt"], height=100, disabled=True)
                        
                        # Show retrieved contexts
                        st.subheader("Retrieved Contexts")
                        for i, context in enumerate(result["contexts"]):
                            with st.expander(f"Context {i+1}", expanded=i==0):
                                st.markdown(f"""
                                <div class='context-box'>
                                    <p><strong>Content:</strong> {context['content']}</p>
                                    <p><strong>Source:</strong> {context['metadata'].get('source', 'Unknown')}</p>
                                    <p><strong>Type:</strong> {context['metadata'].get('type', 'Unknown')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Show landmark visualizations if any
                        if result.get("visualization_paths") and len(result["visualization_paths"]) > 0:
                            st.subheader("Landmark Visualizations")
                            vis_files = result["visualization_paths"]
                            img_cols = st.columns(min(3, len(vis_files)))
                            for i, vis_path in enumerate(vis_files):
                                if os.path.exists(vis_path):
                                    col_idx = i % len(img_cols)
                                    img_cols[col_idx].image(vis_path, caption=f"Landmarks {i+1}")
                    
                    # Force cleanup
                    if torch.backends.mps.is_available() and not use_cpu:
                        torch.mps.empty_cache()
                    gc.collect()
                    
            except RuntimeError as e:
                if "buffer size" in str(e):
                    st.error(f"""
                    Memory error: Not enough GPU memory to generate the video with current settings.
                    
                    Try these solutions:
                    1. Reduce the number of frames (currently {num_frames})
                    2. Use a lower resolution setting
                    3. Enable "Force CPU mode" in Advanced Settings (will be much slower)
                    
                    Technical Error: {str(e)}
                    """)
                else:
                    st.error(f"Error during translation: {str(e)}")
                    st.exception(e)
            except Exception as e:
                st.error(f"Error during translation: {str(e)}")
                st.exception(e)
    
    # Clear results if requested
    if clear_button:
        result_placeholder.empty()
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---- PREPROCESSING TAB ----
with tab2:
    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='subheader'>Video Preprocessing</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-box">
        <p>This tab allows you to preprocess sign language videos by extracting landmarks, generating visualizations, 
        and preparing feature vectors for the RAG system. Preprocessing videos enables more accurate sign language translation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for preprocessing
    st.sidebar.header("Preprocessing Configuration")
    
    # Video source selection
    video_source = st.sidebar.radio(
        "Video Source",
        ["Upload New Video", "Process Existing Video", "Process by Sign Name"]
    )
    
    # Main preprocessing content
    if video_source == "Upload New Video":
        st.subheader("Upload a Sign Language Video")
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
        
        if uploaded_file is not None:
            # Save uploaded file
            video_path = os.path.join("./data/videos", uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"Video uploaded: {uploaded_file.name}")
            
            # Display the uploaded video
            st.video(video_path)
            
            # Process button
            if st.button("Process Video"):
                with st.spinner("Processing video..."):
                    preprocessor = load_preprocessor()
                    if preprocessor:
                        result = preprocessor.process_video(video_path)
                        
                        if result:
                            st.success("Video processed successfully!")
                            
                            # Show the visualization
                            st.subheader("Extracted Landmarks")
                            st.image(result["visualization_path"])
                            
                            # Show details
                            st.json(result)
                        else:
                            st.error("Failed to process video. No landmarks detected.")
    
    elif video_source == "Process Existing Video":
        st.subheader("Process an Existing Video")
        
        # Find existing videos
        videos_dir = os.path.join("./data/videos")
        if not os.path.exists(videos_dir):
            st.warning(f"Videos directory not found: {videos_dir}")
        else:
            video_files = []
            for ext in [".mp4", ".avi", ".mov"]:
                video_files.extend([f for f in os.listdir(videos_dir) if f.endswith(ext)])
            
            if not video_files:
                st.warning("No video files found in ./data/videos directory")
            else:
                selected_video = st.selectbox("Select a video to process", video_files)
                video_path = os.path.join(videos_dir, selected_video)
                
                # Display the selected video
                st.video(video_path)
                
                # Process button
                if st.button("Process Selected Video"):
                    with st.spinner("Processing video..."):
                        preprocessor = load_preprocessor()
                        if preprocessor:
                            result = preprocessor.process_video(video_path)
                            
                            if result:
                                st.success("Video processed successfully!")
                                
                                # Show the visualization
                                st.subheader("Extracted Landmarks")
                                st.image(result["visualization_path"])
                                
                                # Show details
                                st.json(result)
                            else:
                                st.error("Failed to process video. No landmarks detected.")
    
    elif video_source == "Process by Sign Name":
        st.subheader("Process Videos by Sign Name")
        
        # Load preprocessor to get sign data
        preprocessor = load_preprocessor()
        if preprocessor:
            # Get available signs
            sign_options = list(preprocessor.sign_data.keys())
            
            if not sign_options:
                st.warning("No sign data found. Make sure ASL descriptions and WLASL data are loaded.")
            else:
                selected_sign = st.selectbox("Select a sign to process", sorted(sign_options))
                
                if selected_sign:
                    sign_info = preprocessor.sign_data[selected_sign]
                    
                    # Show sign information
                    st.markdown(f"""
                    <div class="info-text">
                        <p><strong>Sign:</strong> {selected_sign}</p>
                        <p><strong>Description:</strong> {sign_info.get('description', 'No description available')}</p>
                        <p><strong>Available Videos:</strong> {len(sign_info.get('videos', []))}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Process button
                    if st.button("Process All Videos for this Sign"):
                        with st.spinner(f"Processing videos for sign '{selected_sign}'..."):
                            results = preprocessor.process_sign(selected_sign)
                            
                            if results:
                                st.success(f"Processed {len(results)} videos for sign '{selected_sign}'")
                                
                                # Show visualizations
                                st.subheader("Extracted Landmarks")
                                
                                # Create a grid of images
                                cols = st.columns(3)
                                for i, result in enumerate(results):
                                    vis_path = result["visualization_path"]
                                    if os.path.exists(vis_path):
                                        cols[i % 3].image(vis_path, caption=f"Video {i+1}")
                            else:
                                st.error(f"No videos processed for sign '{selected_sign}'")
    
    # Add batch processing option
    st.markdown("<h2 class='subheader'>Batch Processing</h2>", unsafe_allow_html=True)
    
    if st.button("Process All Videos"):
        with st.spinner("Processing all videos... This may take a while."):
            preprocessor = load_preprocessor()
            if preprocessor:
                # Import glob here to ensure it's available
                import glob
                results = preprocessor.process_all_videos()
                
                if results:
                    st.success(f"Processed {len(results)} videos successfully!")
                    
                    # Show summary
                    df = pd.DataFrame([{
                        "video_id": r["video_id"],
                        "num_frames": r["num_frames"],
                        "features_shape": str(r["features_shape"])
                    } for r in results])
                    
                    st.dataframe(df)
                else:
                    st.error("No videos processed.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Data directory status
st.markdown("<h3 class='subheader'>System Status</h3>", unsafe_allow_html=True)
status_cols = st.columns(3)

with status_cols[0]:
    if os.path.exists(data_dir):
        st.success(f"Data directory found: {data_dir}")
        json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        st.info(f"Found {len(json_files)} JSON files")
    else:
        st.error(f"Data directory not found: {data_dir}")

with status_cols[1]:
    if os.path.exists(os.path.join(data_dir, "processed")):
        processed_files = len([f for f in os.listdir(os.path.join(data_dir, "processed")) if f.endswith('_landmarks.json')])
        st.success(f"Processed data directory found")
        st.info(f"Found {processed_files} processed landmark files")
    else:
        st.warning(f"No processed landmark data found")

with status_cols[2]:
    if os.path.exists(chroma_dir):
        st.success(f"Chroma DB directory found: {chroma_dir}")
    else:
        st.warning(f"Chroma DB directory not found: {chroma_dir}")

if __name__ == "__main__":
    print("Starting Streamlit app for Sign Language Translation with preprocessing...")
