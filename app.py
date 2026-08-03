import streamlit as st
import pandas as pd

from prcsfle import process_pds

st.write('### Depression Screening Support — Multimodal Interview Analysis')
st.info(
    "Research prototype, not a medical device. This tool does not diagnose "
    "depression; it estimates risk from a fixed LSTM model trained on a small "
    "(n<200) research corpus and should not be used for clinical decisions. "
    "If you or someone you know is struggling, contact a mental health "
    "professional or a crisis line in your region."
)

transcript = st.file_uploader("Upload Transcript File")
covarep = st.file_uploader("Upload COVAREP File")
clnf_au = st.file_uploader("Upload CLNF AU File")
clnf_feat = st.file_uploader("Upload CLNF Feature File")
clnf_feat3d = st.file_uploader("Upload CLNF Feature3D File")
clnf_gaze = st.file_uploader("Upload CLNF Gaze File")
clnf_pose = st.file_uploader("Upload CLNF Pose File")

is_submit = st.button('Submit')

if is_submit:
    uploads = [transcript, covarep, clnf_au, clnf_feat, clnf_feat3d, clnf_gaze, clnf_pose]
    if not all(uploads):
        st.error("Please upload all seven files before submitting.")
    else:
        try:
            transcript_df = pd.read_csv(transcript, delimiter='\t', encoding='utf-8', engine='python')
            covarep_df = pd.read_csv(covarep, header=None)
            with st.spinner("Extracting features and running the model..."):
                result = process_pds(transcript_df, covarep_df, clnf_au, clnf_feat, clnf_feat3d, clnf_gaze, clnf_pose)
        except Exception as e:
            st.error(f"Could not process the uploaded files: {e}")
        else:
            st.write(f"##### Estimated risk score: {result['probability']*100:.0f}%")
            st.caption(
                "This is a raw model score, not a calibrated probability and not a diagnosis. "
                f"Audio coverage: {result['audio_coverage']*100:.0f}% of the fixed window, "
                f"video coverage: {result['video_coverage']*100:.0f}%, "
                f"{result['n_turns']} participant turns detected."
            )
