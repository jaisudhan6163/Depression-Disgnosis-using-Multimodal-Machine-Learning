import streamlit as st
import pandas as pd
from prcsfle import process_pds
st.write('### Depression Detection using Multimodal Machine Learning')

transcript = st.file_uploader("Upload Transript File")
if transcript:
    transcript = pd.read_csv(transcript, delimiter = '\t', encoding = 'utf-8', engine = 'python')

covarep = st.file_uploader("Upload COVAREP File")
if covarep:
    covarep = pd.read_csv(covarep, header = None)

clnf_au = st.file_uploader("Upload CLNF AU File")
if clnf_au:
    clnf_au = pd.read_csv(clnf_au, delimiter = ',', engine = 'python')

clnf_feat = st.file_uploader("Upload CLNF Feature File")
if clnf_feat:
    clnf_feat = pd.read_csv(clnf_feat, delimiter = ',', engine = 'python')

clnf_feat3d = st.file_uploader("Upload CLNF Feature3D File")
if clnf_feat3d:
    clnf_feat3d = pd.read_csv(clnf_feat3d, delimiter = ',', engine = 'python')

clnf_gaze = st.file_uploader("Upload CLNF Gaze File")
if clnf_gaze:
    clnf_gaze = pd.read_csv(clnf_gaze, delimiter = ',', engine = 'python')

clnf_pose = st.file_uploader("Upload CLNF Pose File")
if clnf_pose:
    clnf_pose = pd.read_csv(clnf_pose, delimiter = ',', engine = 'python')

is_submit = st.button('Submit')

if is_submit:
    #st.write(all([transcript, covarep, clnf_au, clnf_feat, clnf_feat3d, clnf_gaze, clnf_pose]))
    out = process_pds(transcript, covarep, clnf_au, clnf_feat, clnf_feat3d, clnf_gaze, clnf_pose)
    st.write('##### There is ' + str(int(out*100)) + '% of chance the person being depressed.')