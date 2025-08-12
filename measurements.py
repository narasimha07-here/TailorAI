import streamlit as st
from PIL import Image
import pandas as pd
from streamlit_option_menu import option_menu
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Models.pipeline import Measurements


st.set_page_config(page_title="TailorAI")

st.title("TailorAI-Precision in Every Seam")
st.subheader("Measure Smarter, Wear Better")
st.markdown(" ")

col_empty1,col1, col2, col_empty2 = st.columns([1, 3, 3, 1])
with col1:
    st.image(r"streamlits\images\frontal.png", caption="Frontal image",width=200)
with col2:
    st.image(r"streamlits\images\lateral.png", caption="Side image",width=200)

frontal=st.file_uploader("upload a front image as shown above",type=["jpg","png"],accept_multiple_files=False)

lateral=st.file_uploader("upload a side image as shown above",type=["jpg","png"],accept_multiple_files=False)

col_empty1, col3,col4,col_empty2 = st.columns([1.5, 3,3,1.5])
with col3:
    if frontal is not None:
        st.image(frontal,width=180)
with col4:
    if lateral is not None:
        st.image(lateral,width=180)

if frontal and lateral:

    button = st.button("📏 Get My Measurements")
    predictor = Measurements()

    if button:
        with st.spinner("AI Tailoring to You...."):
            frontal_img = Image.open(frontal)
            lateral_img = Image.open(lateral)
                

            st.session_state.results = predictor.predict(frontal_img, lateral_img)

            st.success("AI Tailored to You.....")
        
        chest=st.session_state.results["chest"]
        #st.write(chest)

        if "results" in st.session_state:
            df = pd.DataFrame(list(st.session_state.results.items()),columns=["Measurements","Sizes"],index=None)
            st.table(df)