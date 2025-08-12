import streamlit as st
from PIL import Image
from Models.pipeline import Measurements

st.title("TailorAI-Smart Size")
st.subheader("Measure Smarter, Wear Better")
st.markdown(" ")

col_empty1, col1, col2, col_empty2 = st.columns([1, 3, 3, 1])
with col1:
    st.image("streamlits/images/frontal.png", caption="Frontal image", width=200)
with col2:
    st.image("streamlits/images/lateral.png", caption="Side image", width=200)

frontal = st.file_uploader("Upload a front image as shown above", type=["jpg", "png"])
lateral = st.file_uploader("Upload a side image as shown above", type=["jpg", "png"])

col_empty1, col3, col4, col_empty2 = st.columns([1.5, 3, 3, 1.5])
with col3:
    if frontal is not None:
        st.image(frontal, width=180)
with col4:
    if lateral is not None:
        st.image(lateral, width=180)

if frontal and lateral:
    if st.button("📏 Get My Measurements"):
        with st.spinner("AI Tailoring to You...."):

            predictor = Measurements()
            frontal_img = Image.open(frontal)
            lateral_img = Image.open(lateral)
            st.session_state.results = predictor.predict(frontal_img, lateral_img)
            st.success("AI Tailored to You.....")
        
    if "results" in st.session_state:
        results = st.session_state.results
        chest = results.get("chest")
        waist = results.get("waist")

        if st.checkbox("Shirt size"):
            if chest is not None:
                if 76 <= chest < 81:
                    st.write("Your shirt size is XXXS")
                elif 81 <= chest < 86:
                    st.write("Your shirt size is XXS")
                elif 86 <= chest < 91:
                    st.write("Your shirt size is XS")
                elif 91 <= chest < 96:
                    st.write("Your shirt size is S")
                elif 96 <= chest < 101:
                    st.write("Your shirt size is M")
                elif 101 <= chest < 106:
                    st.write("Your shirt size is L")
                elif 106 <= chest < 111:
                    st.write("Your shirt size is XL")
                elif 111 <= chest < 116:
                    st.write("Your shirt size is XXL")
                elif 116 <= chest < 121:
                    st.write("Your shirt size is XXXL")
                else:
                    st.write("Chest measurement out of range for size chart")

        if st.checkbox("Formal Pant size"):

            if waist is not None:
                if 80 <= waist < 82:
                    st.write("Your pant size is 30")
                elif 82 <= waist < 88:
                    st.write("Your pant size is 32")
                elif 88 <= waist < 94:
                    st.write("Your pant size is 34")
                elif 94 <= waist < 100:
                    st.write("Your pant size is 36")
                elif 100 <= waist < 106:
                    st.write("Your pant size is 38")
                elif 106 <= waist < 111:
                    st.write("Your pant size is 40")
                elif 111 <= waist < 117:
                    st.write("Your pant size is 42")
                elif 117 <= waist < 122:
                    st.write("Your pant size is 44")
                elif 122 <= waist < 127:
                    st.write("Your pant size is XXXL")
                else:
                    st.write("waist measurement out of range for size chart")


    if st.checkbox("Reference shirt size chart"):
        st.image("streamlits/images/image.png")
    if st.checkbox("Reference Pant size chart"):
        st.image("streamlits/images/pant.png")

    if st.button("🧹 Clear Results"):
        st.session_state.results = None
        st.session_state.show_shirt_size = False
        st.session_state.show_pant_size = False
        st.session_state.show_shirt_chart = False
        st.session_state.show_pant_chart = False
