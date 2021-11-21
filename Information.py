import streamlit as st


class Information:
    @staticmethod
    def print_main_page():
        notes = f"""

        🔍 ** Information **
        - Face biometric masking software

        👋 ** Help **
        - On the left side of the screen there is a section with options to choose from
        - Select the object type for analysis - Image, Video or Live Camera view
        - Next for Image or Video options load the selected file from the folder, for Live Camera option click "Run" button
        - Choose how you want to display facial feature points
        - Press the "Face detection" button to detect a face in the image
        - Using the "Save plot with face detection" button you can save the plot with the obtained face detection on the image
        - In order to use the face recognition functionality on an image already loaded on the page, open another file from the folder, which represents a photo of a person on the basis of which you want to use face recognition
        - Select HOG or CNN method
        - Enter the name of the person to be recognized
        - Press the "Face recognition" button
        - To refresh the page press the "Reset" button
        """
        st.write(notes)

    @staticmethod
    def print_page_title():
        st.set_page_config(layout="wide")
        st.markdown("<h1 style='text-align: center'> Face biometric masking software </h1>",
                    unsafe_allow_html=True)

