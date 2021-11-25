import streamlit as st


class Information:
    @staticmethod
    def print_main_page():
        col1, col2, col3 = st.columns(3)
        with col2:
            Information.print_information()
            Information.print_help()
            # st.image("author.jpg")
            st.image("logo.jpg")

    @staticmethod
    def print_information():
        with st.expander(" 💬 Information "):
            st.write("""


            -  Face Detection Methods 


            Multi-Task Cascaded Convolutional Neural Networks - MTCNN

            MTCNN is a framework developed as a solution for both face detection and face alignment. 
            The process consists of three stages of convolutional networks that are able to recognize 
            faces and landmark location such as eyes, nose, and mouth.


            ---

            -  Face Recognition Methods 
            

            Histogram of Oriented Gradients - HOG 
            
            HOG is a simple and powerful feature descriptor. It is not only used for face detection 
            but also it is widely used for object detection like cars, pets, and fruits. HOG is robust 
            for object detection because object shape is characterized using the local intensity gradient
            distribution and edge direction.

            Convolutional Neural Net - CNN

            CNN is a type of artificial neural network used in image recognition and processing that is 
            specifically designed to process pixel data. CNNs are powerful image processing, artificial 
            intelligence (AI) that use deep learning to perform both generative and descriptive tasks, 
            often using machine vison that includes image and video recognition, along with recommender 
            systems and natural language processing.


            ---

            -   Landmark Detection Algorithms 
            

            MediaPipe - Face Mesh

            MTCNN - Facial Landmark Detection

            Dlib - Facial Landmark 
            
            
            """)

    @staticmethod
    def print_help():
        with st.expander(" 🔍 Help "):
            st.write("""        


            - From sidebar select option: Load Image, Load Video or Live Camera
            - For Load Image/ Load Video upload file for analysis, for Live Camera click "Run"
            - Click "Face detection" to expand the available options
            - To detect a face in the image, select "Check face detection"
            - Select method for mark facial features
            - Set biometric masking mode
            - Click "Face recognition" to upload file with known person, which will be used 
              to learn face recognition model
            - Select HOG or CNN method for recognition
            - Enter the name of the person to be recognized
            - For Load Video mode select all the settings before clicking "play"
            - Test the effect of masking biometric features on face recognition and detection
            
            
            
            """)

    @staticmethod
    def print_page_title():
        st.set_page_config(layout="wide")
        st.markdown("<h1 style='text-align: center'> Face biometric masking software </h1>",
                    unsafe_allow_html=True)
