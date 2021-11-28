import cv2.cv2
import streamlit as st


class VideoHelper:
    @staticmethod
    def save_video(images_after_masking):
        for index, image in enumerate(images_after_masking):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite("images\\img_" + str(index) + ".jpg", image_rgb)

        size = (images_after_masking[0].shape[1], images_after_masking[0].shape[0])

        out = cv2.VideoWriter("project.mp4", cv2.VideoWriter_fourcc(*"avc1"), 30, size)
        for index, image in enumerate(images_after_masking):
            img = cv2.imread("images\\img_" + str(index) + ".jpg")
            out.write(img)
        out.release()

    @staticmethod
    def display_video():
        video_file = open("project.mp4", "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)
