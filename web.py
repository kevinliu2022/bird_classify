import streamlit as st
from PIL import Image
from infer import infer
from infer import onnx_infer
import base64
st.header("鸟类分类识别系统")
model_type=st.sidebar.selectbox("请选择你需要使用的模型",["resnet专用模型","resnet_onnx模型","resnet量化int8模型"])

def main_bg(main_bg):
    main_bg_ext = "png"
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )
# 调用
main_bg('./images/background.jpg')
def sidebar_bg(side_bg):       #背景
    side_bg_ext = 'png'
    st.markdown(
        f"""
      <style>
      [data-testid="stSidebar"]  {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,
        unsafe_allow_html=True,
    )
# 调用
sidebar_bg('./images/sidebar_bg.jpg')
# 播放音乐
audio_file = open('./data/music.mp3', 'rb')
st.sidebar.audio(audio_file, format='audio/mp3')
import datetime
date1 = st.sidebar.date_input("今天的日期", datetime.date(2024, 11, 4))
col1, col2, col3 = st.sidebar.columns(3)
col1.metric("Temperature", "70 °F", "1.2 °F")
col2.metric("Wind", "9 mph", "-8%")
col3.metric("Humidity", "86%", "4%")

label_texts="data/bird_label.txt"
up_image=st.file_uploader("请上传一张需要识别的鸟类图片....")
if up_image is not None:
    image = Image.open(up_image).convert("RGB")
    st.image(image)
    if model_type == "resnet专用模型":
        model_file = "model/best.pth"
        labels=infer(image,model_file,label_texts)
    if model_type == "resnet_onnx模型":
        model_file = "model/best.onnx"
        labels=onnx_infer(image,model_file,label_texts)
    # if model_type == "resnet量化int8模型":
    #     model_file = "model/quan_int8.pth"
    #     labels=quan_int8_infer(image,model_file,label_texts)
    for label, prob in labels:
        st.write("预测类别为", label, "概率为", prob)
    st.balloons()