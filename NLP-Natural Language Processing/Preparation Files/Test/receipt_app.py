import streamlit as st
from PIL import Image
import pandas as pd
import re
import numpy as np
import cv2
from paddleocr import PaddleOCR

# ---------------------------
# 初始化 PaddleOCR
# ---------------------------
ocr = PaddleOCR(use_angle_cls=True, lang="en")  # 初始化时开启自动旋转

# ---------------------------
# 商品分类规则
# ---------------------------
CATEGORY_RULES = {
    "Food": ["apple", "banana", "bread", "milk", "rice", "noodle", "egg", "chicken"],
    "Drink": ["water", "cola", "juice", "coffee", "tea"],
    "Daily": ["tissue", "soap", "shampoo", "toothpaste", "detergent"],
}

def classify_item(text):
    text = text.lower()
    for category, keywords in CATEGORY_RULES.items():
        for k in keywords:
            if k in text:
                return category
    return "Other"

# ---------------------------
# 图像预处理
# ---------------------------
def preprocess_image(pil_img):
    # 转为 numpy
    img = np.array(pil_img)
    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    # 转回三通道 BGR
    processed_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return processed_img

# ---------------------------
# OCR识别
# ---------------------------
def extract_text_from_image(image_np):
    result = ocr.ocr(image_np)  # 最新版本不需要 cls 参数
    lines = []
    for block in result:
        for line in block:
            lines.append(line[1][0])
    return "\n".join(lines)

# ---------------------------
# 解析商品
# ---------------------------
def parse_items(text):
    lines = text.split("\n")
    items = []

    for line in lines:
        line = line.strip()
        if len(line) < 3:
            continue

        # 匹配价格
        price_match = re.search(r"(\d+\.\d{2})", line)
        price = price_match.group(1) if price_match else ""

        # 去掉价格得到商品名
        name = re.sub(r"\d+\.\d{2}", "", line).strip()

        if name:
            category = classify_item(name)
            items.append({
                "Item": name,
                "Price": price,
                "Category": category
            })

    return pd.DataFrame(items)

# ---------------------------
# Streamlit 页面布局
# ---------------------------
st.set_page_config(page_title="Receipt NLP System", layout="wide")
st.title("🧾 超市收据识别系统（PaddleOCR 最新版）")

left_col, right_col = st.columns([1, 2])

# 左侧上传图片
with left_col:
    st.subheader("📷 图片预览")
    image_placeholder = st.empty()

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "📤 上传收据图片",
        type=["jpg", "png", "jpeg"]
    )

# 右侧显示结果
with right_col:
    st.subheader("📄 识别结果")
    text_placeholder = st.empty()
    table_placeholder = st.empty()

# ---------------------------
# 处理逻辑
# ---------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_placeholder.image(image, use_container_width=True)

    # 预处理
    processed_image = preprocess_image(image)

    # OCR识别
    raw_text = extract_text_from_image(processed_image)

    # 显示 OCR 文本
    text_placeholder.text_area(
        "OCR识别文本",
        raw_text,
        height=250
    )

    # 解析商品
    df = parse_items(raw_text)

    st.markdown("### 📊 分类表格")
    table_placeholder.dataframe(df, use_container_width=True)

else:
    image_placeholder.info("请上传一张收据图片")
    text_placeholder.info("识别内容将在这里显示")
