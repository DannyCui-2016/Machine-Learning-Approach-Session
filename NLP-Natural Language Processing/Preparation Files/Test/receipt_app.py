import streamlit as st
import pytesseract
from PIL import Image
import pandas as pd
import re

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
# OCR识别
# ---------------------------
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)


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

        price_match = re.search(r"(\d+\.\d{2})", line)
        price = price_match.group(1) if price_match else ""
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
# 页面布局
# ---------------------------
st.set_page_config(page_title="Receipt NLP System", layout="wide")
st.title("🧾 超市收据识别系统")

# 左右两列
left_col, right_col = st.columns([1, 2])

# 左侧区域
with left_col:
    st.subheader("📷 图片预览")
    image_placeholder = st.empty()

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "📤 上传收据图片",
        type=["jpg", "png", "jpeg"]
    )

# 右侧区域
with right_col:
    st.subheader("📄 识别结果")
    text_placeholder = st.empty()
    table_placeholder = st.empty()


# ---------------------------
# 处理逻辑
# ---------------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    image_placeholder.image(image, use_container_width=True)

    raw_text = extract_text_from_image(image)

    text_placeholder.text_area(
        "OCR识别文本",
        raw_text,
        height=250
    )

    df = parse_items(raw_text)

    st.markdown("### 📊 分类表格")
    table_placeholder.dataframe(df, use_container_width=True)

else:
    image_placeholder.info("请上传一张收据图片")
    text_placeholder.info("识别内容将在这里显示")
