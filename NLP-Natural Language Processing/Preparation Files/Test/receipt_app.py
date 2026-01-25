import streamlit as st
import pytesseract
from PIL import Image
import pandas as pd
import re

# ---------------------------
# 商品分类规则（可扩展）
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
    text = pytesseract.image_to_string(image)
    return text


# ---------------------------
# 简单解析商品行
# ---------------------------
def parse_items(text):
    lines = text.split("\n")
    items = []

    for line in lines:
        line = line.strip()
        if len(line) < 3:
            continue

        # 尝试提取价格
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
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Receipt NLP Analyzer", layout="wide")
st.title("🧾 超市收据识别与分类系统")

uploaded_file = st.file_uploader("上传收据图片", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="上传的收据", use_container_width=True)

    with col2:
        st.subheader("OCR识别文本")
        raw_text = extract_text_from_image(image)
        st.text_area("识别结果", raw_text, height=300)

    st.divider()

    st.subheader("📊 分类结果")
    df = parse_items(raw_text)

    if len(df) > 0:
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("未识别到有效商品数据")
