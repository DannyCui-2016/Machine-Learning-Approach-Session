from paddleocr import PaddleOCR

ocr = PaddleOCR(use_textline_orientation=True, lang='en')

def extract_text(image):
    result = ocr.ocr(image)
    text = ""

    if result and result[0]:
        for line in result[0]:
            text += line[1][0] + "\n"

    return text
