from fastapi import FastAPI, WebSocket
import base64
import cv2
import numpy as np

from ocr_service import extract_text
from question_detector import detect_question_change
from teaching_agent import generate_teaching

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = "child_1"

    while True:
        data = await websocket.receive_text()

        # 去掉data:image/jpeg;base64,
        header, encoded = data.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        text = extract_text(frame)

        changed, question = detect_question_change(session_id, text)

        if changed and question.strip() != "":
            response = generate_teaching(question)
            await websocket.send_text(response)
