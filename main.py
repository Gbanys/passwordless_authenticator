from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
import cv2
import time
from websockets.exceptions import ConnectionClosed
from ultralytics import YOLO
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import math

app = FastAPI(debug=True)
model = YOLO("/home/gbanys/repositories/passwordless_authenticator/runs/detect/train8/weights/best.pt")

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.parent.absolute() / "passwordless_authenticator" / "app" / "static"),
    name="static",
)

templates = Jinja2Templates(directory="app/templates")


@app.websocket("/ws")
async def get_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        cap = cv2.VideoCapture(0)
        close_window = False
        time_elapsed = 0
        start_time = 0

        classNames = ["BATMAN", "GIEDRIUS_EYE", "TWO_FINGERS"]

        microseconds = []
        confidences = []
        while True:
            success, img = cap.read()
            if not success:
                break
            else:
                results = model(img, stream=True)

                # coordinates
                for r in results:
                    start_time = datetime.now()
                    boxes = r.boxes

                    if not boxes:
                        confidences.append(0)
                        microseconds = []
                        await websocket.send_text("no_scanning_color_change")

                    for box in boxes:
                        # bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                        # confidence
                        confidence = math.ceil((box.conf[0] * 100)) / 100
                        print("Confidence --->", confidence)
                        confidences.append(confidence)

                        if confidence < 0.7:
                            await websocket.send_text("no_scanning_color_change")
                            continue

                        print("TIME ELAPSED --->", time_elapsed)
                        if confidence > 0.7:
                            end_time = datetime.now()
                            diff = end_time - start_time
                            microseconds.append(diff.microseconds)
                            if sum(microseconds) >= 6000:
                                close_window = True

                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        # class name
                        cls = int(box.cls[0])
                        print("Class name -->", classNames[cls])
                        await websocket.send_text(classNames[cls])

                        # object details
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2

                        cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

                ret, buffer = cv2.imencode('.jpg', img)
                if not close_window:
                    await websocket.send_bytes(buffer.tobytes())
                else:
                    cap.release()
                    await websocket.send_text("redirect_page")
    except (WebSocketDisconnect, ConnectionClosed):
        print("Client disconnected")


@app.get('/home')
async def return_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get('/streaming')
async def streaming(request: Request):
    return templates.TemplateResponse("streaming.html", {"request": request})


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)