# API
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import uvicorn
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Main
from lib.VIOCR.VIOCR import Process_VIOCR as VIOCR
img_path = "examples/test.jpg"
res = VIOCR(
    img_path, 
    bbox_padding=5,
    same_line_max_bbox_y_diff=10, 
    new_line_char="\n", 
    same_line_char=" | ", 
    print_debug=False
)


@app.get("/")
def booooo_2():
    return {
        "response": f"{res}"
    }

if not os.getenv("OS_ENV_DOCKER"):
    uvicorn.run(
        app, 
        host = "127.0.0.1",
        port = 6969,
    )