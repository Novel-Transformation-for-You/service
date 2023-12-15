"""
This is main.py
"""
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

# from module.load_model import load_fs, load_ner
# from module.input_process import make_ner_input, make_instance_list, input_data_loader
# from module.ner_utils import make_name_list, show_name_list
# from module.fs_utils import get_alias2id, find_speak


class AppData:
    def __init__(self):
        self.file_content = ""
        self.name_list = []


app_data = AppData()

# 설정
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def page_home(request: Request):
    """INDEX.HTML 화면"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/put.html", response_class=HTMLResponse)
async def page_put(request: Request):
    """PUT.HTML 화면"""
    return templates.TemplateResponse("put.html", {"request": request})


@app.get("/confirm.html", response_class=HTMLResponse)
async def page_confirm(request: Request):
    """confirm.HTML 화면"""
    return templates.TemplateResponse("confirm.html", {"request": request})


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(file: UploadFile = File(...)):
    """파일 업로드 및 저장"""
    with open("uploads/" + file.filename, "wb") as f:
        f.write(file.file.read())
    app_data.file_content = f
    return RedirectResponse(url="/put.html")


@app.post("/ners", response_class=HTMLResponse)
async def ner_file(file: UploadFile = File(...)):
    """저장된 파일에 NER 작업"""
    from module.load_model import load_ner
    from module.input_process import make_ner_input
    from module.ner_utils import make_name_list, show_name_list

    cintent = app_data.file_content
    ner_model, ner_checkpoint = load_ner()

    contents = make_ner_input(cintent)
    name_list = make_name_list(contents, ner_checkpoint)
    show = show_name_list(name_list)


# app_data = AppData()

# # 사용되는 값들 미리 불러오기
# name_list = []
# file_content =''
# check_name = 'data/names.txt'
# alias2id = get_alias2id(check_name)

# # 모델 불러오기
# # fs_model, fs_checkpoint = load_fs()
# 

# # 현재 날짜와 시간을 이용하여 새로운 파일 이름 생성
# current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
# new_filename = f"{current_datetime}.txt"
# text_file_path = os.path.join(UPLOAD_FOLDER, new_filename)








# @app.post("/upload-and-redirect", response_class=HTMLResponse)
# async def upload_and_show(request: Request, file: UploadFile = File(...)):
#     """파일을 업로드하고 confirm.html로 결과를 보여줍니다."""
#     # 파일 저장
#     with open(text_file_path, "wb") as f:
#         f.write(file.file.read())

#     # 파일 내용을 가져와서 file_content 변수에 할당
#     with open(text_file_path, "r", encoding="utf-8") as f:
#         app_data.file_content = f.read()

#     content = make_ner_input(app_data.file_content)
#     app_data.name_list = make_name_list(content, ner_checkpoint)
#     show = show_name_list(app_data.name_list)

#     return templates.TemplateResponse("./confirm.html", {
#         "request": request, "name_list": show, "file_content": app_data.file_content})




# @app.post("/find-speaker")
# async def find_speaker(request: Request):
#     """발화자를 찾고 싶다."""
#     return templates.TemplateResponse("show.html", {
#         "request": request, "name_list": app_data.name_list})


# @app.get("/find-speaker", response_class=HTMLResponse)
# async def speaker_page(request: Request):
#     """화자를 찾아주자"""
#     instances = make_instance_list(app_data.file_content)
#     inputs = input_data_loader(instances, alias2id)
#     output = find_speak(fs_model, inputs, fs_checkpoint, alias2id)

#     return templates.TemplateResponse("result.html", {"request": request, "output": output})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
