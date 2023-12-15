"""
This is main.py
"""
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates

# from module.load_model import load_fs, load_ner
# from module.input_process import make_ner_input, make_instance_list, input_data_loader
# from module.ner_utils import make_name_list, show_name_list
# from module.fs_utils import get_alias2id, find_speak


class AppData:
    def __init__(self):
        self.file_content = ""
        self.name_list = []
        self.place = []
        self.times = []


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
    return templates.TemplateResponse("confirm.html", {"request": request, "file_content": app_data.file_content})


@app.post("/upload", response_class=HTMLResponse)
async def upload_file(file: UploadFile = File(...)):
    """파일 업로드 및 저장"""
    with open("uploads/" + file.filename, "wb") as f:
        f.write(file.file.read())

    with open("uploads/" + file.filename, "r", encoding="utf-8") as f:
        app_data.file_content = f.read()

    return RedirectResponse(url="/put.html")


@app.post("/ners", response_class=JSONResponse)
async def ner_file():
    """저장된 파일에 NER 작업을 해서 화자랑 장소를 구분"""
    from module.load_model import load_ner
    from module.input_process import make_ner_input
    from module.ner_utils import make_name_list, show_name_list, combine_similar_names

    content = app_data.file_content
    _, ner_checkpoint = load_ner()

    contents = make_ner_input(content)
    name_list, place, times = make_name_list(contents, ner_checkpoint)
    name_dic = show_name_list(name_list)
    similar_name = combine_similar_names(name_dic)
    result_list = [', '.join(names) for names, _ in similar_name.items()]

    # JSONResponse로 응답
    return JSONResponse(content={"itemList": result_list})


@app.get("/find-speaker", response_class=HTMLResponse)
async def page_put(request: Request):
    """"""




# # 사용되는 값들 미리 불러오기
# name_list = []
# file_content =''
# check_name = 'data/names.txt'
# alias2id = get_alias2id(check_name)

# # 모델 불러오기
# # fs_model, fs_checkpoint = load_fs()
#


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
