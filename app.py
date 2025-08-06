import os
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
import datetime
from redis import Redis
from rq import Queue
from task import evaluate

# ---------------------- 基本配置 ----------------------
ALLOWED_EXTENSIONS = {".pth", ".pt"}   # 根据需求修改
UPLOAD_DIR = Path(__file__).parent / "uploaded_files"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = "change-this-to-a-random-secret"        # 用于 Session 加密
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

# 连接 Redis 并初始化队列
redis_conn = Redis(host="localhost", port=6380, db=0)
task_q     = Queue("file_tasks", connection=redis_conn)

def allowed(fname): return Path(fname).suffix.lower() in ALLOWED_EXTENSIONS

# ---------------------- 工具函数 ----------------------
def allowed_file(filename: str) -> bool:
    return (Path(filename).suffix.lower() in ALLOWED_EXTENSIONS)

# ---------------------- 路由 ----------------------
@app.route("/", methods=["GET", "POST"])
def login():
    if "username" in session:
        return redirect(url_for("upload"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        ...
        if not username:
            flash("请输入用户名")
            return redirect(url_for("login"))
        # 登录成功：写入 Session
        session["username"] = username
        return redirect(url_for("upload"))
    return render_template("login.html")

ALLOWED_EXTENSIONS = {".pt", ".pth"}            # 只允许 .pt

def allowed_file(fname):
    return Path(fname).suffix.lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        file = request.files.get("file")
        robot_type = request.form.get("robot_type")
        # --- 基本校验 ---
        if not robot_type or robot_type not in {"robot_1", "robot_2", "robot_3"}:
            flash("请选择 robot_1 / robot_2 / robot_3")
            return redirect(request.url)

        if not file or file.filename == "":
            flash("未选择文件")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash(f"仅支持{ALLOWED_EXTENSIONS}后缀")
            return redirect(request.url)

        # 此处是队伍id或者token，因此不需要检验是否安全
        token = session["username"]
        date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        user_dir = UPLOAD_DIR / token / date_str # ./uploaded_files/<用户名>
        user_dir.mkdir(parents=True, exist_ok=True)

        safe_name = secure_filename(file.filename)
        save_path = user_dir / safe_name
        file.save(save_path)
        task_q.enqueue(evaluate, str(save_path), token, robot_type)

        return render_template(
            "result.html",
            username=session["username"],
            filepath=save_path,
            notice="文件已入队处理，稍后可在系统中查看结果"
        )
    return render_template("upload.html", username=session["username"])

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------------- 入口 ----------------------
if __name__ == "__main__":
    app.run(host="10.19.125.13", port=5555, debug=True)