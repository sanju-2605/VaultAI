from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
import uuid

from modules.pdf_tools import pdf_to_docx, pdf_to_excel, excel_to_pdf
from modules.summarizer import summarize_text, extract_text_from_pdf, extract_text_from_docx, extract_text_from_excel
from modules.doc_chat import answer_question_langchain

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///vaultai.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = "your_secret_key_here"
db = SQLAlchemy(app)

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(30), default="employee")  # "admin" or "employee"
    documents = db.relationship("Document", backref="user", lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256))
    upload_time = db.Column(db.DateTime, server_default=db.func.now())
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    qas = db.relationship("QA", backref="document", lazy=True)

class QA(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(512))
    answer = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    document_id = db.Column(db.Integer, db.ForeignKey("document.id"))

# Create tables with app context
with app.app_context():
    db.create_all()

UPLOAD_FOLDER = 'temp_files'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Role-based access control decorator
def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user_id = session.get('user_id')
            if not user_id:
                flash("Login required.", "error")
                return redirect(url_for('login'))
            user = User.query.get(user_id)
            if not user or user.role not in roles:
                flash("Access denied.", "error")
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# inject user for templates
@app.context_processor
def inject_user():
    user = None
    if session.get("user_id"):
        user = User.query.get(session.get("user_id"))
    return dict(user=user)

# Home / Document Processing
@app.route("/", methods=["GET", "POST"])
def index():
    summary, chat_answer, answer_context, converted_file_name, uploaded_file_name = None, None, None, None, None
    previous_question = ""

    if request.method == "POST":
        try:
            uploaded_file = request.files.get("file")
            convert_to = request.form.get("convert_to", "")
            do_summarize = request.form.get("summarize") == "on"
            question = request.form.get("question", "").strip()
            previous_question = question

            if not uploaded_file or uploaded_file.filename == "":
                flash("No file uploaded.", "error")
                return render_template("index.html")

            file_ext = uploaded_file.filename.rsplit('.', 1)[-1].lower()
            unique_id = str(uuid.uuid4())[:8]
            temp_input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_input.{file_ext}")
            uploaded_file.save(temp_input_path)
            converted_file_path = temp_input_path
            uploaded_file_name = os.path.basename(converted_file_path)

            user_id = session.get("user_id")
            doc_entry = Document(filename=uploaded_file.filename, user_id=user_id)
            db.session.add(doc_entry)
            db.session.commit()

            if convert_to and convert_to != file_ext:
                if convert_to == "docx" and file_ext == "pdf":
                    converted_file_path = pdf_to_docx(temp_input_path)
                elif convert_to == "xlsx" and file_ext == "pdf":
                    converted_file_path = pdf_to_excel(temp_input_path)
                elif convert_to == "pdf" and file_ext in ("xls", "xlsx"):
                    converted_file_path = excel_to_pdf(temp_input_path)
                else:
                    flash(f"Conversion from {file_ext} to {convert_to} is not supported.", "error")
                    return render_template("index.html")
                converted_file_name = os.path.basename(converted_file_path)
                flash("File converted successfully!", "success")
            else:
                converted_file_name = os.path.basename(converted_file_path)

            ext = convert_to if convert_to else file_ext
            if ext == "pdf":
                raw_text = extract_text_from_pdf(converted_file_path)
            elif ext == "docx":
                raw_text = extract_text_from_docx(converted_file_path)
            elif ext in ("xls", "xlsx"):
                raw_text = extract_text_from_excel(converted_file_path)
            else:
                raw_text = ""

            if do_summarize and raw_text.strip():
                summary = summarize_text(raw_text, sentences_count=5)
                flash("Document summarized successfully!", "success")

            if question and raw_text.strip():
                chat_answer, answer_context = answer_question_langchain(
                    converted_file_path, ext, question)
                if doc_entry:
                    qa_entry = QA(
                        question=question, answer=chat_answer, document_id=doc_entry.id
                    )
                    db.session.add(qa_entry)
                    db.session.commit()
        except Exception as e:
            flash(f"Processing failed: {e}", "error")

    user_id = session.get("user_id")
    if user_id:
        recent_qas = QA.query.join(Document).filter(Document.user_id == user_id).order_by(QA.timestamp.desc()).limit(10).all()
    else:
        recent_qas = []

    return render_template(
        "index.html",
        summary=summary,
        chat_answer=chat_answer,
        answer_context=answer_context,
        uploaded_file=uploaded_file_name,
        converted_file=converted_file_name,
        previous_question=previous_question,
        recent_qas=recent_qas,
    )

# Register
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        role = "admin" if request.args.get("as") == "admin" else "employee"
        if User.query.filter_by(username=username).first():
            flash("Username already exists!", "error")
            return render_template("register.html")
        if User.query.filter_by(email=email).first():
            flash("Email already exists!", "error")
            return render_template("register.html")
        user = User(username=username, email=email, role=role)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for('login'))
    return render_template("register.html")

# Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session["user_id"] = user.id
            flash("Logged in successfully.", "success")
            if user.role == "admin":
                return redirect(url_for("admin_dashboard"))
            else:
                return redirect(url_for("index"))
        else:
            flash("Invalid username or password.", "error")
            return render_template("login.html")
    return render_template("login.html")

# Logout
@app.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("Logged out.", "success")
    return redirect(url_for("login"))

# Admin dashboard
@app.route("/admin")
@role_required("admin")
def admin_dashboard():
    users = User.query.all()
    docs = Document.query.order_by(Document.upload_time.desc()).limit(10).all()
    return render_template("admin_dashboard.html", users=users, docs=docs)

# Download Converted File
@app.route("/download/<filename>")
def download_file(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    flash("File not found", "error")
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
