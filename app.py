import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, send_from_directory
)
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

# ======================================================
# BASE DIRECTORY (Python & EXE Compatible)
# ======================================================
def base_dir():
    """Return correct base directory for Python and PyInstaller EXE."""
    if hasattr(sys, "_MEIPASS"):  # Running inside EXE
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = base_dir()

print("BASE_DIR =", BASE_DIR)

# Folders inside BASE_DIR
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR   = os.path.join(BASE_DIR, "static")
UPLOAD_DIR   = os.path.join(STATIC_DIR, "uploads")
MODEL_DIR    = os.path.join(BASE_DIR, "saved_models")
DB_PATH      = os.path.join(BASE_DIR, "alzheimer.db")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ======================================================
# FLASK APP SETUP
# ======================================================
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key = "your_secret_key_here"
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# ======================================================
# STATIC SERVING FOR UPLOADS
# ======================================================
@app.route("/uploads/<path:filename>")
def uploaded_files(filename):
    """Serve uploaded images."""
    return send_from_directory(UPLOAD_DIR, filename)

# ======================================================
# DATABASE
# ======================================================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    db = get_db()

    # USERS TABLE
    db.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # PREDICTIONS TABLE
    db.execute("""
        CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            model_used TEXT,
            image_path TEXT,
            grad_cam_path TEXT,
            prediction TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    # ADMINS TABLE
    db.execute("""
        CREATE TABLE IF NOT EXISTS admin(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT DEFAULT 'admin'
        )
    """)

    # AUDIT LOG TABLE
    db.execute("""
        CREATE TABLE IF NOT EXISTS admin_logs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_username TEXT,
            action TEXT,
            target TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create first SuperAdmin
    admin = db.execute("SELECT * FROM admin").fetchone()
    if not admin:
        db.execute(
            "INSERT INTO admin(username,password,role) VALUES (?,?,?)",
            ("admin", generate_password_hash("admin123"), "superadmin")
        )

    db.commit()
    db.close()


def log_action(admin_username, action, target):
    db = get_db()
    db.execute(
        "INSERT INTO admin_logs(admin_username, action, target) VALUES (?,?,?)",
        (admin_username, action, target)
    )
    db.commit()

init_db()

# ======================================================
# MODEL PATHS
# ======================================================
MODEL_PATHS = {
    "model1": os.path.join(MODEL_DIR, "model1_TASK_02.h5"),
    "model2": os.path.join(MODEL_DIR, "model2_TASK_03.h5"),
    "model3": os.path.join(MODEL_DIR, "model3_TASK_04.h5"),
    "model4": os.path.join(MODEL_DIR, "model4_TASK_05.h5"),
}

MODEL_INFO = {
    "model1": {"name": "Model 1"},
    "model2": {"name": "Model 2"},
    "model3": {"name": "Model 3"},
    "model4": {"name": "Model 4"},
}

loaded_models = {}

# ======================================================
# LOAD MODEL (safe for Python + EXE)
# ======================================================
def load_selected_model(model_key):
    if model_key not in MODEL_PATHS:
        raise ValueError("Unknown model:", model_key)

    model_path = MODEL_PATHS[model_key]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if model_key not in loaded_models:
        print("Loading model:", model_path)
        loaded_models[model_key] = load_model(model_path)

    return loaded_models[model_key]

# ======================================================
# IMAGE PROCESSING & GRAD CAM
# ======================================================
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img.astype("float32"))
    return img


def generate_grad_cam(model, img_array, layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        loss = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(conv_outputs[0], weights.numpy())
    cam = np.maximum(cam, 0)
    return cam / (cam.max() if cam.max() != 0 else 1)

# ======================================================
# AUTH HELPERS
# ======================================================
def require_superadmin():
    if session.get("role") != "superadmin":
        flash("Access denied! SuperAdmin only.", "error")
        return False
    return True

# ======================================================
# ROUTES
# ======================================================

@app.route("/")
def index():
    return render_template("index.html")

# ------------------ USER AUTH -------------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        db = get_db()
        try:
            db.execute(
                "INSERT INTO users(username,email,password) VALUES (?,?,?)",
                (
                    request.form["username"],
                    request.form["email"],
                    generate_password_hash(request.form["password"])
                )
            )
            db.commit()
            flash("Account created!", "success")
            return redirect(url_for("login"))
        except:
            flash("Username or Email already exists!", "error")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE username=?",
            (request.form["username"],)
        ).fetchone()

        if user and check_password_hash(user["password"], request.form["password"]):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("dashboard"))

        flash("Invalid login!", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

# ------------------ DASHBOARD -------------------------

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    db = get_db()
    preds = db.execute(
        "SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 5",
        (session["user_id"],)
    ).fetchall()

    return render_template(
        "dashboard.html",
        predictions=preds,
        model_info=MODEL_INFO,
        selected_model=session.get("selected_model", "model3")
    )


@app.route("/set_model", methods=["POST"])
def set_model():
    m = request.form["model"]
    if m in MODEL_PATHS:
        session["selected_model"] = m
        flash("Model changed!", "success")
    return redirect(url_for("dashboard"))

# ------------------ PREDICTION -------------------------

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":

        file = request.files.get("image")
        if not file or file.filename == "":
            flash("No image selected!", "error")
            return redirect("/predict")

        filename = f"{session['user_id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)

        selected = request.form.get("model_choice", "auto")

        fname = file.filename.lower()
        auto = None
        for i in range(1, 7):
            if f"model{i}" in fname:
                auto = f"model{i}"

        if selected == "auto":
            selected = auto if auto else session.get("selected_model", "model3")

        model = load_selected_model(selected)

        img = np.array(Image.open(filepath).convert("RGB"))
        processed = preprocess_image(img)
        processed = np.expand_dims(processed, 0)

        preds = model.predict(processed, verbose=0)
        confidence = float(np.max(preds))
        label = "Alzheimer's Disease" if np.argmax(preds) == 0 else "Healthy Control"

        # GRAD CAM
        heatmap = generate_grad_cam(model, processed)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        gc_filename = f"gc_{filename}"
        gc_path = os.path.join(UPLOAD_DIR, gc_filename)
        plt.imsave(gc_path, blended)

        db = get_db()
        db.execute(
            """INSERT INTO predictions(user_id,model_used,image_path,grad_cam_path,prediction,confidence)
               VALUES (?,?,?,?,?,?)""",
            (session["user_id"], selected, filename, gc_filename, label, confidence)
        )
        db.commit()

        return render_template(
            "result.html",
            prediction=label,
            confidence=confidence,
            original_image=f"/uploads/{filename}",
            grad_cam_image=f"/uploads/{gc_filename}"
        )

    return render_template("predict.html", model_info=MODEL_INFO)

# ------------------ HISTORY -------------------------

@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect(url_for("login"))

    db = get_db()
    preds = db.execute(
        "SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC",
        (session["user_id"],)
    ).fetchall()

    return render_template("history.html", predictions=preds)

# ======================================================
# ADMIN PANEL
# ======================================================

@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        db = get_db()

        user = db.execute(
            "SELECT * FROM admin WHERE username=?",
            (request.form["username"],)
        ).fetchone()

        if user and check_password_hash(user["password"], request.form["password"]):
            session["admin"] = user["username"]
            session["role"] = user["role"]
            session["admin_id"] = user["id"]
            return redirect(url_for("admin_dashboard"))

        flash("Invalid admin login", "error")

    return render_template("admin/login.html")

@app.route("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    session.pop("role", None)
    session.pop("admin_id", None)
    flash("Logged out from Admin Panel", "success")
    return redirect(url_for("admin_login"))


@app.route("/admin/dashboard")
def admin_dashboard():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    db = get_db()
    users = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    preds = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    return render_template("admin/dashboard.html", users=users, predictions=preds)


# ------------------ MANAGE USERS -------------------------
@app.route("/admin/users")
def admin_users():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    db = get_db()
    users = db.execute("SELECT * FROM users").fetchall()
    return render_template("admin/users.html", users=users)


@app.route("/admin/user/add", methods=["GET", "POST"])
def admin_add_user():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    if request.method == "POST":
        db = get_db()
        try:
            db.execute(
                "INSERT INTO users(username,email,password) VALUES (?,?,?)",
                (
                    request.form["username"],
                    request.form["email"],
                    generate_password_hash(request.form["password"])
                )
            )
            db.commit()
            flash("User added!", "success")
            return redirect(url_for("admin_users"))
        except:
            flash("User exists!", "error")

    return render_template("admin/add_user.html")


@app.route("/admin/user/edit/<int:uid>", methods=["GET", "POST"])
def admin_edit_user(uid):
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()

    if request.method == "POST":
        db.execute(
            "UPDATE users SET username=?, email=? WHERE id=?",
            (request.form["username"], request.form["email"], uid)
        )
        db.commit()
        flash("Updated!", "success")
        return redirect(url_for("admin_users"))

    return render_template("admin/edit_user.html", user=user)


@app.route("/admin/user/delete/<int:uid>")
def admin_delete_user(uid):
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    db = get_db()
    db.execute("DELETE FROM users WHERE id=?", (uid,))
    db.commit()
    flash("User deleted!", "success")
    return redirect(url_for("admin_users"))


# ------------------ PREDICTIONS -------------------------
@app.route("/admin/predictions")
def admin_predictions():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    db = get_db()
    preds = db.execute("SELECT * FROM predictions ORDER BY created_at DESC").fetchall()
    return render_template("admin/predictions.html", predictions=preds)


@app.route("/admin/prediction/<int:pid>")
def admin_pred_detail(pid):
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    db = get_db()
    pred = db.execute("SELECT * FROM predictions WHERE id=?", (pid,)).fetchone()
    return render_template("admin/prediction_detail.html", pred=pred)


# ------------------ ADMIN MANAGEMENT -------------------------
@app.route("/admin/admins")
def manage_admins():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    if not require_superadmin():
        return redirect(url_for("admin_dashboard"))

    db = get_db()
    admins = db.execute("SELECT * FROM admin").fetchall()
    return render_template("admin/admins.html", admins=admins)


@app.route("/admin/admins/add", methods=["GET", "POST"])
def add_admin():
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    if not require_superadmin():
        return redirect(url_for("manage_admins"))

    if request.method == "POST":
        db = get_db()
        try:
            db.execute(
                "INSERT INTO admin(username,password,role) VALUES (?,?,?)",
                (
                    request.form["username"],
                    generate_password_hash(request.form["password"]),
                    request.form["role"]
                )
            )
            db.commit()
            log_action(session["admin"], "ADD_ADMIN", request.form["username"])
            flash("Admin added!", "success")
            return redirect(url_for("manage_admins"))
        except:
            flash("Admin username exists!", "error")

    return render_template("admin/add_admin.html")


@app.route("/admin/admins/edit/<int:aid>", methods=["GET", "POST"])
def edit_admin(aid):
    if "admin" not in session:
        return redirect(url_for("admin_login"))

    if not require_superadmin():
        return redirect(url_for("manage_admins"))

    db = get_db()
    admin_user = db.execute("SELECT * FROM admin WHERE id=?", (aid,)).fetchone()

    if request.method == "POST":
        db.execute(
            "UPDATE admin SET username=?, role=? WHERE id=?",
            (request.form["username"], request.form["role"], aid)
        )
        db.commit()
        log_action(session["admin"], "EDIT_ADMIN", request.form["username"])
        flash("Admin updated!", "success")
        return redirect(url_for("manage_admins"))

    return render_template("admin/edit_admin.html", admin=admin_user)


@app.route("/admin/admins/password/<int:aid>", methods=["GET", "POST"])
def admin_change_password(aid):
    if not require_superadmin():
        return redirect(url_for("manage_admins"))

    db = get_db()
    admin_user = db.execute("SELECT * FROM admin WHERE id=?", (aid,)).fetchone()

    if request.method == "POST":
        new_pass = generate_password_hash(request.form["password"])
        db.execute("UPDATE admin SET password=? WHERE id=?", (new_pass, aid))
        db.commit()
        log_action(session["admin"], "CHANGE_ADMIN_PASSWORD", f"admin_id={aid}")
        flash("Password updated!", "success")
        return redirect(url_for("manage_admins"))

    return render_template("admin/change_admin_password.html", admin=admin_user)


@app.route("/admin/admins/promote/<int:aid>")
def promote_admin(aid):
    if not require_superadmin():
        return redirect(url_for("manage_admins"))

    db = get_db()
    db.execute("UPDATE admin SET role='superadmin' WHERE id=?", (aid,))
    db.commit()

    log_action(session["admin"], "PROMOTE_ADMIN", f"admin_id={aid}")

    flash("Admin promoted to superadmin!", "success")
    return redirect(url_for("manage_admins"))


@app.route("/admin/admins/demote/<int:aid>")
def demote_admin(aid):
    if not require_superadmin():
        return redirect(url_for("manage_admins"))

    if session.get("admin_id") == aid:
        flash("You cannot demote yourself!", "error")
        return redirect(url_for("manage_admins"))

    db = get_db()
    db.execute("UPDATE admin SET role='admin' WHERE id=?", (aid,))
    db.commit()

    log_action(session["admin"], "DEMOTE_ADMIN", f"admin_id={aid}")

    flash("Admin demoted!", "success")
    return redirect(url_for("manage_admins"))


@app.route("/admin/admins/delete/<int:aid>")
def delete_admin(aid):
    if not require_superadmin():
        return redirect(url_for("manage_admins"))

    if session.get("admin_id") == aid:
        flash("You cannot delete your own admin account!", "error")
        return redirect(url_for("manage_admins"))

    db = get_db()
    db.execute("DELETE FROM admin WHERE id=?", (aid,))
    db.commit()

    log_action(session["admin"], "DELETE_ADMIN", f"admin_id={aid}")

    flash("Admin deleted!", "success")
    return redirect(url_for("manage_admins"))


# ------------------ AUDIT LOGS -------------------------
@app.route("/admin/logs")
def admin_logs():
    if not require_superadmin():
        return redirect(url_for("admin_dashboard"))

    db = get_db()
    logs = db.execute("SELECT * FROM admin_logs ORDER BY timestamp DESC").fetchall()
    return render_template("admin/logs.html", logs=logs)



# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    print("Running on http://127.0.0.1:5001")
    app.run(debug=False, port=5001)
