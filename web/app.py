"""Flask web app cho Virtual Assistant.

Routes:
    GET  /                         → home, list users
    GET  /enroll                   → form đăng ký user mới
    POST /api/enroll               → multipart upload nhiều audio mẫu
    GET  /users/<id>               → user detail + edit preferences
    POST /api/users/<id>/update    → update preferences
    POST /api/users/<id>/delete    → xóa user
    GET  /assistant                → chat UI
    POST /api/assistant/turn       → audio in → JSON {transcript, intent, ...}
    GET  /api/tts?text=...         → MP3 bytes cho browser play

Chạy:
    python -m web.app
    # → http://localhost:5000
"""
import io
import json
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

from flask import (Flask, render_template, request, jsonify, send_file,
                   redirect, url_for, flash)

# Thêm project root vào path để import được core/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import audio_io, config
from core.asr import get_asr
from core.tts import get_tts
from core.nlu import get_nlu
from core.database import UserDB, SpeakerManager
from core.router import Router


# ==========================================================================
# App init — tất cả model load 1 lần lúc startup (singleton)
# ==========================================================================
def create_app():
    app = Flask(__name__,
                template_folder="templates",
                static_folder="static")
    app.secret_key = "dev-secret-change-in-production"
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB cho upload audio

    # Eager-load các model — startup chậm 1 chút nhưng request sẽ nhanh
    print("=" * 60)
    print("Loading models (lần đầu mất 30-60s)...")
    db = UserDB()
    spk_mgr = SpeakerManager(db)
    asr = get_asr()
    tts = get_tts()
    nlu = get_nlu()
    router = Router(spk_mgr)
    print("✓ Tất cả model đã sẵn sàng.")
    print("=" * 60)

    # Lưu vào app context để các handler dùng
    app.config["db"] = db
    app.config["spk_mgr"] = spk_mgr
    app.config["asr"] = asr
    app.config["tts"] = tts
    app.config["nlu"] = nlu
    app.config["router"] = router

    register_routes(app)
    return app


# ==========================================================================
# Routes
# ==========================================================================
def register_routes(app):

    # ----- Pages -----
    @app.route("/")
    def home():
        users = app.config["db"].list_users()
        return render_template("home.html", users=users)

    @app.route("/enroll")
    def enroll_page():
        return render_template("enroll.html",
                               num_samples=config.ENROLL_NUM_SAMPLES,
                               duration=config.ENROLL_DURATION)

    @app.route("/users/<user_id>")
    def user_detail(user_id):
        user = app.config["db"].get_user(user_id)
        if not user:
            flash(f"Không tìm thấy user '{user_id}'", "error")
            return redirect(url_for("home"))
        return render_template("user_detail.html", user=user)

    @app.route("/assistant")
    def assistant_page():
        users = app.config["db"].list_users()
        return render_template("assistant.html", users=users)

    # ----- API: Enrollment -----
    @app.route("/api/enroll", methods=["POST"])
    def api_enroll():
        """Body multipart:
           - user_id: text
           - name: text
           - preferences: JSON text (optional)
           - sample_0, sample_1, ... : audio blobs (WebM/wav)
        """
        user_id = request.form.get("user_id", "").strip()
        name = request.form.get("name", "").strip()
        prefs_str = request.form.get("preferences", "{}")

        # Validate
        if not user_id or " " in user_id:
            return jsonify({"error": "user_id không hợp lệ (không trống, không space)"}), 400
        if not name:
            return jsonify({"error": "Thiếu name"}), 400
        try:
            preferences = json.loads(prefs_str)
        except json.JSONDecodeError:
            return jsonify({"error": "preferences không phải JSON hợp lệ"}), 400

        db = app.config["db"]
        spk_mgr = app.config["spk_mgr"]
        if db.get_user(user_id):
            return jsonify({"error": f"User '{user_id}' đã tồn tại"}), 409

        # Decode + VAD trim từng sample
        audios = []
        sample_dir = config.ENROLL_AUDIO_DIR / user_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        for key in sorted(request.files.keys()):
            if not key.startswith("sample_"):
                continue
            blob = request.files[key].read()
            try:
                audio = audio_io.decode_browser_audio(blob)
            except Exception as e:
                return jsonify({"error": f"Decode {key} fail: {e}"}), 400

            audio = audio_io.SileroVAD.trim(audio)
            if audio.size < config.SAMPLE_RATE:  # < 1s
                return jsonify({
                    "error": f"{key} quá ngắn sau VAD ({audio.size/16000:.2f}s). "
                             "Hãy đảm bảo nói rõ ràng đủ thời lượng."
                }), 400

            audio_io.save_wav(audio, sample_dir / f"{key}.wav")
            audios.append(audio)

        if len(audios) < 2:
            return jsonify({"error": f"Cần ít nhất 2 mẫu, chỉ có {len(audios)}"}), 400

        # Enroll
        try:
            centroid = spk_mgr.enroll(user_id, name, audios, preferences)
        except Exception as e:
            return jsonify({"error": f"Enroll fail: {e}"}), 500

        return jsonify({
            "ok": True,
            "user_id": user_id,
            "name": name,
            "n_samples": len(audios),
            "centroid_norm": float((centroid**2).sum()**0.5),
        })

    # ----- API: User CRUD -----
    @app.route("/api/users/<user_id>/update", methods=["POST"])
    def api_update_prefs(user_id):
        db = app.config["db"]
        if not db.get_user(user_id):
            return jsonify({"error": "User không tồn tại"}), 404
        try:
            preferences = request.get_json()["preferences"]
            if not isinstance(preferences, dict):
                return jsonify({"error": "preferences phải là object"}), 400
        except (KeyError, TypeError):
            return jsonify({"error": "Body cần field 'preferences'"}), 400

        db.update_preferences(user_id, preferences)
        return jsonify({"ok": True})

    @app.route("/api/users/<user_id>/delete", methods=["POST"])
    def api_delete_user(user_id):
        db = app.config["db"]
        if not db.get_user(user_id):
            return jsonify({"error": "User không tồn tại"}), 404
        db.delete_user(user_id)
        # Invalidate cache trong SpeakerManager
        app.config["spk_mgr"]._cache = None
        return jsonify({"ok": True})

    # ----- API: Assistant turn -----
    @app.route("/api/assistant/turn", methods=["POST"])
    def api_assistant_turn():
        """Body: multipart với file 'audio' (WebM hoặc wav).

        Trả về:
          {
            transcript, intent, auth_level, entities,
            identified_user_id, identified_user_name, sid_score,
            sv_required, sv_passed, sv_score,
            response, blocked,
            tts_url    # GET URL để stream MP3 response
          }
        """
        if "audio" not in request.files:
            return jsonify({"error": "Thiếu file 'audio'"}), 400

        blob = request.files["audio"].read()
        try:
            audio = audio_io.decode_browser_audio(blob)
        except Exception as e:
            return jsonify({"error": f"Decode audio fail: {e}"}), 400

        audio = audio_io.SileroVAD.trim(audio)
        if audio.size < config.SAMPLE_RATE // 2:
            return jsonify({"error": "Audio quá ngắn / không có speech"}), 400

        # ASR → NLU → Router
        asr = app.config["asr"]
        nlu = app.config["nlu"]
        router = app.config["router"]

        transcript = asr.transcribe(audio)
        if not transcript:
            return jsonify({"error": "ASR không nhận diện được"}), 400

        nlu_result = nlu.parse(transcript)
        result = router.handle_turn(audio, transcript, nlu_result)

        payload = asdict(result)
        # Thêm URL để client gọi TTS
        from urllib.parse import quote
        payload["tts_url"] = f"/api/tts?text={quote(result.response)}"
        return jsonify(payload)

    # ----- API: TTS streaming -----
    @app.route("/api/tts")
    def api_tts():
        text = request.args.get("text", "").strip()
        if not text:
            return jsonify({"error": "Thiếu param 'text'"}), 400
        try:
            mp3 = app.config["tts"].synthesize_to_mp3_bytes(text)
        except Exception as e:
            return jsonify({"error": f"TTS fail: {e}"}), 500
        return send_file(io.BytesIO(mp3), mimetype="audio/mpeg",
                         as_attachment=False)

    # ----- Misc -----
    @app.route("/api/health")
    def health():
        return jsonify({
            "ok": True,
            "n_users": len(app.config["db"].list_users()),
            "encoder_mode": app.config["spk_mgr"].encoder.mode,
            "asr_model": config.WHISPER_MODEL,
            "nlu": app.config["nlu"].__class__.__name__,
        })


# ==========================================================================
# Entrypoint
# ==========================================================================
if __name__ == "__main__":
    app = create_app()
    # threaded=False vì model torch không thread-safe theo default
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
