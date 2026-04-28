"""Flask web app cho Virtual Assistant.

Routes:
    GET  /                         → home, list users
    GET  /enroll                   → form đăng ký user mới
    POST /api/enroll               → multipart upload nhiều audio mẫu
    GET  /users/<id>               → user detail + edit preferences
    GET  /api/users/<id>           → trả về user dict + has_password
    POST /api/users/<id>/update    → update preferences (cần password)
    POST /api/users/<id>/update-info → cập nhật name / password
    POST /api/users/<id>/delete    → xóa user (cần password)
    GET  /assistant                → chat UI
    POST /api/assistant/turn       → audio in → JSON {transcript, intent, ...}
    GET  /api/tts?text=...         → MP3 bytes cho browser play
    POST /api/files/verify-password → xác thực bằng password thay vì giọng

Chạy:
    python -m web.app
    # → http://localhost:5000
"""
import io
import json
import sys
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from flask import (Flask, render_template, request, jsonify, send_file,
                   redirect, url_for, flash, session)
from werkzeug.utils import secure_filename

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
        return render_template("home.html", users=users,
                               num_samples=config.ENROLL_NUM_SAMPLES,
                               duration=config.ENROLL_DURATION)

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
           - password: text (optional)
           - sample_0, sample_1, ... : audio blobs (WebM/wav)
        """
        user_id = request.form.get("user_id", "").strip()
        name = request.form.get("name", "").strip()
        prefs_str = request.form.get("preferences", "{}")
        password = request.form.get("password", "").strip()

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
            centroid = spk_mgr.enroll(user_id, name, audios, preferences, password)
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
    @app.route("/api/users/<user_id>", methods=["GET"])
    def api_get_user(user_id):
        db = app.config["db"]
        user = db.get_user(user_id)
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404
        result = dict(user)
        result["has_password"] = db.has_password(user_id)
        return jsonify(result)

    @app.route("/api/users/<user_id>/update", methods=["POST"])
    def api_update_prefs(user_id):
        db = app.config["db"]
        if not db.get_user(user_id):
            return jsonify({"error": "User không tồn tại"}), 404
        try:
            data = request.get_json()
            preferences = data["preferences"]
            if not isinstance(preferences, dict):
                return jsonify({"error": "preferences phải là object"}), 400
        except (KeyError, TypeError):
            return jsonify({"error": "Body cần field 'preferences'"}), 400

        password = data.get("password", "")
        if not db.check_password(user_id, password):
            return jsonify({"error": "Sai mật khẩu"}), 403

        db.update_preferences(user_id, preferences)
        return jsonify({"ok": True})

    @app.route("/api/users/<user_id>/update-info", methods=["POST"])
    def api_update_user_info(user_id):
        db = app.config["db"]
        if not db.get_user(user_id):
            return jsonify({"error": "User không tồn tại"}), 404
        data = request.get_json() or {}
        password = data.get("password", "")
        if not db.check_password(user_id, password):
            return jsonify({"error": "Sai mật khẩu"}), 403

        if "name" in data:
            db.update_user_name(user_id, data["name"])
        new_password = data.get("new_password", "")
        if new_password:
            db.update_password(user_id, new_password)
        return jsonify({"ok": True})

    @app.route("/api/users/<user_id>/delete", methods=["POST"])
    def api_delete_user(user_id):
        db = app.config["db"]
        if not db.get_user(user_id):
            return jsonify({"error": "User không tồn tại"}), 404
        data = request.get_json() or {}
        password = data.get("password", "")
        if not db.check_password(user_id, password):
            return jsonify({"error": "Sai mật khẩu"}), 403

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
        from urllib.parse import quote as _quote
        payload["tts_url"] = f"/api/tts?text={_quote(result.response)}"

        # action_type + action_data cho frontend mở panel tương ứng
        if result.intent == "play_music" and not result.blocked:
            uid = result.identified_user_id
            _user = app.config["db"].get_user(uid) if uid else None
            _prefs = _user["preferences"] if _user else {}
            genre  = result.entities.get("genre") or _prefs.get("favorite_genre", "pop")
            artist = _prefs.get("favorite_artist", "")
            query  = f"{artist} {genre}".strip() if artist else genre
            payload["action_type"] = "play_music"
            payload["action_data"] = {"query": query, "uid": uid or ""}

        elif result.intent == "open_files" and not result.blocked:
            uid = result.identified_user_id
            user_dir = config.USER_FILES_DIR / uid
            files = []
            if user_dir.exists():
                for _f in sorted(user_dir.iterdir()):
                    if _f.is_file():
                        files.append({
                            "name": _f.name,
                            "size": _f.stat().st_size,
                            "modified": datetime.fromtimestamp(
                                _f.stat().st_mtime).strftime("%d/%m/%Y %H:%M"),
                        })
            # Cấp quyền session cho panel files
            session["file_uid"]  = uid
            session["file_name"] = result.identified_user_name
            payload["action_type"] = "show_files"
            payload["action_data"] = {
                "files": files,
                "user_name": result.identified_user_name,
                "user_id": uid,
            }

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

    # ==========================================================================
    # Music routes
    # ==========================================================================
    @app.route("/music")
    def music_page():
        users = app.config["db"].list_users()
        return render_template("music.html", users=users)

    @app.route("/api/music/search")
    def api_music_search():
        q = request.args.get("q", "").strip()
        if not q:
            return jsonify({"tracks": []})
        try:
            import requests as _req
            resp = _req.get("https://api.deezer.com/search",
                            params={"q": q, "limit": 20}, timeout=6)
            tracks = [
                {"id": t["id"], "title": t["title"],
                 "artist": t["artist"]["name"],
                 "cover": t["album"]["cover_small"],
                 "preview": t["preview"]}
                for t in resp.json().get("data", []) if t.get("preview")
            ]
        except Exception as e:
            return jsonify({"error": str(e), "tracks": []})
        return jsonify({"tracks": tracks})

    @app.route("/api/music/playlist/<user_id>")
    def api_playlist_get(user_id):
        user = app.config["db"].get_user(user_id)
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404
        return jsonify({
            "playlist": user["preferences"].get("playlist", []),
            "favorite_genre": user["preferences"].get("favorite_genre", ""),
            "favorite_artist": user["preferences"].get("favorite_artist", ""),
        })

    @app.route("/api/music/playlist/<user_id>/add", methods=["POST"])
    def api_playlist_add(user_id):
        db = app.config["db"]
        user = db.get_user(user_id)
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404
        track = request.get_json()
        if not track or not {"id", "title", "artist", "preview"}.issubset(track):
            return jsonify({"error": "Thiếu field track"}), 400
        playlist = user["preferences"].get("playlist", [])
        if not any(t["id"] == track["id"] for t in playlist):
            playlist.append({k: track[k] for k in
                             ["id", "title", "artist", "cover", "preview"] if k in track})
            db.update_preferences(user_id, {**user["preferences"], "playlist": playlist})
        return jsonify({"ok": True, "count": len(playlist)})

    @app.route("/api/music/playlist/<user_id>/remove", methods=["POST"])
    def api_playlist_remove(user_id):
        db = app.config["db"]
        user = db.get_user(user_id)
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404
        track_id = (request.get_json() or {}).get("id")
        playlist = [t for t in user["preferences"].get("playlist", [])
                    if t["id"] != track_id]
        db.update_preferences(user_id, {**user["preferences"], "playlist": playlist})
        return jsonify({"ok": True, "count": len(playlist)})

    # ==========================================================================
    # File manager routes (SV-protected via Flask session)
    # ==========================================================================
    @app.route("/files")
    def files_page():
        return render_template("files.html")

    @app.route("/api/files/verify", methods=["POST"])
    def api_files_verify():
        if "audio" not in request.files:
            return jsonify({"error": "Thiếu audio"}), 400
        blob = request.files["audio"].read()
        try:
            aud = audio_io.decode_browser_audio(blob)
        except Exception as e:
            return jsonify({"error": f"Decode fail: {e}"}), 400
        aud = audio_io.SileroVAD.trim(aud)
        if aud.size < config.SAMPLE_RATE // 2:
            return jsonify({"error": "Audio quá ngắn / không có giọng nói"}), 400
        spk_mgr = app.config["spk_mgr"]
        uid, name, sid_score = spk_mgr.identify(aud)
        if uid is None:
            return jsonify({"ok": False,
                            "error": f"Không nhận ra giọng (score={sid_score:.2f}). "
                                     "Đăng ký trước hoặc nói rõ hơn."}), 403
        sv_passed, sv_score = spk_mgr.verify(aud, uid)
        if not sv_passed:
            return jsonify({"ok": False,
                            "error": f"Xác thực thất bại (score={sv_score:.2f}). "
                                     "Giọng không khớp với dữ liệu đã đăng ký."}), 403
        session["file_uid"] = uid
        session["file_name"] = name
        return jsonify({"ok": True, "user_id": uid, "user_name": name,
                        "sid_score": round(sid_score, 3),
                        "sv_score": round(sv_score, 3)})

    @app.route("/api/files/verify-password", methods=["POST"])
    def api_files_verify_password():
        data = request.get_json() or {}
        user_id = data.get("user_id", "")
        password = data.get("password", "")
        db = app.config["db"]
        user = db.get_user(user_id)
        if not user:
            return jsonify({"error": "User không tồn tại"}), 404
        if not db.check_password(user_id, password):
            return jsonify({"error": "Sai mật khẩu"}), 403
        session["file_uid"] = user_id
        session["file_name"] = user.get("name", user_id)
        return jsonify({"ok": True, "user_name": user.get("name", user_id)})

    @app.route("/api/files/logout", methods=["POST"])
    def api_files_logout():
        session.pop("file_uid", None)
        session.pop("file_name", None)
        return jsonify({"ok": True})

    @app.route("/api/files/list")
    def api_files_list():
        uid = session.get("file_uid")
        if not uid:
            return jsonify({"error": "Chưa xác thực"}), 403
        user_dir = config.USER_FILES_DIR / uid
        files = []
        if user_dir.exists():
            for f in sorted(user_dir.iterdir()):
                if f.is_file():
                    files.append({
                        "name": f.name,
                        "size": f.stat().st_size,
                        "modified": datetime.fromtimestamp(
                            f.stat().st_mtime).strftime("%d/%m/%Y %H:%M"),
                    })
        return jsonify({"files": files,
                        "user_name": session.get("file_name"),
                        "user_id": uid})

    @app.route("/api/files/upload", methods=["POST"])
    def api_files_upload():
        uid = session.get("file_uid")
        if not uid:
            return jsonify({"error": "Chưa xác thực"}), 403
        uploaded = []
        for key in request.files:
            f = request.files[key]
            if not f.filename:
                continue
            fname = secure_filename(f.filename)
            user_dir = config.USER_FILES_DIR / uid
            user_dir.mkdir(parents=True, exist_ok=True)
            f.save(user_dir / fname)
            uploaded.append(fname)
        if not uploaded:
            return jsonify({"error": "Không có file hợp lệ"}), 400
        return jsonify({"ok": True, "uploaded": uploaded})

    @app.route("/api/files/download/<path:filename>")
    def api_files_download(filename):
        uid = session.get("file_uid")
        if not uid:
            return jsonify({"error": "Chưa xác thực"}), 403
        base = (config.USER_FILES_DIR / uid).resolve()
        fpath = (base / filename).resolve()
        if not str(fpath).startswith(str(base)):
            return jsonify({"error": "Không hợp lệ"}), 400
        if not fpath.exists():
            return jsonify({"error": "File không tồn tại"}), 404
        return send_file(fpath, as_attachment=True)

    @app.route("/api/files/delete/<path:filename>", methods=["POST"])
    def api_files_delete(filename):
        uid = session.get("file_uid")
        if not uid:
            return jsonify({"error": "Chưa xác thực"}), 403
        base = (config.USER_FILES_DIR / uid).resolve()
        fpath = (base / filename).resolve()
        if not str(fpath).startswith(str(base)):
            return jsonify({"error": "Không hợp lệ"}), 400
        if not fpath.exists():
            return jsonify({"error": "File không tồn tại"}), 404
        fpath.unlink()
        return jsonify({"ok": True})

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl", action="store_true",
                        help="Bật HTTPS (cần pyopenssl) — cho phép dùng mic qua IP")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    app = create_app()

    ssl_context = None
    if args.ssl:
        cert_path = Path(__file__).parent / "cert.pem"
        key_path  = Path(__file__).parent / "key.pem"
        if cert_path.exists() and key_path.exists():
            ssl_context = (str(cert_path), str(key_path))
            print(f"✓ HTTPS enabled (persistent cert) — truy cập qua https://<IP>:{args.port}")
        else:
            print("⚠ Chưa có cert.pem / key.pem — dùng adhoc (cert thay đổi mỗi lần restart)")
            print("  Tạo cert cố định: python web/gen_cert.py")
            ssl_context = "adhoc"

    # threaded=False vì model torch không thread-safe theo default
    app.run(host="0.0.0.0", port=args.port, debug=False,
            threaded=False, ssl_context=ssl_context)
