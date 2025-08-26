from flask import Blueprint, request, jsonify, Response
from services.sessions import RTSession, SessionStore, open_wav_writer
from services.ffmpeg_pipe import FFmpegPipe
from services.transcriber import WhisperService
from config import SESSION_DIR
import os, time, json, logging, wave, contextlib, threading, queue

import struct, math
def _rms_last_half_sec(wav_path: str) -> float:
    with contextlib.closing(wave.open(wav_path,'rb')) as r:
        fr = r.getframerate(); frames = r.getnframes()
        take = int(fr * 0.5); start = max(0, frames - take)
        r.setpos(start); buf = r.readframes(frames - start)
    if not buf: return 0.0
    vals = struct.unpack("<" + "h"*(len(buf)//2), buf)
    return (sum(v*v for v in vals)/max(1,len(vals)))**0.5 / 32768.0


bp = Blueprint("realtime", __name__)
store = SessionStore()
whisper = WhisperService()
log = logging.getLogger("whisper_app")

def _duration_sec(wav_path: str) -> float:
    with contextlib.closing(wave.open(wav_path,'rb')) as r:
        return r.getnframes() / float(r.getframerate())

def _notify_subscribers(sess: RTSession, text: str):
    for qsub in list(sess.sse_subs):
        try: qsub.put_nowait(text)
        except: pass

def _worker(sess: RTSession):
    log.info("[WORKER] start sid=%s", sess.sid)
    while not sess.stop_flag:
        try:
            sess.q.get(timeout=0.5)
        except queue.Empty:
            continue
        if sess.stop_flag: break
        try:
            dur = _duration_sec(sess.wav_path)
            target_until = max(0.0, dur - 2.0)
            if target_until <= sess.last_decoded_sec + 0.5:
                continue

            # 🔑 ここがポイント：VADを切る & しきい値を緩める
            # 日本語メインなら language="ja" で固定すると尚安定
            model = whisper.get_model()
            segments, info = model.transcribe(
                sess.wav_path,
                language=sess.language or "ja",  # ←必要に応じて "ja" 固定
                vad_filter=False,                # ← VAD 無効
                no_speech_threshold=0.3,         # ← 既定より緩め
                temperature=0.0,
                beam_size=5
            )
            segs = list(segments)
            txt = "".join(s.text for s in segs if s.end <= target_until)

            if len(txt) >= len(sess.text_accum):
                sess.text_accum = txt
                sess.last_decoded_sec = target_until
                _notify_subscribers(sess, sess.text_accum)
                log.debug("[WORKER] sid=%s confirmed=%.2fs segs=%d len=%d",
                          sess.sid, sess.last_decoded_sec, len(segs), len(txt))
        except Exception as e:
            log.exception("[WORKER] error sid=%s: %s", sess.sid, e)
    log.info("[WORKER] stop sid=%s", sess.sid)

@bp.post("/rt/start")
def rt_start():
    data = request.get_json(force=True, silent=True) or {}
    language = data.get("language") or None
    sid = os.urandom(16).hex()
    os.makedirs(SESSION_DIR, exist_ok=True)
    wav_path = str(SESSION_DIR / f"{sid}.wav")
    wav = open_wav_writer(wav_path)
    sess = RTSession(sid=sid, language=language, wav_path=wav_path, wav=wav)
    store.add(sess)
    threading.Thread(target=_worker, args=(sess,), daemon=True).start()
    log.info("[START] sid=%s lang=%s", sid, language)
    return jsonify({"sid": sid})

@bp.post("/rt/push")
def rt_push():
    sid = request.form.get("sid")
    mtype = request.form.get("mtype") or ""
    f = request.files.get("chunk")
    if not sid: return "missing sid", 400
    sess = store.get(sid)
    if not sess: return "gone", 410
    if not f: return "no chunk", 400

    data = f.read()
    if not data: return "ok"

    # 初回 push で ffmpeg パイプ起動
    if sess.ff is None:
        def write_pcm(buf: bytes):
            sess.wav.writeframes(buf)
            sess.pcm_frames += len(buf)//2
            try: sess.q.put_nowait(None)
            except: pass
        sess.ff = FFmpegPipe(mtype, write_pcm)
        sess.ff.start()

    ok = sess.ff.write(data)
    if not ok:
        return "gone", 410

    log.debug("[PUSH] sid=%s in_bytes=%d mtype=%s", sid, len(data), mtype)
    return "ok"

@bp.get("/rt/stream")
def rt_stream():
    sid = request.args.get("sid")
    sess = store.get(sid) if sid else None
    if not sess: return "bad sid", 400
    qsub: "queue.Queue[str]" = queue.Queue()
    sess.sse_subs.append(qsub)
    log.info("[STREAM] open sid=%s subs=%d", sid, len(sess.sse_subs))

    def gen():
        try:
            yield f"data: {json.dumps({'text': sess.text_accum}, ensure_ascii=False)}\n\n"
            while not sess.stop_flag:
                try:
                    text = qsub.get(timeout=10.0)
                except queue.Empty:
                    text = sess.text_accum
                yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
        finally:
            try: sess.sse_subs.remove(qsub)
            except ValueError: pass
            log.info("[STREAM] close sid=%s subs=%d", sid, len(sess.sse_subs))
    return Response(gen(), mimetype="text/event-stream", headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@bp.post("/rt/stop")
def rt_stop():
    ...
    try:
        if sess.ff: sess.ff.close()
    except: pass
    try: sess.wav.close()
    except: pass

    # 最終結果（_worker と同じ条件）
    try:
        model = whisper.get_model()
        segments, _info = model.transcribe(
            sess.wav_path,
            language=sess.language or "ja",
            vad_filter=False,
            no_speech_threshold=0.3,
            temperature=0.0,
            beam_size=5
        )
        text = "".join(s.text for s in segments)
        sess.text_accum = text
        _notify_subscribers(sess, text)
    except Exception as e:
        log.exception("[STOP] finalize error sid=%s: %s", sid, e)
