"""Local prompt -> speech demo (standard library HTTP server, no extra dependencies).

Type a sentence and a free-form direction ("whisper it like a secret"); the prompt adapter maps the direction to
a style tag + speaker-z controls, which you can then override per dimension, and the chosen checkpoint renders it.

python -m steer.demo --ckpt m3hyb_40k=runs/mix_m3hyb/step40000.pt --ckpt attn_40k=runs/mix_s1/step40000.pt \
    [--port 7860]      -> http://localhost:7860
"""
import argparse
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf

from .config import SR, CONTROL_DIMS

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SteerTTS demo</title><style>
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#78716c;--card:#fff;--line:#e7e5e4;--acc:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#1c1917;--fg:#f5f5f4;--mut:#a8a29e;--card:#292524;--line:#44403c;--acc:#60a5fa}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0;padding:16px}
main{max-width:900px;margin:auto}h1{font-size:20px;margin:0 0 4px}p.sub{color:var(--mut);margin:0 0 16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:12px}
label{display:block;font-size:13px;color:var(--mut);margin:8px 0 3px}
input[type=text],textarea,select{width:100%;box-sizing:border-box;padding:8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);font:inherit}
.row{display:flex;gap:10px;flex-wrap:wrap}.row>div{flex:1;min-width:180px}
button{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:9px 16px;font:inherit;cursor:pointer;margin-top:10px}
button:disabled{opacity:.5}.chips button{background:transparent;color:var(--acc);border:1px solid var(--line);padding:4px 9px;margin:3px 3px 0 0;font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:4px 16px}
.dim{display:flex;align-items:center;gap:6px;font-size:13px}.dim span.n{width:62px}.dim input[type=range]{flex:1}.dim output{width:36px;text-align:right}
.out{display:flex;flex-direction:column;gap:10px}.res{border-top:1px solid var(--line);padding-top:8px;font-size:13px}
audio{width:100%}code{font-size:12px;color:var(--mut)}
.tabs{display:flex;gap:6px;margin-bottom:12px}.tabs .tab{margin:0;background:transparent;color:var(--fg);border:1px solid var(--line)}
.tabs .tab.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.ab{display:grid;grid-template-columns:minmax(150px,220px) 1fr 52px;gap:4px 10px;align-items:center;font-size:13px}
.ab .u{text-align:right;color:var(--mut)}@media(max-width:560px){.ab{grid-template-columns:1fr 52px}.ab .nm{grid-column:1/-1}}
</style></head><body><main>
<h1>SteerTTS demo</h1><p class="sub">Free-form direction &rarr; prompt adapter &rarr; style tag + speaker-relative controls (z) &rarr; synthesis.</p>
<div class="tabs"><button class="tab on" data-t="gen">Generate</button><button class="tab" data-t="ab">Decoder A/B (flow matching)</button></div>
<section id="ab" hidden><div class="card"><b>Same sentence, same voice, same durations and pitch: only the mel decoder differs.</b>
<p class="sub" style="margin:6px 0 0">"base L1 decoder" is the current model (predicts the average spectrum, hazy). The flow-matching decoder samples
the fine detail on top of it, shown at several training steps. "first try" = 7.5k steps at temperature 0.8 (too noisy); the others use 0.4.
UTMOS (predicted MOS, 1-5) beside each clip; it under-rates breathy / whispered speech and real-recording texture, so trust your ears.</p></div>
<div id="abl"></div></section>
<section id="gen"><div class="card">
<label>Presets (text + direction)</label><div class="chips" id="presets"></div>
<label>Text <small>- inline markup: <code>*emphasis*</code> <code>[pause 500ms]</code> <code>&lt;slow&gt; &lt;fast&gt; &lt;high&gt; &lt;low&gt; &lt;loud&gt; &lt;soft&gt;</code> (per word), <code>&lt;whisper&gt;..&lt;/whisper&gt;</code> / any style tag, <code>&lt;style prompt="angry"&gt;..&lt;/style&gt;</code> (span), sound events <code>[knock] [dog] [door] [phone] [bell] [applause]</code>, in-voice <code>[laugh] [cough] [sigh]</code>; the prompt can also set a scene: "in a busy cafe", "over the phone", "in a huge hall", "then a dog barks"</small></label><textarea id="text" rows="3">The meeting has been moved to Thursday afternoon at three.</textarea>
<div class="chips" id="tchips"></div>
<label>Direction (prompt)</label><input type="text" id="prompt" value="whisper it like a secret">
<div class="chips" id="chips"></div>
<div class="row"><div><label>Model</label><select id="model"></select></div>
<div><label>Voice</label><select id="voice"></select></div>
<div><label>Pace (1 = model's choice)</label><input type="text" id="pace" value="1.0"></div></div>
<label><input type="checkbox" id="both"> render with every model (side by side)</label>
<label><input type="checkbox" id="stream" checked> stream (play each sentence as soon as it is rendered; first model only)</label>
<details><summary style="margin-top:10px;cursor:pointer">Manual control overrides (applied on top of the prompt)</summary>
<label>Tag override</label><select id="tag"></select>
<div class="grid" id="dims"></div></details>
<button id="go">Generate</button>
</div>
<div class="out" id="out"></div></section>
</main><script>
const EX=["speak sadly, as if something bad just happened","you are furious and shouting at your neighbour","whisper it like a secret",
"a calm, slow meditation guide","excited, talking fast and loud","in a deep, low voice, very slowly","nervous, with a shaky voice","cheerfully, like you just got great news",
"bored and monotone, reading a list","a news anchor, clear and steady","laughing, can't believe it","high pitched and questioning","soft and tender, saying goodnight",
"disgusted, like something smells awful","quietly, with long pauses between words","sarcastic","dramatic movie trailer voice",""];
const TX=[["neutral","The meeting has been moved to Thursday afternoon at three."],
["question","Did you really leave the front door open all night?"],
["exclamation","We won! I can't believe we actually won the whole thing!"],
["numbers","Your order number is 4 7 2 9, and the total comes to thirty two dollars and fifty cents."],
["long","When the rain finally stopped, the children ran outside, splashing through every puddle they could find, while their parents watched from the porch."],
["short","No."],["dialogue","\"Wait,\" she said. \"Are you sure this is the right address?\""],
["names","Doctor Patel will see Ms. Johnson at Saint Mary's Hospital on Monday."],
["instructions","Turn left at the second light and park behind the building."],
["bad news","I'm sorry, but the flight has been cancelled and there are no seats until tomorrow."],
["markup: emphasis","I did *not* say she stole the money. I said she *borrowed* it."],
["markup: pauses","And the winner is... [pause 800ms] the blue team! [pause 300ms] Congratulations."],
["markup: whisper span","I have to tell you something. <whisper>The surprise party is on Friday.</whisper> Don't tell anyone!"],
["markup: style spans","<style prompt=\"calm and gentle\">Everything is fine.</style> <style prompt=\"you are furious and shouting\">Who ate my sandwich?</style>"],
["sound events","Did you hear that? [knock] [pause 400ms] Someone is at the door. [dog]"],
["laugh + cough (in-voice)","[laugh] Okay, that was funny. [cough] Sorry, where was I?"],
["markup: per-word prosody","It costs <high>three</high> dollars, <low>not thirty</low>. Please speak <slow>very, very slowly</slow> now."],
["long, streamed","Welcome back. Today we have three updates. First, the new office opens on Monday. Second, parking moves to the north lot. And third, lunch is on us this Friday. See you there!"]];
const PRE=[["Secret","I hid the key under the third flowerpot by the door.","whisper it like a secret"],
["Cafe chat","So I finally told him the truth, and he just laughed.","talking calmly in a busy cafe"],
["Phone call","Hi, it's me. I'm running about ten minutes late, sorry.","nervous, on a phone call"],
["Stadium","Ladies and gentlemen, please welcome your champions!","a cheerful announcement echoing through a huge hall"],
["Storm","We need to get inside before it gets any worse.","worried, during a thunderstorm"],
["Dog at the end","Fine, I'll take him for a walk after dinner.","annoyed, then a dog barks"],
["Furious","You promised me this would be done yesterday, and it still isn't!","you are furious and shouting at your neighbour"],
["Grief","He was my best friend, and I never got to say goodbye.","speak sadly, as if something bad just happened"],
["Good news","They called this morning, and I got the job!","cheerfully, like you just got great news"],
["Meditation","Breathe in slowly. Hold it. And now, gently let it go.","a calm, slow meditation guide"],
["Hype","Tonight only, every ticket is half price, so get here now!","excited, talking fast and loud"],
["Nervous","Um, I think I might have broken the printer again.","nervous, with a shaky voice"],
["Narrator","Long ago, in a village by the sea, there lived an old clockmaker.","in a deep, low voice, very slowly"],
["Announcer","The train to Boston is now boarding on platform four.","a news anchor, clear and steady"],
["Bored","Item one, paper. Item two, staples. Item three, more paper.","bored and monotone, reading a list"],
["Disbelief","You're telling me the cat opened the fridge by itself?","high pitched and questioning"],
["Goodnight","Sleep well, little one. I'll be right here in the morning.","soft and tender, saying goodnight"]];
let META;
fetch('meta').then(r=>r.json()).then(m=>{META=m;
 for(const k of m.models){const o=new Option(k,k);document.getElementById('model').add(o)}
 for(const v of m.voices){document.getElementById('voice').add(new Option(v,v))}
 const t=document.getElementById('tag');t.add(new Option('(from prompt)',''));for(const g of m.tags)t.add(new Option(g,g));
 const d=document.getElementById('dims');
 for(const k of m.dims){d.insertAdjacentHTML('beforeend',`<div class="dim"><input type="checkbox" id="on_${k}"><span class="n">${k}</span><input type="range" id="z_${k}" min="-4" max="4" step="0.25" value="0" oninput="this.nextElementSibling.value=this.value;document.getElementById('on_${k}').checked=true"><output>0</output></div>`)}
});
function chip(box,label,title,fn){const b=document.createElement('button');b.textContent=label;b.title=title;b.onclick=fn;document.getElementById(box).appendChild(b)}
const $=id=>document.getElementById(id);
let abDone=false;
for(const b of document.querySelectorAll('.tab'))b.onclick=()=>{
 for(const x of document.querySelectorAll('.tab'))x.classList.toggle('on',x===b);
 $('gen').hidden=b.dataset.t!=='gen';$('ab').hidden=b.dataset.t!=='ab';if(b.dataset.t==='ab'&&!abDone)loadAB()};
async function loadAB(){
 const r=await fetch('ab/manifest.json');if(!r.ok){$('abl').textContent='no A/B set rendered yet (python -m steer.flow ab ...)';return}
 abDone=true;const man=await r.json();const esc=t=>t.replace(/&/g,'&amp;').replace(/</g,'&lt;');
 for(const c of man){const req=[c.tag!=='none'?c.tag:'',...Object.entries(c.ctrl).map(([k,v])=>k+' '+(v>0?'+':'')+v)].filter(Boolean).join(', ')||'plain';
  let h=`<div class="card"><div><b>${esc(c.text)}</b></div><code>${c.voice} &middot; ${esc(req)}</code><div class="ab" style="margin-top:8px">`;
  for(const a of c.audio)h+=`<span class="nm">${esc(a.name)}</span><audio controls preload="none" src="ab/${a.file}"></audio><span class="u">${a.utmos==null?'':a.utmos.toFixed(2)}</span>`;
  $('abl').insertAdjacentHTML('beforeend',h+'</div></div>')}}
for(const e of EX)chip('chips',e||'(no direction)',e,()=>{$('prompt').value=e});
for(const [k,t] of TX)chip('tchips',k,t,()=>{$('text').value=t});
for(const [k,t,e] of PRE)chip('presets',k,t+'  |  '+e,()=>{$('text').value=t;$('prompt').value=e});
let AC=null;
async function streamRender(body){
 AC=AC||new (window.AudioContext||window.webkitAudioContext)();await AC.resume();
 const t0=performance.now();let first=null,next=AC.currentTime+0.05,n=0,info=[];
 const box=document.createElement('div');box.className='card';document.getElementById('out').prepend(box);
 box.innerHTML=`<div><b>${body.prompt||'(no prompt)'}</b> &middot; ${body.voice} &middot; streaming</div><code id=st>waiting for first chunk…</code><div class=res id=segs></div>`;
 const r=await fetch('stream',{method:'POST',body:JSON.stringify(body)});const rd=r.body.getReader();const dec=new TextDecoder();let buf='';
 for(;;){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});let i;
  while((i=buf.indexOf('\n'))>=0){const line=buf.slice(0,i);buf=buf.slice(i+1);if(!line.trim())continue;const m=JSON.parse(line);
   if(m.error){box.querySelector('#st').textContent='error: '+m.error;return}
   if(m.done){box.querySelector('#st').innerHTML=`${n} chunks &middot; first audio after <b>${first.toFixed(0)} ms</b> &middot; total ${(performance.now()-t0).toFixed(0)} ms for ${m.seconds.toFixed(1)} s audio`;
     box.insertAdjacentHTML('beforeend',`<audio controls src="data:audio/wav;base64,${m.wav}"></audio>`);continue}
   const ab=await AC.decodeAudioData(Uint8Array.from(atob(m.wav),c=>c.charCodeAt(0)).buffer);
   const src=AC.createBufferSource();src.buffer=ab;src.connect(AC.destination);next=Math.max(next,AC.currentTime);src.start(next);next+=ab.duration;
   if(first===null){first=performance.now()-t0;box.querySelector('#st').textContent=`first audio after ${first.toFixed(0)} ms…`}
   n++;box.querySelector('#segs').insertAdjacentHTML('beforeend',`<div><code>${m.event?('[sound: '+m.event+']'):m.pause_ms?('[pause '+m.pause_ms+' ms]'):('['+m.tag+(Object.keys(m.ctrl).length?' '+JSON.stringify(m.ctrl):'')+'] '+m.text.replace(/</g,'&lt;'))} &middot; ${m.ms} ms</code></div>`)}}
}
document.getElementById('go').onclick=async()=>{
 const btn=document.getElementById('go');btn.disabled=true;btn.textContent='Generating…';
 const ctrl={};for(const k of META.dims){if(document.getElementById('on_'+k).checked)ctrl[k]=parseFloat(document.getElementById('z_'+k).value)}
 const models=document.getElementById('both').checked?META.models:[document.getElementById('model').value];
 const body={text:document.getElementById('text').value,prompt:document.getElementById('prompt').value,voice:document.getElementById('voice').value,
  pace:parseFloat(document.getElementById('pace').value)||1,tag:document.getElementById('tag').value,ctrl,models};
 if(document.getElementById('stream').checked){try{await streamRender(body)}catch(e){alert(e)}finally{btn.disabled=false;btn.textContent='Generate'};return}
 try{const r=await (await fetch('synth',{method:'POST',body:JSON.stringify(body)})).json();
  if(r.error)throw new Error(r.error);
  const box=document.createElement('div');box.className='card';
  const m=r.mapping;box.innerHTML=`<div><b>${body.prompt||'(no prompt)'}</b> &middot; ${body.voice}</div><code>tags ${JSON.stringify(m.top_tags)} &rarr; used <b>${m.tag}</b> | ctrl ${JSON.stringify(m.ctrl)}${m.env?' | env '+JSON.stringify(m.env)+' '+JSON.stringify(m.events||{}):''}</code>`;
  for(const a of r.audio){box.insertAdjacentHTML('beforeend',`<div class="res">${a.model} <code>${a.seconds.toFixed(2)} s audio, ${a.ms} ms</code><audio controls src="data:audio/wav;base64,${a.wav}"></audio></div>`)}
  document.getElementById('out').prepend(box);box.querySelector('audio').play().catch(()=>{});
 }catch(e){alert(e)}finally{btn.disabled=false;btn.textContent='Generate'}};
</script></body></html>"""


class App:
    def __init__(self, ckpts, device="cuda"):
        from .synth import Synth
        from .prompt import PromptMapper
        from .markup import MarkupSynth
        def make(spec):  # "base.pt" or "base.pt,flow=flow.pt" (flow-matching decoder on top, steer.flow)
            base, _, fl = spec.partition(",flow=")
            return Synth(base, device=device, flow=fl or None)
        self.models = {n: make(p) for n, p in ckpts.items()}
        from .config import META
        # acoustic environment: scenes / rooms / phone / environmental events mixed from real recordings
        # (steer.mixer) whenever the noise library exists; the environment-aware prompt adapter parses them
        mixer = None
        if (META / "noise_lib.jsonl").exists() and (META / "prompt_adapter_env.pt").exists():
            from .mixer import EnvMixer
            mixer = EnvMixer()
        self.pm = PromptMapper(device=device, env=mixer is not None)
        self.ms = {n: MarkupSynth(s, self.pm, mixer) for n, s in self.models.items()}
        s = next(iter(self.models.values()))
        # the list shows public names only (steer.voices); expressive EARS voices first, in public-name order
        from .voices import assign
        r2p = assign(s.stats["speakers"])
        ears = sorted((r2p[v] for v in s.stats["speakers"] if v.startswith("ears")))
        self.voices = ears + sorted(r2p[v] for v in s.stats["speakers"] if not v.startswith("ears"))
        self.tags = [t for t in s.stats["tags"] if t != "none"]
        self.lock = threading.Lock()

    def mapping(self, req, s=None):
        """Default style for unmarked text: prompt -> best tag the checkpoint knows + controls; manual overrides."""
        mapping = dict(top_tags=[], tag="none", ctrl={})
        if req.get("prompt", "").strip():
            r = self.pm(req["prompt"])
            known = [t for t, _ in r["top_tags"] if s is None or t in s.tag2i]
            mapping = dict(top_tags=r["top_tags"], tag=(known or ["none"])[0], ctrl=r["ctrl"])
            if "env" in r:  # shown in the UI's mapping line: what the prompt asked of the acoustic environment
                mapping.update(env=r["env"], events={k: v for k, v in r["events"].items() if v})
        if req.get("tag"):
            mapping["tag"] = req["tag"]
        mapping["ctrl"] = {**mapping["ctrl"], **{k: float(v) for k, v in req.get("ctrl", {}).items() if k in CONTROL_DIMS}}
        return mapping

    def stream(self, req, write):
        """NDJSON: one line per rendered segment (sentence / pause / style span), then a done line with the full wav."""
        name = (req.get("models") or list(self.models))[0]
        s, ms = self.models[name], self.ms[name]
        m = self.mapping(req, s)
        parts = []
        with self.lock:
            for c in ms.stream(req["text"], voice=req["voice"], tag=m["tag"] if m["tag"] in s.tag2i else "none",
                               ctrl=m["ctrl"] or None, pace=float(req.get("pace", 1.0))):
                parts.append(c.audio)
                buf = io.BytesIO(); sf.write(buf, c.audio, SR, format="WAV", subtype="PCM_16")
                write(dict(wav=base64.b64encode(buf.getvalue()).decode(), ms=int(c.synth_ms), text=c.segment.text,
                           tag=c.segment.tag, ctrl=c.segment.ctrl, pause_ms=c.segment.pause_ms,
                           event=c.segment.event))
        y = np.concatenate(parts) if parts else np.zeros(1, np.float32)
        buf = io.BytesIO(); sf.write(buf, y, SR, format="WAV", subtype="PCM_16")
        write(dict(done=True, model=name, seconds=len(y) / SR, wav=base64.b64encode(buf.getvalue()).decode(), mapping=m))

    def synth(self, req):
        import time
        mapping = self.mapping(req)
        audio = []
        for name in req.get("models") or list(self.models)[:1]:
            s = self.models[name]
            t = time.time()
            m = self.mapping(req, s)
            with self.lock:  # markup-aware render (plain text renders exactly as before: one segment per sentence)
                wav = self.ms[name].render(req["text"], voice=req["voice"], tag=m["tag"] if m["tag"] in s.tag2i else "none",
                                           ctrl=m["ctrl"] or None, pace=float(req.get("pace", 1.0)))
            buf = io.BytesIO(); sf.write(buf, wav, SR, format="WAV", subtype="PCM_16")
            audio.append(dict(model=name, wav=base64.b64encode(buf.getvalue()).decode(), seconds=len(wav) / SR,
                              ms=int((time.time() - t) * 1000)))
        return dict(mapping=mapping, audio=audio)


def serve(app, port):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if "/ab/" in self.path:  # the decoder A/B listening set (steer.flow ab)
                from .config import ROOT
                base = (ROOT / "runs" / "_flow_ab").resolve()
                f = (base / self.path.split("/ab/", 1)[1].split("?")[0]).resolve()
                if base in f.parents and f.is_file() and f.suffix in (".wav", ".json"):
                    self._send(200, f.read_bytes(), "audio/wav" if f.suffix == ".wav" else "application/json")
                else:
                    self._send(404, b"not found", "text/plain")
                return
            if self.path.rstrip("/").endswith("meta"):
                self._send(200, json.dumps(dict(models=list(app.models), voices=app.voices, tags=app.tags,
                                                dims=CONTROL_DIMS)).encode(), "application/json")
            else:
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")

        def do_POST(self):
            if self.path.rstrip("/").endswith("stream"):
                self.send_response(200); self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Cache-Control", "no-cache"); self.end_headers()  # HTTP/1.0: body ends at close

                def write(obj):
                    self.wfile.write((json.dumps(obj) + "\n").encode()); self.wfile.flush()
                try:
                    app.stream(json.loads(self.rfile.read(int(self.headers["Content-Length"]))), write)
                except Exception as e:  # noqa: BLE001
                    write(dict(error=f"{type(e).__name__}: {e}"))
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                out = app.synth(req)
            except Exception as e:  # noqa: BLE001
                out = dict(error=f"{type(e).__name__}: {e}")
            self._send(200, json.dumps(out).encode(), "application/json")

        def log_message(self, fmt, *args):
            pass

    print(f"SteerTTS demo on http://localhost:{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path[,flow=flow.pt], repeatable; first is the default")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--device", default="cuda", help="cpu keeps the GPU free for training")
    a = ap.parse_args()
    serve(App(dict(c.split("=", 1) for c in a.ckpt), a.device), a.port)


if __name__ == "__main__":
    main()
