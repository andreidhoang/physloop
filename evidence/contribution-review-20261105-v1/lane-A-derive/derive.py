"""Derivation script for researcher-A report (agent-authored, CPU only).

Parameter counts from the public configs (Alpamayo-1.5-10B config.json, Qwen3-VL-8B config.json)
and the module definitions in transformers v4.57.1 / alpamayo1.5 @36aeb4c. FLOP/byte budgets
are analytical (2 FLOP per MAC); attention FLOPs counted explicitly. No GPU, no weights.
"""
import json, math, sys

a15 = json.load(open(sys.argv[1]))
qv = json.load(open(sys.argv[2]))
tc, vc = qv["text_config"], qv["vision_config"]

V = a15["vocab_size"]
d, L, H, KV, hd, I = tc["hidden_size"], tc["num_hidden_layers"], tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"], tc["intermediate_size"]

# ---- LLM (Qwen3-VL-8B text, vocab resized) ----
llm_layer = d*H*hd + 2*d*KV*hd + H*hd*d + 2*hd + 3*d*I + 2*d
llm_body = L*llm_layer + d
emb = V*d
lm_head = V*d
llm_total = llm_body + emb + lm_head

# ---- Vision tower ----
vd, vI, vdepth, p, tps, m = vc["hidden_size"], vc["intermediate_size"], vc["depth"], vc["patch_size"], vc["temporal_patch_size"], vc["spatial_merge_size"]
out_h = vc["out_hidden_size"]
patch = 3*tps*p*p*vd + vd
pos = vc["num_position_embeddings"]*vd
vblock = 2*(2*vd) + (vd*3*vd + 3*vd) + (vd*vd + vd) + (vd*vI + vI) + (vI*vd + vd)
mh = vd*m*m
merger_main = 2*vd + (mh*mh + mh) + (mh*out_h + out_h)
merger_ds = 2*mh + (mh*mh + mh) + (mh*out_h + out_h)
nds = len(vc["deepstack_visual_indexes"])
vis_total = patch + pos + vdepth*vblock + merger_main + nds*merger_ds

# ---- Expert ----
e = a15["expert_cfg"]
ed, eI, eH, ehd = e["hidden_size"], e["intermediate_size"], e["num_attention_heads"], e["head_dim"]
exp_layer = ed*eH*ehd + 2*ed*KV*ehd + eH*ehd*ed + 2*ehd + 3*ed*eI + 2*ed
exp_total = L*exp_layer + ed
aip = a15["action_in_proj_cfg"]
nin = 3*aip["num_fourier_feats"]; hs = aip["hidden_size"]
act_in = (nin*hs + hs) + (hs) + (hs*hs + hs) + hs + (hs*ed + ed) + 2*ed
act_out = ed*2 + 2
total = llm_total + vis_total + exp_total + act_in + act_out
print(f"LLM body {llm_body/1e9:.4f}B emb {emb/1e9:.4f}B lm_head {lm_head/1e9:.4f}B total {llm_total/1e9:.4f}B")
print(f"Vision tower {vis_total/1e9:.4f}B (blocks {vdepth*vblock/1e9:.4f}B, mergers {(merger_main+nds*merger_ds)/1e9:.4f}B)")
print(f"Expert {exp_total/1e9:.4f}B, action_in {act_in}, action_out {act_out}")
print(f"TOTAL {total:,}  vs safetensors 11,078,526,194  diff {total-11078526194:,}")

# ---- Token counts ----
def img_tokens(h, w):
    return (h//p)*(w//p)//(m*m), (h//p)*(w//p)
for name,(h,w) in {"gym_320x512":(320,512),"alpasim_320x576":(320,576)}.items():
    t, pp = img_tokens(h,w)
    print(name, "patches/img", pp, "merged tokens/img", t, "x16 =", 16*t)

def smart_resize(h, w, factor=32, mn=163840, mx=196608):
    hb, wb = round(h/factor)*factor, round(w/factor)*factor
    if hb*wb > mx:
        beta = math.sqrt(h*w/mx); hb = max(factor, math.floor(h/beta/factor)*factor); wb = max(factor, math.floor(w/beta/factor)*factor)
    elif hb*wb < mn:
        beta = math.sqrt(mn/(h*w)); hb = math.ceil(h*beta/factor)*factor; wb = math.ceil(w*beta/factor)*factor
    return hb, wb
for (h,w) in [(1080,1916),(1080,1920),(320,512)]:
    print("smart_resize", (h,w), "->", smart_resize(h,w))
print("Gym anisotropy (x/y scale) 1916->512, 1080->320:", (512/1916)/(320/1080))
print("AlpaSim anisotropy 1916->576:", (576/1916)/(320/1080))

# ---- FLOPs ----
BW = 1.792e12; BF16 = 503.8e12; FP8 = 1007.6e12
print(f"ridge BF16 {BF16/BW:.1f} FLOP/B ; FP8 {FP8/BW:.1f} FLOP/B")
KVB = L*2*KV*hd*2  # bytes per prefix token, bf16
print("KV bytes/token", KVB)

def vit_flops(npatch):
    lin = vdepth*(2*npatch*(vd*3*vd + vd*vd + 2*vd*vI))
    att = vdepth*(2*2*npatch*npatch*vd)  # QK^T and AV, full (bidirectional) per image
    patchf = 2*npatch*(3*tps*p*p*vd)
    ntok = npatch//(m*m)
    merg = (1+nds)*2*ntok*(mh*mh + mh*out_h)
    return lin+att+patchf+merg

def llm_prefill_flops(Lp):
    lin = L*2*Lp*(d*H*hd + 2*d*KV*hd + H*hd*d + 3*d*I)
    att = L*2*2*Lp*Lp*H*hd/2  # causal ~ half
    head = 2*1*d*V  # logits_to_keep / last token only (generate computes last-token logits)
    return lin+att+head

def expert_step_flops(B, Lp, nq=64):
    lin = L*2*B*nq*(ed*eH*ehd + 2*ed*KV*ehd + eH*ehd*ed + 3*ed*eI)
    att = L*2*2*B*nq*(Lp+nq)*eH*ehd
    return lin+att

def expert_step_bytes(B, Lp, nq=64):
    w = 2*exp_total
    kv = B*Lp*KVB  # read prefix KV per row (after batch_repeat each row has its own copy)
    return w+kv

configs = {
  "Gym rollout (no CoC, 320x512)": dict(np=640, Lp=2730, coc=0, steps=5),
  "AlpaSim e2e driver (CoC, 320x576)": dict(np=720, Lp=3090, coc=24, steps=10),
}
llm_body_bytes = 2*(llm_body + lm_head)  # decode reads all body weights + lm_head
for name,c in configs.items():
    fv = 16*vit_flops(c["np"]); fl = llm_prefill_flops(c["Lp"])
    fe = expert_step_flops(1, c["Lp"]+c["coc"]); be = expert_step_bytes(1, c["Lp"]+c["coc"])
    print(f"\n== {name}")
    print(f"ViT x16: {fv:.3e} FLOP  ({fv/BF16*1e3:.1f} ms @BF16 peak)")
    print(f"LLM prefill L={c['Lp']}: {fl:.3e} FLOP ({fl/BF16*1e3:.1f} ms @peak); weight bytes {2*llm_body/1e9:.2f} GB -> intensity {fl/(2*llm_body):.0f} FLOP/B")
    if c["coc"]:
        per_tok_b = llm_body_bytes + (c["Lp"]+c["coc"]//2)*KVB
        print(f"CoC decode: {per_tok_b/1e9:.2f} GB/token -> {per_tok_b/BW*1e3:.2f} ms/token bound; x{c['coc']} tokens = {c['coc']*per_tok_b/BW*1e3:.0f} ms")
    print(f"Expert step B=1: {fe:.3e} FLOP, {be/1e9:.2f} GB -> intensity {fe/be:.0f} FLOP/B; bound max({fe/BF16*1e3:.2f}, {be/BW*1e3:.2f}) ms; x{c['steps']} steps = {c['steps']*max(fe/BF16, be/BW)*1e3:.1f} ms")
    tot = fv+fl+c["steps"]*fe
    print(f"Policy call FLOP (excl. decode) {tot:.3e} -> {tot/BF16*1e3:.0f} ms @peak; prefill share {(fv+fl)/tot*100:.1f}%")
    # batch crossover for expert step
    for B in (1,2,4,8,16):
        f=expert_step_flops(B,c["Lp"]+c["coc"]); b=expert_step_bytes(B,c["Lp"]+c["coc"])
        print(f"   expert B={B}: {f/b:.0f} FLOP/B, t_bound {max(f/BF16,b/BW)*1e3:.2f} ms ({'compute' if f/BF16>b/BW else 'memory'})")

# ---- Trainer replay per row (Gym, frozen VLM) ----
Lp=2730; T=5
pre = 16*vit_flops(640)+llm_prefill_flops(Lp)
nq = T*64
efwd_lin = L*2*nq*(ed*eH*ehd + 2*ed*KV*ehd + eH*ehd*ed + 3*ed*eI)
efwd_att = L*2*2*nq*(Lp+nq)*eH*ehd
efwd = efwd_lin+efwd_att
ebwd = 2*efwd
adam_bytes = exp_total*(4+4+4+4+4 + 4+4+4 + 2)  # fp32 master r/w, grad r, m r/w, v r/w, bf16 copy (approx)
print(f"\n== Gym trainer replay per row: prefill {pre:.3e} FLOP ({pre/BF16*1e3:.0f} ms @peak); expert fwd {efwd:.3e} + bwd {ebwd:.3e}; prefill share of FLOP {pre/(pre+efwd+ebwd)*100:.1f}%")
print(f"AdamW per optimizer step ~{adam_bytes/1e9:.0f} GB traffic -> {adam_bytes/BW*1e3:.0f} ms at peak BW (mini_batch=1 => one per row)")
print(f"KV per row {Lp*KVB/1e9:.3f} GB; 132 rows -> {132*Lp*KVB/1e9:.1f} GB")
for X_ms in (8, 16):
  for mu in (1,2,4):
    P=pre/BF16/0.6*1e3; E=(efwd+ebwd)/BF16/0.5*1e3; O=adam_bytes/BW/0.8*1e3
    base=mu*(P+E+O); cached=P+mu*(E+O)+(mu-1)*X_ms if mu>1 else base
    print(f"  X={X_ms}ms mu={mu}: P~{P:.0f} E~{E:.0f} O~{O:.0f} ms ; recompute {base:.0f} ms/row vs cached {cached:.0f} ms/row -> x{base/cached:.2f}")

# ---- SDE noise scale (PR18 flow_matching._sde_step_with_logprob) ----
def sde_std(steps, eta):
    ts=[i/steps for i in range(steps+1)]; out=[]
    smax=1-ts[1]
    for i in range(steps):
        t=ts[i]; dt=ts[i+1]-ts[i]; sigma=1-t
        std=math.sqrt(sigma/(1-min(sigma,smax)))*eta
        out.append(std*math.sqrt(dt))
    return out
print("\nSDE per-step std (normalized action units), 5 steps eta=0.2:", [round(s,4) for s in sde_std(5,0.2)])
print("SDE per-step std, 4 steps eta=0.2:", [round(s,4) for s in sde_std(4,0.2)])

# ---- planner delay quantization (DelayBuffer.item_at semantics) ----
def eff_delay(d_us, dt_us):
    if d_us<=0: return 0
    return math.ceil(d_us/dt_us)*dt_us
for d_us in (0,50_000,100_000,150_000,250_000):
    print(f"planner_delay {d_us/1e3:.0f} ms -> effective {eff_delay(d_us,100_000)/1e3:.0f} ms @10 Hz, {eff_delay(d_us,200_000)/1e3:.0f} ms @5 Hz")
