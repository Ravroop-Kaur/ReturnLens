"""
Generates a single self-contained static HTML dashboard from the
pipeline's JSON report, following the plain-English merchant flow:

    UPLOAD DATA -> WE FOUND A PROBLEM -> WHICH ORDERS ARE AT RISK? ->
    WHY? -> HOW MUCH MONEY IS EXPOSED? -> WHAT SHOULD YOU DO? ->
    DID IT WORK?

Technical evidence (precision/recall/F1/confusion matrix/etc.) is
present but placed in a secondary, collapsible section -- never the
headline numbers.

No JS framework, no build step, no external network calls: this is
meant to be opened directly in a browser or embedded in the demo.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CSS = """
#login-gate{position:fixed;inset:0;background:rgba(15,23,32,.97);z-index:9999;display:flex;align-items:center;justify-content:center;padding:20px}
#login-card{width:min(420px,100%);background:#141d29;border-radius:16px;padding:28px;box-shadow:0 20px 60px rgba(0,0,0,.35)}
#login-card input{width:100%;padding:11px 12px;margin:7px 0;border-radius:8px;border:1px solid #26344a;background:#0f1720;color:#e7edf5}
#login-card button{width:100%;padding:11px;margin-top:10px;border:0;border-radius:8px;background:#5b8cff;color:white;font-weight:700;cursor:pointer}
#login-error{color:#ffb454;font-size:13px;min-height:20px;margin-top:8px}
#logout-btn{position:fixed;top:16px;right:16px;z-index:10;padding:8px 12px;border-radius:8px;border:1px solid #26344a;background:#141d29;color:#e7edf5;cursor:pointer}
:root{--bg:#0f1720;--card:#141d29;--accent:#5b8cff;--accent2:#ff6b6b;--text:#e7edf5;--muted:#8ea0b5;--good:#3ecf8e;--warn:#ffb454;}
*{box-sizing:border-box;}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;background:var(--bg);color:var(--text);line-height:1.5;}
.wrap{max-width:900px;margin:0 auto;padding:28px 20px 80px;}
.badge{display:inline-block;background:#26344a;color:var(--muted);font-size:12px;padding:4px 10px;border-radius:999px;letter-spacing:.03em;}
.badge.demo{background:#4a2f1f;color:var(--warn);}
h1{font-size:28px;margin:18px 0 6px;}
h2{font-size:20px;margin:0 0 4px;}
.sub{color:var(--muted);margin:0 0 18px;}
.card{background:var(--card);border-radius:16px;padding:24px;margin-bottom:20px;box-shadow:0 1px 0 rgba(255,255,255,.03) inset;}
.big{font-size:44px;font-weight:700;margin:6px 0;}
.bar-row{display:flex;align-items:center;gap:12px;margin:10px 0;}
.bar-label{width:170px;color:var(--muted);font-size:14px;flex-shrink:0;}
.bar-track{flex:1;background:#20293a;border-radius:8px;height:26px;position:relative;overflow:hidden;}
.bar-fill{height:100%;border-radius:8px;display:flex;align-items:center;padding-left:8px;font-size:13px;font-weight:600;color:#0b1017;}
.bar-fill.accent{background:var(--accent);color:white;}
.bar-fill.affected{background:var(--accent2);color:white;}
.bar-fill.baseline{background:#3d4b63;color:white;}
.risk-grid{display:flex;gap:14px;margin-top:12px;}
.risk-box{flex:1;background:#1a2130;border-radius:12px;padding:16px;text-align:center;}
.risk-box .n{font-size:26px;font-weight:700;}
.risk-box.high .n{color:var(--accent2);}
.risk-box.med .n{color:var(--warn);}
.risk-box.low .n{color:var(--good);}
.explain-list{margin:10px 0 0;padding-left:18px;color:var(--muted);}
.money{font-size:38px;font-weight:700;color:var(--accent);}
.note{color:var(--muted);font-size:13px;margin-top:6px;}
.rec-box{background:#182234;border-left:4px solid var(--accent);padding:16px 18px;border-radius:8px;}
.verify-grid{display:flex;gap:20px;align-items:flex-end;margin-top:10px;}
.verify-col{text-align:center;flex:1;}
.verify-bar{background:#20293a;border-radius:8px 8px 0 0;width:100%;margin:0 auto;}
.verify-col .lab{color:var(--muted);font-size:13px;margin-top:6px;}
.verify-col .val{font-size:20px;font-weight:700;}
.result-tag{display:inline-block;padding:5px 14px;border-radius:999px;font-weight:600;font-size:14px;margin-top:10px;}
.result-tag.improved{background:#123d2b;color:var(--good);}
.result-tag.indeterminate{background:#3a301a;color:var(--warn);}
details{margin-top:8px;}
summary{cursor:pointer;color:var(--accent);font-weight:600;font-size:14px;}
table{width:100%;border-collapse:collapse;margin-top:10px;font-size:13px;}
td,th{padding:6px 8px;border-bottom:1px solid #26344a;text-align:left;color:var(--muted);}
th{color:var(--text);}
.small{font-size:12px;color:var(--muted);}
.check-row{display:flex;align-items:center;gap:8px;margin:6px 0;font-size:14px;}
.check-icon{width:18px;flex-shrink:0;text-align:center;font-weight:700;}
.check-icon.ok{color:var(--good);}
.check-icon.warn{color:var(--warn);}
.check-icon.bad{color:var(--accent2);}
.readiness-status{display:inline-block;padding:5px 14px;border-radius:999px;font-weight:700;font-size:14px;margin-top:8px;}
.readiness-status.ready{background:#123d2b;color:var(--good);}
.readiness-status.partial{background:#3a301a;color:var(--warn);}
.readiness-status.notready{background:#3a1a1f;color:var(--accent2);}
.cm-grid{display:grid;grid-template-columns:120px 1fr 1fr;gap:6px;margin-top:14px;font-size:13px;max-width:520px;}
.cm-grid .cm-head{color:var(--muted);font-weight:600;text-align:center;padding:6px;}
.cm-grid .cm-label{color:var(--muted);display:flex;align-items:center;padding:6px;}
.cm-grid .cm-cell{background:#1a2130;border-radius:8px;padding:14px;text-align:center;font-weight:700;font-size:18px;}
.cm-grid .cm-cell.tp{color:var(--good);}
.cm-grid .cm-cell.fp{color:var(--accent2);}
.cm-grid .cm-cell.fn{color:var(--accent2);}
.cm-grid .cm-cell.tn{color:var(--good);}
.metric-row{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px;}
.metric-box{flex:1;min-width:100px;background:#1a2130;border-radius:10px;padding:12px;text-align:center;}
.metric-box .v{font-size:22px;font-weight:700;}
.metric-box .l{font-size:12px;color:var(--muted);margin-top:2px;}
.drift-tag{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;margin-left:8px;}
.drift-tag.normal{background:#123d2b;color:var(--good);}
.drift-tag.mild{background:#3a301a;color:var(--warn);}
.drift-tag.significant{background:#3a1a1f;color:var(--accent2);}
.source-grid{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px;}
.source-box{flex:1;min-width:150px;background:#1a2130;border-radius:12px;padding:16px;border:2px solid transparent;}
.source-box.active{border-color:var(--good);}
.source-box .name{font-weight:700;font-size:15px;}
.source-box .tag{display:inline-block;margin-top:4px;font-size:11px;color:var(--muted);}
.source-box .action{margin-top:10px;font-size:13px;color:var(--accent);font-weight:600;}
.claim-card{background:#1a2130;border-radius:12px;padding:16px;margin:10px 0;}
.claim-card .claim-id{color:var(--muted);font-size:13px;}
.claim-card .reason{font-weight:700;font-size:16px;margin:2px 0 8px;}
.status-tag{display:inline-block;padding:4px 12px;border-radius:999px;font-weight:700;font-size:13px;}
.status-tag.SUPPORTED{background:#123d2b;color:var(--good);}
.status-tag.NEEDS_REVIEW{background:#3a301a;color:var(--warn);}
.status-tag.SUSPICIOUS{background:#3a1a1f;color:var(--accent2);}
.status-tag.INDETERMINATE{background:#26344a;color:var(--muted);}
.signal-row{display:flex;align-items:flex-start;gap:8px;margin:5px 0;font-size:14px;}
.signal-row.support .dot{color:var(--good);}
.signal-row.contra .dot{color:var(--accent2);}
.signal-row.missing .dot{color:var(--muted);}
.disclaimer{margin-top:12px;padding:10px 12px;background:#26344a;border-radius:8px;font-size:12.5px;color:var(--muted);}
"""


def _bar(label, value, max_value, cls="accent"):
    pct = max(2, min(100, (value / max_value) * 100)) if max_value else 0
    return f"""<div class="bar-row"><div class="bar-label">{label}</div>
<div class="bar-track"><div class="bar-fill {cls}" style="width:{pct:.1f}%">{value:.1%}</div></div></div>"""


def _readiness_card(report: dict) -> str:
    """DATA READINESS section (spec G2): checkmarks per field, plus the
    MODEL STATUS badge. Rendered for both the 'ok' and abstained paths."""
    readiness = report.get("data_readiness")
    if not readiness:
        return ""

    rows = []
    for f in readiness.get("feature_contract", []):
        status = f["status"]
        missingness = f.get("missingness")
        field_label = f["field"].replace("_", " ").title()
        if status in ("REQUIRED", "RECOMMENDED", "OPTIONAL"):
            icon, cls, note = "\u2713", "ok", ""
            if missingness:
                note = f" — {missingness:.0%} missing"
        elif status == "REQUIRED_MISSING":
            icon, cls, note = "\u2715", "bad", " — required, unavailable"
        elif status == "NOT_USABLE":
            icon, cls, note = "\u26a0", "warn", " — present but too sparse/invalid to use"
        else:  # NOT_AVAILABLE (recommended/optional field absent entirely)
            icon, cls, note = "\u26a0", "warn", " — not available"
        rows.append(f'<div class="check-row"><span class="check-icon {cls}">{icon}</span>{field_label}{note}</div>')

    model_status = readiness.get("model_readiness_label", "NOT_READY")
    status_cls = {"FULLY_SUPPORTED": "ready", "PARTIALLY_SUPPORTED": "partial", "NOT_READY": "notready"}.get(model_status, "notready")

    reasons_html = ""
    reasons = readiness.get("reasons_not_ready") or []
    if reasons:
        reasons_html = "<p class=\"note\">Reason: " + " ".join(reasons) + "</p>"

    lifecycle = readiness.get("label_lifecycle", {})
    lifecycle_html = ""
    if lifecycle:
        lifecycle_html = (
            f'<p class="note">Return labels: {lifecycle.get("n_returned",0)} returned, '
            f'{lifecycle.get("n_no_return",0)} confirmed no-return, '
            f'{lifecycle.get("n_pending",0)} still pending (return window not yet closed for those -- '
            f'never counted as "no return").</p>'
        )

    return f"""<div class="card"><span class="badge">DATA READINESS</span>
    <h2>What does this dataset actually support?</h2>
    {''.join(rows)}
    {lifecycle_html}
    <div class="readiness-status {status_cls}">MODEL STATUS: {model_status.replace('_',' ')}</div>
    {reasons_html}
    </div>"""


def render_data_sources(active_connector_type: str | None = None) -> str:
    """DATA SOURCES section (spec G1). Static list of the four supported
    ingestion paths, with whichever one the organization is actually using
    (if any) visually marked active. CSV is explicitly labelled
    fallback/import, never the product, per PART G1 / L."""
    sources = [
        ("razorpay", "Razorpay", "Connect", "Real-time payments/refunds via API + webhooks"),
        ("merchant_api", "Merchant API", "Connect", "Future: generic REST connector"),
        ("database", "Database / Warehouse", "Configure", "Future: read-only DB/warehouse/ERP/OMS connector"),
        ("csv", "CSV", "Import historical data", "Fallback / manual import -- not the product"),
        ("mock", "Synthetic demo data", "n/a", "Bundled demo dataset, for evaluation only"),
    ]
    boxes = []
    for key, name, action, tag in sources:
        active = " active" if key == active_connector_type else ""
        boxes.append(
            f'<div class="source-box{active}"><div class="name">{name}</div>'
            f'<div class="tag">{tag}</div><div class="action">[ {action} ]</div></div>'
        )
    razorpay_flow = """<div class="source-box active" style="flex-basis:100%">
      <div class="name">Real Razorpay Test Mode</div>
      <div class="tag">Connect Test Mode keys → verify connection → import payments/refunds → configure one webhook</div>
      <div class="small" style="margin-top:8px">Real sandbox flow when credentials are supplied. No live money is used. Refunds stay separate from true return labels.</div>
      <div style="margin-top:12px;display:grid;gap:7px;max-width:520px">
        <input id="rzp-key-id" placeholder="Razorpay Test Key ID (rzp_test_...)" style="padding:10px;border-radius:8px;border:1px solid #26344a;background:#0f1720;color:#e7edf5">
        <input id="rzp-key-secret" type="password" placeholder="Test Key Secret" style="padding:10px;border-radius:8px;border:1px solid #26344a;background:#0f1720;color:#e7edf5">
        <input id="rzp-webhook-secret" type="password" placeholder="Webhook Secret (optional for API-only demo)" style="padding:10px;border-radius:8px;border:1px solid #26344a;background:#0f1720;color:#e7edf5">
        <button onclick="connectRazorpay()" style="padding:10px;border:0;border-radius:8px;background:#5b8cff;color:white;font-weight:700;cursor:pointer">Connect Test Mode</button>
        <div id="rzp-status" class="small"></div>
      </div>
    </div>"""
    return f"""<div class="card"><span class="badge">DATA SOURCES</span>
    <h2>Where does your data come from?</h2>
    <div class="source-grid">{''.join(boxes)}</div>
    {razorpay_flow}
    </div>"""


def render_claim_card(evidence: dict) -> str:
    """RETURN CLAIMS / EVIDENCE section (spec G4 / G5). `evidence` is the
    dict produced by src.claims.evidence.EvidenceAggregate.to_dict() (via
    src.pipeline.review_return_claim). Never renders a fraud verdict --
    only the four defined statuses, always with the disclaimer."""
    status = evidence.get("status", "INDETERMINATE")
    reason = evidence.get("normalized_reason") or evidence.get("raw_reason", "")

    def _rows(items, cls, dot):
        return "".join(f'<div class="signal-row {cls}"><span class="dot">{dot}</span>{i}</div>' for i in items)

    signal_html = (
        _rows(evidence.get("supporting_signals", []), "support", "\u2713")
        + _rows(evidence.get("contradictory_signals", []), "contra", "\u26a0")
        + _rows(evidence.get("missing_evidence", []), "missing", "\u2013")
    )

    image_html = ""
    img = evidence.get("image_signal")
    if img:
        image_html = (
            f'<p class="small">Image signal: <b>{img["overall_signal"]}</b>'
            f'{" (demo/mock analyzer)" if img.get("is_demo") else ""} -- {img.get("disclaimer","")}</p>'
        )

    return f"""<div class="claim-card">
    <div class="claim-id">Claim #{evidence.get('claim_id','')}</div>
    <div class="reason">{reason}</div>
    {signal_html}
    {image_html}
    <span class="status-tag {status}">{status.replace('_',' ')}</span>
    <div class="disclaimer">{evidence.get('disclaimer','')}</div>
    </div>"""


def render_claims_section(claims: list) -> str:
    """Wraps one or more render_claim_card() results as the RETURN CLAIMS
    list view (spec G4): a count needing review plus each claim card."""
    if not claims:
        return ""
    needs_review = sum(1 for c in claims if c.get("status") != "SUPPORTED")
    cards = "".join(render_claim_card(c) for c in claims)
    return f"""<div class="card"><span class="badge">RETURN CLAIMS</span>
    <h2>{needs_review} claim{'s' if needs_review != 1 else ''} need review</h2>
    {cards}
    </div>"""


def render(report: dict, claims: list | None = None, show_data_sources: bool = False) -> str:
    readiness_html = _readiness_card(report)
    sources_html = render_data_sources(report.get("connector_type")) if show_data_sources else ""
    claims_html = render_claims_section(claims or [])

    if report.get("status") != "ok":
        message = report.get("model_decision", {}).get("reason") or report.get("ingestion", {}).get("message", "")
        return f"""<!doctype html><html><head><meta charset="utf-8"><title>Return Risk Manager</title>
<style>{CSS}</style></head><body><div class="wrap"><span class="badge demo">{report.get('dataset_label','')}</span>
{sources_html}
<h1>{'Model status: not ready' if report.get('status') == 'abstained' else "Couldn't run analysis"}</h1>
<div class="card"><p>{message}</p><p class="note">[ Connect additional data ]</p></div>
{readiness_html}
{claims_html}
</div>
<div id="login-gate"><div id="login-card"><span class="badge">MERCHANT RISK MANAGER</span><h1 style="font-size:24px">Sign in</h1><p class="sub">Tenant-scoped merchant risk dashboard.</p><input id="login-email" type="email" placeholder="Email" autocomplete="username"><input id="login-password" type="password" placeholder="Password" autocomplete="current-password"><button onclick="login()">Sign in</button><div id="login-error"></div><details><summary>Create demo account</summary><input id="reg-email" type="email" placeholder="Email"><input id="reg-password" type="password" placeholder="Password (8+ characters)"><input id="reg-org" type="text" placeholder="Organization ID"><button onclick="registerDemo()">Create account</button></details></div></div><button id="logout-btn" onclick="logout()" style="display:none">Log out</button>
<script>
const tokenKey='rrm_token'; function err(m){{document.getElementById('login-error').textContent=m||'';}}
async function login(){{const r=await fetch('/auth/login',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{email:document.getElementById('login-email').value,password:document.getElementById('login-password').value}})}});const d=await r.json().catch(()=>({{}}));if(!r.ok){{err(d.error||'Sign-in failed.');return;}}localStorage.setItem(tokenKey,d.token);showApp();}}
async function registerDemo(){{const r=await fetch('/auth/register',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{email:document.getElementById('reg-email').value,password:document.getElementById('reg-password').value,organization_id:document.getElementById('reg-org').value}})}});const d=await r.json().catch(()=>({{}}));if(!r.ok){{err(d.error||'Registration failed.');return;}}document.getElementById('login-email').value=document.getElementById('reg-email').value;document.getElementById('login-password').value=document.getElementById('reg-password').value;err('Account created. Sign in.');}}
async function logout(){{const t=localStorage.getItem(tokenKey);if(t)await fetch('/auth/logout',{{method:'POST',headers:{{Authorization:'Bearer '+t}}}}).catch(()=>{{}});localStorage.removeItem(tokenKey);document.getElementById('logout-btn').style.display='none';document.getElementById('login-gate').style.display='flex';}}
async function connectRazorpay(){{const t=localStorage.getItem(tokenKey);if(!t){{err('Sign in first.');return;}}const status=document.getElementById('rzp-status');status.textContent='Checking Razorpay Test Mode...';const r=await fetch('/integrations/razorpay/test-connection',{{method:'POST',headers:{{'Content-Type':'application/json',Authorization:'Bearer '+t}},body:JSON.stringify({{key_id:document.getElementById('rzp-key-id').value,key_secret:document.getElementById('rzp-key-secret').value,webhook_secret:document.getElementById('rzp-webhook-secret').value}})}});const d=await r.json().catch(()=>({{}}));if(!r.ok){{status.textContent=d.detail||d.error||'Connection failed.';return;}}status.textContent='Connected to Razorpay Test Mode. Import endpoint: '+d.data_source_id+' ('+d.payments_preview_count+' preview payments).';}}async function showApp(){{const t=localStorage.getItem(tokenKey);if(!t)return;const r=await fetch('/risk/latest',{{headers:{{Authorization:'Bearer '+t}}}});if(!r.ok){{localStorage.removeItem(tokenKey);return;}}document.getElementById('login-gate').style.display='none';document.getElementById('logout-btn').style.display='block';}} showApp();
</script></body></html>"""

    tf = report.get("top_finding")
    rec = report.get("recommendation")
    ver = report.get("verification")
    ev = report["test_evaluation"]
    fin = report["financial_exposure"]
    counts = ev["counts"]

    n_high = fin["n_high_risk_orders"]
    n_total = fin["n_total_orders"]
    # crude med/low split for the risk distribution box using proba bands is not
    # available post-hoc without re-scoring; approximate using predicted-positive
    # vs rest, split rest into "watch" vs "low" by a simple 2x heuristic already
    # captured upstream. Here we keep it simple and honest: only high vs rest.
    n_rest = n_total - n_high

    finding_html = ""
    if tf:
        finding_html = f"""
        <div class="card">
          <span class="badge">STEP 1</span>
          <h1>We found a return-risk problem</h1>
          <p class="sub">Orders in <b>{tf['segment']}</b> ({tf['dimension'].replace('_',' ')}) are much more likely to be returned.</p>
          {_bar('Affected segment', tf['segment_return_rate'], max(tf['segment_return_rate'], tf['baseline_return_rate'])*1.15, 'affected')}
          {_bar('Everyone else', tf['baseline_return_rate'], max(tf['segment_return_rate'], tf['baseline_return_rate'])*1.15, 'baseline')}
          <p class="note">That's about {tf['relative_risk']:.1f}x higher, based on {tf['segment_n']} orders in this segment.</p>
        </div>"""
    else:
        finding_html = """<div class="card"><span class="badge">STEP 1</span>
        <h1>No strong, evidence-backed pattern found</h1>
        <p class="sub">Return risk does not appear strongly concentrated in any single segment we checked
        (fulfilment, category, region, shipping service) at a level we're confident isn't just noise.</p></div>"""

    dashboard = f"""
    <div class="card">
      <span class="badge">STEP 2</span>
      <h2>Which orders are most at risk?</h2>
      <div class="risk-grid">
        <div class="risk-box high"><div class="n">{n_high}</div>orders flagged high risk</div>
        <div class="risk-box low"><div class="n">{n_rest}</div>other orders</div>
      </div>
      <p class="note">{fin['pct_orders_high_risk']:.1%} of orders in the evaluated period were flagged high risk
      (threshold selected on a separate validation period, then frozen).</p>
    </div>"""

    # The original benchmark report carries a top-level `feature_importance`
    # list; the newer per-organization pipeline result instead nests ranked
    # findings under `diagnosis[dimension]`. Support both without assuming
    # either key exists, so this section degrades gracefully instead of
    # raising when the other shape is fed in.
    # The original benchmark report carries a top-level `feature_importance`
    # list of {feature, direction}. The newer per-organization pipeline
    # result has no equivalent -- its `diagnosis[dimension]` entries are
    # SegmentFinding dicts (dimension/segment/rates/relative_risk), a
    # different shape describing other segments in the same dimension, not
    # individual model features. Render whichever is actually present
    # rather than assuming one exists.
    feature_bullets = report.get("feature_importance")
    other_segment_bullets = []
    if feature_bullets is None and tf:
        for f in report.get("diagnosis", {}).get(tf["dimension"], [])[:5]:
            if f.get("segment") == tf["segment"]:
                continue
            other_segment_bullets.append(
                f"{f['segment']} ({f['segment_return_rate']:.1%} vs {f['baseline_return_rate']:.1%} baseline)"
            )

    why_html = ""
    if tf:
        if feature_bullets is not None:
            extra_html = f"""<ul class="explain-list">
            {''.join(f"<li>{f['feature'].replace('_',' ')} ({f['direction']})</li>" for f in feature_bullets[:5])}
            </ul>"""
        elif other_segment_bullets:
            extra_html = f"""<p class="small">Other segments checked in this dimension:</p>
            <ul class="explain-list">{''.join(f"<li>{b}</li>" for b in other_segment_bullets)}</ul>"""
        else:
            extra_html = ""
        why_html = f"""<div class="card"><span class="badge">STEP 3</span><h2>Why do we think this is happening?</h2>
        <p>{tf['dimension'].replace('_',' ').title()} <b>{tf['segment']}</b> shows a persistently higher observed
        return rate than the rest of the dataset. This is an observed association, not a proven cause.</p>
        {extra_html}</div>"""

    money_html = f"""<div class="card"><span class="badge">STEP 4</span><h2>How much is exposed?</h2>
    <div class="money">₹{fin['predicted_return_exposure']:,.0f}</div>
    <p class="note">Transaction value associated with orders predicted as high return risk. This is an
    estimate of exposure, not a guaranteed realised loss.</p>
    <p class="note">For comparison, ₹{fin['observed_historical_return_value']:,.0f} of transaction value was
    associated with orders that <i>actually</i> returned in this evaluated period.</p>
    <div class="metric-row">
      <div class="metric-box"><div class="v">₹{fin.get('false_positive_exposure', 0):,.0f}</div><div class="l">False-positive exposure</div></div>
      <div class="metric-box"><div class="v">₹{fin.get('false_negative_exposure', 0):,.0f}</div><div class="l">False-negative exposure</div></div>
    </div>
    <p class="small">These are two independent calculations from the same confusion matrix -- the FALSE
    NEGATIVE COUNT above (missed returns) is not the same number as this FALSE NEGATIVE FINANCIAL EXPOSURE
    (the transaction value of those missed returns).</p>
    </div>"""

    rec_html = ""
    if rec:
        rec_html = f"""<div class="card"><span class="badge">STEP 5</span><h2>What should you do?</h2>
        <div class="rec-box"><b>{rec['text']}</b></div></div>"""

    verify_html = ""
    if ver:
        max_r = max(ver["before_rate"], ver["after_rate"], 0.01) * 1.2
        before_h = int(120 * ver["before_rate"] / max_r)
        after_h = int(120 * ver["after_rate"] / max_r)
        tag_cls = "improved" if ver["improved"] else "indeterminate"
        tag_txt = "IMPROVED" if ver["improved"] else ("NOT IMPROVED" if ver["improved"] is False else "INDETERMINATE")
        verify_html = f"""<div class="card"><span class="badge demo">STEP 6 -- {ver['label']}</span>
        <h2>Did it work?</h2>
        <div class="verify-grid">
          <div class="verify-col"><div class="verify-bar" style="height:{before_h}px;background:#ff6b6b"></div>
            <div class="lab">Before</div><div class="val">{ver['before_rate']:.1%}</div></div>
          <div class="verify-col"><div class="verify-bar" style="height:{after_h}px;background:#3ecf8e"></div>
            <div class="lab">After (simulated)</div><div class="val">{ver['after_rate']:.1%}</div></div>
        </div>
        <span class="result-tag {tag_cls}">{tag_txt}</span>
        <p class="note">This is a simulated intervention outcome for demo purposes -- no real merchant action was taken.</p>
        </div>"""

    def _fmt(v):
        return "n/a" if v != v else f"{v:.3f}" if isinstance(v, float) else v

    # (A9 / the report card must work unchanged for the original benchmark
    # shape -- evaluation/reports/full_report.json, which has
    # model.split.{train_end,val_end,test_end} and model.random_seed -- and
    # for the newer per-organization pipeline result (src.pipeline), which
    # doesn't carry those fields because it isn't a fixed benchmark run.
    # Every lookup below is defensive rather than assuming either shape.)
    split = report["model"].get("split", {})
    period_note = ""
    if all(k in split for k in ("train_end", "val_end", "test_end")):
        period_note = (
            f" (ends {split['train_end'][:10]} / {split['val_end'][:10]} / {split['test_end'][:10]})"
        )
    seed_row = ""
    if report["model"].get("random_seed") is not None:
        seed_row = f"<tr><td>Random seed</td><td>{report['model']['random_seed']}</td></tr>"

    # A9: the confusion matrix / metrics block must be clearly labelled
    # DEMO/SYNTHETIC vs MERCHANT HELD-OUT TEST rather than presented as an
    # unqualified number. is_synthetic_demo is only present on results that
    # went through a real connector; its absence (the original hackathon
    # benchmark report) also means synthetic, since that pipeline only ever
    # ran against data/sample/generic_merchant_orders.csv.
    is_demo_result = report.get("is_synthetic_demo", True)
    eval_label = "DEMO / SYNTHETIC" if is_demo_result else "MERCHANT HELD-OUT TEST"
    eval_label_cls = "demo" if is_demo_result else "good"

    cm_html = f"""
    <span class="badge {'demo' if is_demo_result else ''}" style="{'background:#123d2b;color:var(--good);' if not is_demo_result else ''}">{eval_label}</span>
    <div class="metric-row">
      <div class="metric-box"><div class="v">{_fmt(ev['precision'])}</div><div class="l">Precision</div></div>
      <div class="metric-box"><div class="v">{_fmt(ev['recall'])}</div><div class="l">Recall</div></div>
      <div class="metric-box"><div class="v">{_fmt(ev['f1'])}</div><div class="l">F1</div></div>
      <div class="metric-box"><div class="v">{_fmt(ev['fpr'])}</div><div class="l">FPR</div></div>
      <div class="metric-box"><div class="v">{_fmt(ev['fnr'])}</div><div class="l">FNR</div></div>
    </div>
    <div class="cm-grid">
      <div></div><div class="cm-head">Actual: Return</div><div class="cm-head">Actual: No Return</div>
      <div class="cm-label">Predicted Return</div><div class="cm-cell tp">{counts['tp']}</div><div class="cm-cell fp">{counts['fp']}</div>
      <div class="cm-label">Predicted No Return</div><div class="cm-cell fn">{counts['fn']}</div><div class="cm-cell tn">{counts['tn']}</div>
    </div>
    <ul class="explain-list">
      <li><b>True positives ({counts['tp']}):</b> returns correctly identified by the model.</li>
      <li><b>False positives ({counts['fp']}):</b> orders flagged as likely returns that did not return.</li>
      <li><b>False negatives ({counts['fn']}):</b> returns the model failed to identify -- the most
      important risk for this detector, since a missed return is not flagged for any follow-up at all.</li>
      <li><b>True negatives ({counts['tn']}):</b> orders correctly identified as unlikely to return.</li>
    </ul>
    <p class="small">
      Dataset: <b>{ev.get('dataset_type') or ('synthetic_holdout' if is_demo_result else 'unlabelled')}</b> &middot;
      Test set: <b>{ev['n_total']:,} orders</b> &middot;
      Positive class: <b>{ev.get('positive_class', 'return')}</b> &middot;
      Decision threshold: <b>{report['model']['threshold']}</b>
      {f" &middot; Model version: <b>{ev['model_version']}</b>" if ev.get('model_version') else ""}
    </p>"""

    tech_html = f"""<div class="card"><h2>Technical evidence</h2>
    {cm_html}
    <details><summary>Full model performance detail</summary>
    <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Model</td><td>{report['model']['model_name']}</td></tr>
    <tr><td>Threshold (frozen, chosen on validation)</td><td>{report['model']['threshold']}</td></tr>
    <tr><td>Precision</td><td>{_fmt(ev['precision'])}</td></tr>
    <tr><td>Recall</td><td>{_fmt(ev['recall'])}</td></tr>
    <tr><td>F1</td><td>{_fmt(ev['f1'])}</td></tr>
    <tr><td>False Positive Rate</td><td>{_fmt(ev['fpr'])}</td></tr>
    <tr><td>False Negative Rate</td><td>{_fmt(ev['fnr'])}</td></tr>
    <tr><td>TP / FP / TN / FN</td><td>{counts['tp']} / {counts['fp']} / {counts['tn']} / {counts['fn']}</td></tr>
    <tr><td>Test set size</td><td>{ev['n_total']} orders ({ev['n_positive']} actual returns)</td></tr>
    <tr><td>Train / Val / Test split</td>
        <td>{split.get('train_n', 'n/a')} / {split.get('val_n', 'n/a')} / {split.get('test_n', 'n/a')}{period_note}</td></tr>
    {seed_row}
    </table>
    <p class="small">Threshold and model were selected on the validation period only, then evaluated once,
    unmodified, on the later held-out test period.</p>
    </details></div>"""

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Merchant Return-Risk Manager</title><style>{CSS}</style></head><body>
<div class="wrap">
<span class="badge demo">{report['dataset_label']}</span>
{sources_html}
{readiness_html}
{finding_html}
{dashboard}
{why_html}
{money_html}
{rec_html}
{verify_html}
{tech_html}
{claims_html}
</div>
<div id="login-gate"><div id="login-card"><span class="badge">MERCHANT RISK MANAGER</span><h1 style="font-size:24px">Sign in</h1><p class="sub">Tenant-scoped merchant risk dashboard.</p><input id="login-email" type="email" placeholder="Email" autocomplete="username"><input id="login-password" type="password" placeholder="Password" autocomplete="current-password"><button onclick="login()">Sign in</button><div id="login-error"></div><details><summary>Create demo account</summary><input id="reg-email" type="email" placeholder="Email"><input id="reg-password" type="password" placeholder="Password (8+ characters)"><input id="reg-org" type="text" placeholder="Organization ID"><button onclick="registerDemo()">Create account</button></details></div></div><button id="logout-btn" onclick="logout()" style="display:none">Log out</button><script>const tokenKey='rrm_token';function err(m){{document.getElementById('login-error').textContent=m||'';}}async function login(){{const r=await fetch('/auth/login',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{email:document.getElementById('login-email').value,password:document.getElementById('login-password').value}})}});const d=await r.json().catch(()=>({{}}));if(!r.ok){{err(d.error||'Sign-in failed.');return;}}localStorage.setItem(tokenKey,d.token);showApp();}}async function registerDemo(){{const r=await fetch('/auth/register',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{email:document.getElementById('reg-email').value,password:document.getElementById('reg-password').value,organization_id:document.getElementById('reg-org').value}})}});const d=await r.json().catch(()=>({{}}));if(!r.ok){{err(d.error||'Registration failed.');return;}}document.getElementById('login-email').value=document.getElementById('reg-email').value;document.getElementById('login-password').value=document.getElementById('reg-password').value;err('Account created. Sign in.');}}async function logout(){{const t=localStorage.getItem(tokenKey);if(t)await fetch('/auth/logout',{{method:'POST',headers:{{Authorization:'Bearer '+t}}}}).catch(()=>{{}});localStorage.removeItem(tokenKey);document.getElementById('logout-btn').style.display='none';document.getElementById('login-gate').style.display='flex';}}async function showApp(){{const t=localStorage.getItem(tokenKey);if(!t)return;const r=await fetch('/risk/latest',{{headers:{{Authorization:'Bearer '+t}}}});if(!r.ok){{localStorage.removeItem(tokenKey);return;}}document.getElementById('login-gate').style.display='none';document.getElementById('logout-btn').style.display='block';}}showApp();</script></body></html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default=str(REPO_ROOT / "evaluation/reports/full_report.json"))
    parser.add_argument("--claims", default=None, help="Optional path to a JSON list of claim evidence dicts (spec G4).")
    parser.add_argument("--show-data-sources", action="store_true", help="Render the DATA SOURCES section (spec G1).")
    parser.add_argument("--out", default=str(REPO_ROOT / "app/ui/dashboard.html"))
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text())
    claims = json.loads(Path(args.claims).read_text()) if args.claims else None
    html = render(report, claims=claims, show_data_sources=args.show_data_sources)
    Path(args.out).write_text(html)
    print(f"Dashboard written to {args.out}")


if __name__ == "__main__":
    main()
