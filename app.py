import os, json, cv2, numpy as np, streamlit as st
HERE = os.path.dirname(os.path.abspath(__file__))
from PIL import Image, ImageOps
from model import Predictor

st.set_page_config(page_title='DermShift: skin lesion check', page_icon='🩺', layout='wide')
st.markdown("""<style>
.block-container{max-width:1100px;padding-top:2rem}
.card{border:1px solid #e3e6ea;border-radius:14px;padding:18px 20px;background:#fff}
.big{font-size:44px;font-weight:700;line-height:1}
.pill{display:inline-block;padding:4px 12px;border-radius:999px;color:#fff;font-weight:600;font-size:14px}
.bar{height:10px;border-radius:6px;background:linear-gradient(90deg,#1b7a3a,#e0a800 55%,#b03a2e);position:relative;margin:14px 0 4px}
.mark{position:absolute;top:-5px;width:4px;height:20px;background:#111;border-radius:2px}
.small{color:#5f6b7a;font-size:13px}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load(): return Predictor(os.path.join(HERE, 'app_model_v13.pt'))
P = load()

st.title('DermShift')
st.caption('Research demo of a skin-lesion classifier adapted from dermoscopy to smartphone photographs. Not a medical device.')

left, right = st.columns([1, 1.25], gap='large')
with left:
    src = st.radio('Image', ['Upload a photo', 'Use camera'], horizontal=True, label_visibility='collapsed')
    f = st.file_uploader('Close-up photo of one lesion', type=['jpg', 'jpeg', 'png']) if src == 'Upload a photo' else st.camera_input('Take a close-up photo')
    with st.expander('How to take a good photo'):
        st.markdown('- Hold the phone 10–15 cm away, lesion in the **centre** of the frame\n- Daylight, **no flash**, no filters\n- Keep it in focus; one lesion per photo\n- Avoid pen marks, hair and shadows where possible')
    st.markdown('<p class="small">Photos are processed in memory only and are not stored.</p>', unsafe_allow_html=True)

with right:
    if f is None:
        st.info('Add a photo to see the result.')
    else:
        rgb = np.array(ImageOps.exif_transpose(Image.open(f)).convert('RGB'))
        with st.spinner('Analysing…'): r = P.predict(rgb)
        p, thr = r['p'], r['thr']
        ptxt = '&gt;99' if p > .99 else ('&lt;1' if p < .01 else f'{p*100:.0f}')
        lvl, col = ('Low', '#1b7a3a') if p < thr else (('Moderate', '#e0a800') if p < .7 else ('High', '#b03a2e'))
        st.markdown(f"""<div class="card"><div class="small">Estimated probability of a malignant lesion (MEL, BCC, SCC)</div>
        <div style="display:flex;align-items:center;gap:16px;margin-top:6px"><span class="big">{ptxt}%</span><span class="pill" style="background:{col}">{lvl} concern</span></div>
        <div class="bar"><div class="mark" style="left:calc({p*100:.1f}% - 2px)"></div></div>
        <div class="small" style="position:relative;height:18px"><span style="position:absolute;left:0">0%</span><span style="position:absolute;left:{thr*100:.1f}%;transform:translateX(-50%)">▲ threshold {thr*100:.0f}%</span><span style="position:absolute;right:0">100%</span></div></div>""", unsafe_allow_html=True)
        heat = cv2.applyColorMap((r['cam'] * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
        over = (0.6 * r['image'] + 0.4 * heat).astype(np.uint8)
        a, b = st.columns(2); a.image(r['image'], caption='Input (centre crop, 224×224)', use_container_width=True)
        b.image(over, caption='Where the image view looked (Grad-CAM)', use_container_width=True)
        if lvl == 'Low': st.success('Low concern from this photo. Keep an eye on it and see a doctor if it changes in size, shape or colour, bleeds or itches.')
        else: st.warning('Please have this lesion checked by a dermatologist. This tool cannot diagnose skin cancer.')

st.divider()
RP = os.path.join(HERE, 'app_results.json')
R_ = json.load(open(RP)) if os.path.exists(RP) else {}
auc_ = f"Held-out PAD-UFES-20 AUC {R_['auc']:.2f}; " if 'auc' in R_ else ''
st.markdown('<p class="small"><b>Model.</b> PTD-Net with EfficientNet-B0: a two view network (image and 48 ABCD descriptors fused by cross attention), trained on HAM10000 and BCN20000 dermoscopy and adapted without target labels to PAD-UFES-20 smartphone photographs. '
            + auc_ + 'accuracy is lower on darker skin types and unfamiliar cameras. For research and education only; it does not replace a clinical examination.</p>', unsafe_allow_html=True)
