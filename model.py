"""PTD-Net two view model (image + 48 ABCD descriptors, cross attention fusion) for the research demo.
Same architecture and preprocessing as the v13 paper notebook."""
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
from features import hand_feats, FR, NF

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255
def mlp(i, h, o, p=.2): return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, o), nn.ReLU())

class Net(nn.Module):
    def __init__(s, d=128):
        super().__init__()
        s.bb = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0); D = s.bb.num_features
        s.pd = nn.Sequential(nn.Linear(D, d), nn.ReLU(), nn.Dropout(.3)); s.hd = nn.Linear(d, 2)
        s.pa = mlp(NF, d, d); s.ha = nn.Linear(d, 2)
        s.tk = nn.Sequential(nn.Linear(D, d), nn.LayerNorm(d)); s.q = nn.Linear(d, d)
        s.att = nn.MultiheadAttention(d, 4, dropout=.1, batch_first=True); s.ao = nn.LayerNorm(d)
        s.br = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 512), nn.GELU(), nn.Dropout(.1), nn.Linear(512, D)); s.use_br = False
        s.cp = nn.Linear(d, NF); s.cls1 = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Dropout(.3)); s.cls2 = nn.Linear(d, 2)
    def forward(s, x, f):
        fm = s.bb.forward_features(x); tok = fm.flatten(2).transpose(1, 2); h = s.bb.forward_head(fm, pre_logits=True)
        if s.use_br: h = h + s.br(h)
        zi, za = s.pd(h), s.pa(f)
        a, _ = s.att(s.q(za).unsqueeze(1), s.tk(tok), s.tk(tok)); zt = s.ao(a.squeeze(1))
        return s.cls2(s.cls1(torch.cat([zi, za, zt], 1)))

class Predictor:
    def __init__(s, path='app_model_v13.pt'):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        s.net = Net(); s.net.load_state_dict(ck['state_dict']); s.net.use_br = bool(ck.get('use_br', False)); s.net.eval()
        s.mu, s.sd, s.thr, s.img, s.p = ck['feat_mu'], ck['feat_sd'], float(ck['threshold']), int(ck['img']), ck['sog_p']
    def _square(s, rgb):
        h, w = rgb.shape[:2]; m = min(h, w); return rgb[(h - m) // 2:(h - m) // 2 + m, (w - m) // 2:(w - m) // 2 + m]
    def _tensor(s, img):
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float()
        n = x.clamp(min=1).pow(s.p).mean((2, 3), keepdim=True).pow(1 / s.p); x = (x * (n.mean(1, keepdim=True) / n)).clamp(0, 255)  # Shades of Gray
        return (x - MEAN) / STD
    def predict(s, rgb):
        sq = s._square(rgb)
        img = cv2.resize(sq, (s.img, s.img), interpolation=cv2.INTER_AREA)
        f = np.nan_to_num(hand_feats(cv2.resize(sq, (FR, FR), interpolation=cv2.INTER_AREA)))
        f = torch.from_numpy(np.clip((f - s.mu) / s.sd, -5, 5).astype(np.float32))[None]; x = s._tensor(img)
        # probability: mean logits over identity, horizontal and vertical flip (as in the paper); Grad-CAM on the last conv block
        acts = {}; hk = s.net.bb.bn2.register_forward_hook(lambda m, i, o: acts.__setitem__('a', o))
        cams, lgs = [], []
        for fl in (None, 3, 2):
            xi = x if fl is None else x.flip(fl)
            lg = s.net(xi, f); sc = lg[0, 1] - lg[0, 0]; g = torch.autograd.grad(sc, acts['a'])[0]
            cam = F.relu((g.mean((2, 3), keepdim=True) * acts['a']).sum(1))[0].detach().numpy()
            cams.append(cam if fl is None else (cam[:, ::-1] if fl == 3 else cam[::-1, :])); lgs.append(sc.item())
        hk.remove()
        cam = cv2.GaussianBlur(cv2.resize(np.mean(cams, 0), (s.img, s.img), interpolation=cv2.INTER_CUBIC), (0, 0), s.img / 16)
        cam = np.clip(cam, 0, None); cam = cam / (cam.max() + 1e-8); p = float(1 / (1 + np.exp(-np.mean(lgs))))
        return dict(p=p, malignant=p >= s.thr, thr=s.thr, image=img, cam=cam)
