# -*- coding: utf-8 -*-
"""
YYG assistant (templates-based) for Windows + WeChat mini program window.
- Captures the selected window client area (RAM only, no image saved)
- Detects tiles by color-mask candidates + template signature matching
- Computes "clickable" conservatively using overlap + confidence heuristics
- Suggests a next move via small beam-search with strong "don't open new type" bias

Run (recommended):
    py -3.11 yyg_assist_templates_fixed_v2.py

Keys:
    b  reselect BOARD ROI
    s  reselect SLOT ROI
    h  toggle hint
    r  reset ROI file
    q  quit
"""

import os, json, time
import cv2
import numpy as np

# Windows-only
import win32gui, win32con, win32process
import psutil
import mss

# -------------------- paths (always relative to this script) --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROI_FILE = os.path.join(BASE_DIR, "yyg_roi.json")
BANK_FILE = os.path.join(BASE_DIR, "templates", "bank.json")

SLOT_CAP = 7  # YYG slot length

# -------------------- window utils --------------------
def list_wechat_windows():
    """Return list of (hwnd, pid, title, (l,t,r,b)) sorted by area desc."""
    wechat_pids = set()
    try:
        for p in psutil.process_iter(["pid", "name"]):
            try:
                name = (p.info.get("name") or "").lower()
                if "wechat" in name or "weixin" in name:
                    wechat_pids.add(p.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        pass

    wins = []
    def cb(hwnd, _):
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if not title:
                return True
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            is_wechat = (pid in wechat_pids) or ("微信" in title) or ("WeChat" in title) or ("羊了个羊" in title)
            if is_wechat:
                l, t, r, b = win32gui.GetWindowRect(hwnd)
                if (r - l) >= 200 and (b - t) >= 200:
                    wins.append((hwnd, pid, title, (l, t, r, b)))
        except Exception:
            return True
        return True

    try:
        win32gui.EnumWindows(cb, None)
    except Exception:
        pass

    wins.sort(key=lambda x: (x[3][2]-x[3][0])*(x[3][3]-x[3][1]), reverse=True)
    return wins

def get_client_region(hwnd, pad=2):
    """Return (sx,sy,ex,ey) screen coords for window client rect (padded)."""
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    (sx, sy) = win32gui.ClientToScreen(hwnd, (left, top))
    (ex, ey) = win32gui.ClientToScreen(hwnd, (right, bottom))
    return (sx + pad, sy + pad, ex - pad, ey - pad)

# -------------------- ROI persistence --------------------
def load_rois():
    if os.path.exists(ROI_FILE):
        try:
            with open(ROI_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_rois(rois: dict):
    with open(ROI_FILE, "w", encoding="utf-8") as f:
        json.dump(rois, f, ensure_ascii=False, indent=2)

# -------------------- bank / templates --------------------
def load_bank():
    """
    兼容你现有 build_bank.py 生成的 bank.json 格式：
      {
        "thresh": 18,
        "templates": [{"gid": 0, "sig": 123...}, ...]
      }

    返回：(templates, thresh)
      - templates: list[(gid:int, sig:int)]
      - thresh: int（最大汉明距离阈值）
    """
    if not os.path.exists(BANK_FILE):
        raise FileNotFoundError(f"未找到模板库 {BANK_FILE}。请把 templates 文件夹放到脚本同目录下。")
    with open(BANK_FILE, "r", encoding="utf-8") as f:
        bank = json.load(f)

    thresh = int(bank.get("thresh", 18))
    temps = bank.get("templates", [])

    templates = []
    if isinstance(temps, list):
        for t in temps:
            if not isinstance(t, dict):
                continue
            if "gid" not in t:
                continue
            if "sig" in t:
                sig = int(t["sig"])
            elif "tile_sig80" in t:
                sig = int(t["tile_sig80"])
            else:
                continue
            templates.append((int(t["gid"]), sig))

    # 兜底：某些版本可能是 {"tile_sig80":{"0":123,...}}
    if not templates and isinstance(bank.get("tile_sig80"), dict):
        for k, v in bank["tile_sig80"].items():
            templates.append((int(k), int(v)))

    if not templates:
        raise ValueError("bank.json 里没有可用的 templates（需要 gid+sig）。")

    return templates, thresh

# -------------------- drawing helpers --------------------
# -------------------- drawing helpers --------------------
def draw_text(img, text, org, scale=0.7):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2, cv2.LINE_AA)

# -------------------- geometry helpers --------------------
def iou_rect(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / (union + 1e-6)

def nms(rects, scores=None, thr=0.35):
    if not rects:
        return []
    if scores is None:
        scores = [r[2]*r[3] for r in rects]
    idxs = sorted(range(len(rects)), key=lambda i: scores[i], reverse=True)
    keep = []
    while idxs:
        i = idxs.pop(0)
        keep.append(i)
        idxs = [j for j in idxs if iou_rect(rects[i], rects[j]) < thr]
    return keep

def rect_inter_area(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    return iw * ih

# -------------------- signature + matching --------------------
def _bitcount8(u8arr):
    # u8arr: uint8 vector
    return int(np.unpackbits(u8arr).sum())

def phash64(gray_img):
    g = cv2.resize(gray_img, (32, 32), interpolation=cv2.INTER_AREA)
    g = np.float32(g)
    d = cv2.dct(g)
    d8 = d[:8, :8].copy()
    flat = d8.flatten()
    if flat.size > 1:
        med = np.median(flat[1:])  # skip DC
    else:
        med = np.median(flat)
    bits = d8 > med
    h = 0
    for y in range(8):
        for x in range(8):
            h = (h << 1) | int(bits[y, x])
    return int(h)

def color_code16(bgr_img64):
    lab = cv2.cvtColor(bgr_img64, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    l = float(np.mean(L))
    a = float(np.mean(A))
    b = float(np.mean(B))
    l4 = int(np.clip(l / 16.0, 0, 15))     # 4bit
    a6 = int(np.clip(a / 4.0, 0, 63))      # 6bit
    b6 = int(np.clip(b / 4.0, 0, 63))      # 6bit
    return (l4 << 12) | (a6 << 6) | b6

def tile_sig80(tile_bgr):
    tile64 = cv2.resize(tile_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(tile64, cv2.COLOR_BGR2GRAY)
    p = phash64(gray)               # 64bit
    c = color_code16(tile64)        # 16bit
    return (p << 16) | c            # 80bit int

def hamming(a, b):
    return (int(a) ^ int(b)).bit_count()

def _match_gid(sig, templates, thresh):
    best_gid = None
    best_d = 10**9
    for gid, tsig in templates:
        d = hamming(sig, tsig)
        if d < best_d:
            best_d = d
            best_gid = gid
    if best_gid is None or best_d > thresh:
        return None, None
    return best_gid, best_d


# -------------------- candidates by mask / edges --------------------
# -------------------- candidates by mask / edges --------------------
def _mask_candidates(cut_bgr, strict=True):
    """
    Return list of candidate rects (x,y,w,h in cut coords) and base_conf per rect.
    base_conf ~ "whiteness strength" (higher means more likely fully visible tile).
    """
    hsv = cv2.cvtColor(cut_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1].astype(np.uint16)
    V = hsv[:, :, 2].astype(np.uint16)

    # background is green; tiles have white-ish border + bright interior.
    if strict:
        mask = ((S < 90) & (V > 165))
    else:
        mask = ((S < 120) & (V > 140))

    mask = mask.astype(np.uint8) * 255
    k = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects, confs = [], []
    H, W = cut_bgr.shape[:2]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < 20 or h < 20:
            continue
        ar = w / float(h + 1e-6)
        if ar < 0.75 or ar > 1.38:
            continue
        area = cv2.contourArea(c)
        fill = area / float(w*h + 1e-6)
        if fill < (0.48 if strict else 0.42):
            continue
        if w > 0.55*W or h > 0.55*H:
            continue

        # base_conf: mean V in rect * fill
        vmean = float(V[y:y+h, x:x+w].mean()) / 255.0
        bc = vmean * float(fill)
        rects.append((x,y,w,h))
        confs.append(bc)

    return rects, confs

def _edge_candidates(cut_bgr):
    gray = cv2.cvtColor(cut_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray, 60, 140)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects, confs = [], []
    H, W = cut_bgr.shape[:2]
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w < 20 or h < 20:
            continue
        ar = w / float(h + 1e-6)
        if ar < 0.75 or ar > 1.38:
            continue
        if w > 0.55*W or h > 0.55*H:
            continue
        area = cv2.contourArea(c)
        fill = area / float(w*h + 1e-6)
        if fill < 0.20:
            continue
        # base_conf for edges is weaker
        bc = float(fill)
        rects.append((x,y,w,h))
        confs.append(bc)
    return rects, confs

# -------------------- detection in ROI using templates --------------------
def detect_tiles(frame, roi, templates, thresh):
    """
    Return abs_rects, gids, confs in the full frame coordinate system.
    """
    x0, y0, W, H = roi
    cut = frame[y0:y0+H, x0:x0+W]
    if cut.size == 0:
        return [], [], []

    best_rects, best_gids, best_confs = [], [], []

    # strict -> loose -> edge fallback
    for strict in (True, False):
        cand_rects, base_confs = _mask_candidates(cut, strict=strict)
        rects, gids, confs = [], [], []
        for (x,y,w,h), bc in zip(cand_rects, base_confs):
            pad = max(2, int(min(w,h)*0.10))
            inner = cut[y+pad:y+h-pad, x+pad:x+w-pad]
            if inner.size == 0:
                inner = cut[y:y+h, x:x+w]

            sig = tile_sig80(inner)
            gid, d = _match_gid(sig, templates, thresh)
            if gid is None:
                continue
            mconf = 1.0 - (float(d) / float(thresh + 1e-6))
            conf = float(bc) * mconf

            rects.append((x0+x, y0+y, w, h))
            gids.append(int(gid))
            confs.append(float(conf))

        if len(rects) > len(best_rects):
            best_rects, best_gids, best_confs = rects, gids, confs
        if len(best_rects) > 0:
            break

    if len(best_rects) == 0:
        cand_rects, base_confs = _edge_candidates(cut)
        rects, gids, confs = [], [], []
        for (x,y,w,h), bc in zip(cand_rects, base_confs):
            pad = max(2, int(min(w,h)*0.10))
            inner = cut[y+pad:y+h-pad, x+pad:x+w-pad]
            if inner.size == 0:
                inner = cut[y:y+h, x:x+w]
            sig = tile_sig80(inner)
            gid, d = _match_gid(sig, templates, thresh)
            if gid is None:
                continue
            mconf = 1.0 - (float(d) / float(thresh + 1e-6))
            conf = float(bc) * mconf
            rects.append((x0+x, y0+y, w, h))
            gids.append(int(gid))
            confs.append(float(conf))
        best_rects, best_gids, best_confs = rects, gids, confs

    rects, gids, confs = best_rects, best_gids, best_confs

    # NMS by confidence
    if rects:
        keep = nms(rects, scores=confs, thr=0.35)
        rects = [rects[i] for i in keep]
        gids  = [gids[i]  for i in keep]
        confs = [confs[i] for i in keep]

        # median size filter (sync all)
        if len(rects) >= 8:
            ws = np.array([w for (_,_,w,_) in rects], dtype=np.float32)
            hs = np.array([h for (_,_,_,h) in rects], dtype=np.float32)
            mw = float(np.median(ws))
            mh = float(np.median(hs))
            lo, hi = 0.86, 1.14
            keep2 = []
            for i,(x,y,w,h) in enumerate(rects):
                if (lo*mw <= w <= hi*mw) and (lo*mh <= h <= hi*mh):
                    keep2.append(i)
            rects = [rects[i] for i in keep2]
            gids  = [gids[i]  for i in keep2]
            confs = [confs[i] for i in keep2]

    return rects, gids, confs

# -------------------- clickable (conservative, uses overlap + conf) --------------------
def compute_clickable(rects, confs, expand=18, pad=10, need_clear_points=8, conf_margin=0.06):
    """
    Conservative clickable:
    - sample 9 points (center/corners/edge-midpoints) inside an inner box (pad)
    - if a sample point is inside another tile's expanded rect AND that other tile has
      higher confidence (conf_j >= conf_i + conf_margin), treat it as blocked.
    - require at least need_clear_points points to be clear.
    """
    n = len(rects)
    if n == 0:
        return []

    confs = confs if confs is not None and len(confs)==n else [0.0]*n

    def expand_rect(r, e):
        x,y,w,h = r
        return (x-e, y-e, w+2*e, h+2*e)

    exp_rects = [expand_rect(r, expand) for r in rects]

    clickable = [True]*n
    for i,(x,y,w,h) in enumerate(rects):
        px1, py1 = x+pad, y+pad
        px2, py2 = x+w-pad, y+h-pad
        if px2 <= px1 or py2 <= py1:
            px1, py1 = x+2, y+2
            px2, py2 = x+w-2, y+h-2

        cx, cy = (px1+px2)/2.0, (py1+py2)/2.0
        pts = [
            (cx, cy),
            (px1, py1), (px2, py1), (px1, py2), (px2, py2),
            (cx, py1), (cx, py2), (px1, cy), (px2, cy),
        ]

        clear = 0
        for (px,py) in pts:
            blocked = False
            for j in range(n):
                if j==i:
                    continue
                # quick reject by intersection area
                if rect_inter_area(exp_rects[i], exp_rects[j]) <= 0:
                    continue
                ex,ey,ew,eh = exp_rects[j]
                if (ex <= px <= ex+ew) and (ey <= py <= ey+eh):
                    # only treat as blocker if j seems "above" via confidence
                    if confs[j] >= confs[i] + conf_margin:
                        blocked = True
                        break
            if not blocked:
                clear += 1
        clickable[i] = (clear >= need_clear_points)

    return clickable

# -------------------- simple slot utils --------------------
def slot_counts(gids):
    d={}
    for g in gids:
        if g is None or g < 0:
            continue
        d[g] = d.get(g,0)+1
    return d

# -------------------- move selection (beam search with strong no-new-type bias) --------------------
from dataclasses import dataclass
import heapq

@dataclass(frozen=True)
class State:
    slot: tuple  # tuple of gids in slot order (length<=SLOT_CAP)
    board_counts: tuple  # sorted (gid,count) pairs for clickable-only abstraction

def _board_counts_from_gids(gids, clickable):
    d={}
    for g,ok in zip(gids, clickable):
        if not ok or g is None or g < 0:
            continue
        d[g]=d.get(g,0)+1
    return tuple(sorted(d.items()))

def _apply_pick(slot, gid):
    """
    Apply YYG rule: append gid to slot; if any gid appears 3 times -> remove all 3.
    Return new_slot, cleared(0/1), overflow(0/1)
    """
    s=list(slot)
    s.append(gid)
    if len(s) > SLOT_CAP:
        return tuple(s), 0, 1
    # clear triples (can clear multiple types in theory, but in YYG usually one)
    cleared=0
    from collections import Counter
    c=Counter(s)
    for tg, cnt in list(c.items()):
        if cnt>=3:
            cleared += 1
            # remove exactly 3 occurrences
            removed=0
            ns=[]
            for x in s:
                if x==tg and removed<3:
                    removed+=1
                else:
                    ns.append(x)
            s=ns
    return tuple(s), cleared, 0

def _actions_from_board(board_counts):
    # actions are possible gids to pick (clickable types)
    return [gid for gid,count in board_counts for _ in range(count)]

def choose_hint(board_gids, board_clickable, slot_gids, depth=7, beam=70):
    """
    Return preferred gid to pick next (from clickable set), or None.
    This works on gid abstraction (not individual rect), so later we choose a topmost rect of that gid.
    """
    # initial
    bc = _board_counts_from_gids(board_gids, board_clickable)
    if len(bc)==0:
        return None

    slot0 = tuple([g for g in slot_gids if g is not None and g >= 0])

    # precompute counts
    slot_cnt0 = slot_counts(slot0)
    occ0 = len(slot0)
    distinct0 = len(set(slot0))

    # scoring weights (tuned to avoid opening new types)
    W_CLEAR = 1200.0
    W_PAIR  = 220.0
    W_NEW   = -520.0
    W_DISTINCT = -90.0
    W_OCC   = -55.0
    W_DEAD  = -9000.0

    # heuristic gate: if there exists "safe" action (gid already in slot), restrict early
    def allowed_gids(slot_tuple, board_counts):
        acts = [gid for gid,count in board_counts for _ in range(count)]
        if not acts:
            return []
        sc = slot_counts(slot_tuple)
        occ = len(slot_tuple)
        # 1) if any action clears immediately -> only those
        immed = [g for g in acts if sc.get(g,0) >= 2]
        if immed:
            return immed
        # 2) if slot already has at least 2 tiles, prefer matching existing types
        match = [g for g in acts if sc.get(g,0) >= 1]
        if occ >= 2 and match:
            return match
        # 3) if slot nearly full, avoid opening new types unless no alternative
        if occ >= 5:
            safe = [g for g in acts if sc.get(g,0) >= 1]
            if safe:
                return safe
        return acts

    # beam search
    start_state = (slot0, bc)
    # priority queue: store (-score, step, slot, bc)
    pq=[]
    best_first_gid = None
    best_first_score = -1e18

    # we store beams as list of tuples
    beams=[(0.0, start_state, None)]  # (score, (slot,bc), first_gid)

    for step in range(depth):
        new_beams=[]
        for score,(slot, bc), first in beams:
            acts = allowed_gids(slot, bc)
            if not acts:
                continue
            sc = slot_counts(slot)
            occ = len(slot)
            distinct = len(set(slot))
            # prune: if already overflowed, skip
            for gid in set(acts):  # try each type once per depth; counts don't matter much
                # apply pick
                new_slot, cleared, overflow = _apply_pick(slot, gid)
                if overflow:
                    continue

                # update board counts: decrement one
                d = dict(bc)
                if gid not in d or d[gid] <= 0:
                    continue
                d[gid] -= 1
                if d[gid] == 0:
                    del d[gid]
                new_bc = tuple(sorted(d.items()))

                # score components
                new_sc = slot_counts(new_slot)
                new_occ = len(new_slot)
                new_distinct = len(set(new_slot))
                pair_after = sum(1 for v in new_sc.values() if v==2)
                # new type?
                is_new = 1 if sc.get(gid,0)==0 else 0

                s = score
                s += cleared * W_CLEAR
                s += pair_after * W_PAIR
                s += is_new * W_NEW
                s += (new_distinct - distinct0) * W_DISTINCT
                s += (new_occ - occ0) * W_OCC

                # death penalty if slot near full with many distinct
                if new_occ >= 6 and new_distinct >= 6:
                    s += W_DEAD

                first_gid = gid if first is None else first
                new_beams.append((s, (new_slot, new_bc), first_gid))

        if not new_beams:
            break

        # keep top beam states
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam]

    # choose best first action among final beams
    for s,(slot,bc), first_gid in beams:
        if first_gid is None:
            continue
        if s > best_first_score:
            best_first_score = s
            best_first_gid = first_gid

    return best_first_gid

# -------------------- choose rect for chosen gid --------------------
def pick_best_rect_for_gid(rects, gids, confs, gid):
    # choose the most confident, then topmost (smaller y)
    cand = [(i, confs[i], rects[i][1]) for i,g in enumerate(gids) if g==gid]
    if not cand:
        return None
    cand.sort(key=lambda t: (-t[1], t[2]))
    return cand[0][0]

# -------------------- main loop --------------------
def main():
    templates, thresh = load_bank()

    rois = load_rois()  # {"board":[x,y,w,h], "slot":[x,y,w,h]}
    wins = list_wechat_windows()
    if not wins:
        print("未找到微信相关窗口。请先打开微信并进入羊了个羊页面（窗口不要最小化）。")
        return

    print("找到以下窗口（按编号选择）：")
    for i, (hwnd, pid, title, rect) in enumerate(wins):
        l, t, r, b = rect
        print(f"[{i}] pid={pid} hwnd={hwnd} size={r-l}x{b-t} title={title}")

    try:
        idx = int(input("输入编号: ").strip())
    except Exception:
        idx = 0
    idx = max(0, min(idx, len(wins)-1))
    hwnd, pid, title, _ = wins[idx]
    print("已选择:", title)

    if win32gui.IsIconic(hwnd):
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

    show_hint = True
    thresh = 32  # hamming threshold for templates match (bigger -> more matches, but more false)

    cv2.namedWindow("YYG Assist (RAM only)", cv2.WINDOW_NORMAL)

    with mss.mss() as sct:
        while True:
            l, t, r, b = get_client_region(hwnd, pad=2)
            mon = {"left": l, "top": t, "width": r - l, "height": b - t}
            img = np.array(sct.grab(mon))      # BGRA
            frame = img[:, :, :3]              # BGR
            disp = frame.copy()

            # draw ROI
            if "board" in rois:
                x, y, w, h = rois["board"]
                cv2.rectangle(disp, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(disp, "BOARD (press b to reselect)", (x, max(20,y-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            if "slot" in rois:
                x, y, w, h = rois["slot"]
                cv2.rectangle(disp, (x,y), (x+w, y+h), (255,255,0), 2)
                cv2.putText(disp, "SLOT (press s to reselect)", (x, max(20,y-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            board_rects=[]; board_gids=[]; board_confs=[]; board_click=[]
            slot_rects=[]; slot_gids=[]; slot_confs=[]

            if show_hint and ("board" in rois) and ("slot" in rois):
                board_rects, board_gids, board_confs = detect_tiles(frame, rois["board"], templates, thresh)
                slot_rects, slot_gids, slot_confs = detect_tiles(frame, rois["slot"], templates, thresh)

                if board_rects:
                    board_click = compute_clickable(board_rects, board_confs)
                else:
                    board_click = []

                tiles = len(board_rects)
                clickn = int(sum(board_click)) if board_click else 0
                slotn = len([g for g in slot_gids if g is not None and g>=0])

                draw_text(disp, f"hint={'ON' if show_hint else 'OFF'}  tiles={tiles}  clickable={clickn}  slot={slotn}",
                          (10, 55), 0.75)

                if tiles == 0:
                    draw_text(disp, "NO TILES matched (check ROI / thresholds)", (10, 85), 0.65)
                elif clickn == 0:
                    draw_text(disp, "NO CLICKABLE tiles (tighten ROI or increase expand)", (10, 85), 0.65)

                # draw all detected tiles (clickable in dark gray, else light gray)
                for (x,y,w,h), ok in zip(board_rects, board_click):
                    if ok:
                        cv2.rectangle(disp, (x,y), (x+w, y+h), (60,60,60), 2)
                    else:
                        cv2.rectangle(disp, (x,y), (x+w, y+h), (170,170,170), 1)

                # choose hint by gid
                if tiles > 0 and clickn > 0:
                    gid = choose_hint(board_gids, board_click, slot_gids, depth=7, beam=70)
                    if gid is not None:
                        idx_rect = pick_best_rect_for_gid(board_rects, board_gids, board_confs, gid)
                        if idx_rect is not None:
                            x,y,w,h = board_rects[idx_rect]
                            cv2.rectangle(disp, (x,y), (x+w, y+h), (0,0,255), 3)
                            cv2.putText(disp, f"HINT gid={gid}", (x, y+h+22),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            else:
                # status even without hint
                draw_text(disp, f"hint={'ON' if show_hint else 'OFF'}", (10,55), 0.75)

            cv2.imshow("YYG Assist (RAM only)", disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("h"):
                show_hint = not show_hint
            elif key == ord("r"):
                rois = {}
                if os.path.exists(ROI_FILE):
                    try:
                        os.remove(ROI_FILE)
                    except Exception:
                        pass
            elif key == ord("b"):
                sel = cv2.selectROI("Select BOARD ROI", frame, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Select BOARD ROI")
                if sel and sel[2] > 0 and sel[3] > 0:
                    rois["board"] = [int(sel[0]), int(sel[1]), int(sel[2]), int(sel[3])]
                    save_rois(rois)
            elif key == ord("s"):
                sel = cv2.selectROI("Select SLOT ROI", frame, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("Select SLOT ROI")
                if sel and sel[2] > 0 and sel[3] > 0:
                    rois["slot"] = [int(sel[0]), int(sel[1]), int(sel[2]), int(sel[3])]
                    save_rois(rois)

            time.sleep(0.01)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            input("\n发生错误，按回车退出...")
        except Exception:
            pass
