# -*- coding: utf-8 -*-
"""报告书插图生成（matplotlib，SimSun 字体，200dpi PNG）。
产出 4 图：图2-1 系统架构 / 图3-1 D5′协议流程 / 图2-2 端到端推理流程 / 图5-1 通信优化对照。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
OUT = r"D:/safetensors/密码技术竞赛/报告书/figs"
os.makedirs(OUT, exist_ok=True)
INK, ACC, SUB = "#0B1220", "#2563EB", "#475569"

def box(ax, x, y, w, h, text, fc="#EFF6FF", ec=ACC, fs=11, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                 fc=fc, ec=ec, lw=1.4))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=INK, fontweight="bold" if bold else "normal")

def arrow(ax, x1, y1, x2, y2, text="", color=SUB, style="-|>", ls="-", tx=0, ty=0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=14, color=color, lw=1.3, linestyle=ls))
    if text:
        ax.text((x1+x2)/2 + tx, (y1+y2)/2 + ty, text, ha="center", va="center",
                fontsize=9, color=color)

# ---------- 图 2-1 系统架构 ----------
fig, ax = plt.subplots(figsize=(9.5, 5.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
ax.text(5, 6.0, "系统架构与信任边界（m=3 部署示例）", ha="center", fontsize=13, fontweight="bold", color=INK)
box(ax, 4.1, 5.15, 1.8, 0.62, "离线 CA\n（证书/密钥分发）", fc="#F1F5F9", ec=SUB, fs=9.5)
box(ax, 0.35, 3.3, 2.5, 1.35, "P0 客户端（数据属主·信任根）\n持 CKKS 私钥（白名单门控）\n本地加密 / 白名单解密\n明文不出本地", fc="#DBEAFE", bold=False)
box(ax, 3.85, 3.3, 2.3, 1.35, "P1 锚点（计算方1）\n入口份额 y−r1\n出口掩码片 s1\n不持任何私钥", fc="#ECFDF5")
box(ax, 6.85, 3.3, 2.3, 1.35, "P2 合成方（计算方2）\nCKKS 线性层\nMPC 非线性 / 出口合成\n不持任何私钥", fc="#ECFDF5")
box(ax, 3.85, 1.35, 2.3, 1.05, "P3..Pm 计算方\n（m 可扩展至 15）\n掩码片 / 份额持有", fc="#F8FAFC", ec=SUB)
box(ax, 0.35, 0.7, 2.5, 0.95, "敌手模型\n半诚实计算参与方 t=m−1\n恶意网络（线缆全加密）", fc="#FEF2F2", ec="#DC2626", fs=9.5)
for x in (1.6, 5.0, 8.0):
    arrow(ax, 5.0, 5.1, x, 4.7, color=SUB)
arrow(ax, 1.6, 3.3, 1.6, 2.4, color=SUB); arrow(ax, 1.6, 2.4, 3.85, 2.4, color=SUB)
arrow(ax, 3.85, 2.4, 1.6, 2.4, color=SUB, ls="--")
arrow(ax, 6.15, 3.3, 6.15, 2.4, color=SUB)
arrow(ax, 5.0, 2.4, 5.0, 1.35, color=SUB); arrow(ax, 5.0, 1.35, 5.0, 0.7, color=SUB, ls="--")
ax.text(5.0, 2.62, "SM2DH 会话密钥 + SM4-GCM 帧 + ratchet（链路双向）", ha="center", fontsize=8.5, color=SUB)
ax.text(2.9, 1.0, "白名单①②\n解密触点", ha="center", fontsize=8.5, color="#1D4ED8")
ax.text(8.35, 1.95, "掩码链\n出口片", ha="center", fontsize=8.5, color=SUB)
plt.tight_layout(); plt.savefig(OUT + "/fig_2_1_arch.png", dpi=200); plt.close()

# ---------- 图 3-1 D5′ 协议流程 ----------
fig, ax = plt.subplots(figsize=(9.5, 6.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 7.0); ax.axis("off")
ax.text(5, 6.8, "D5′ 域转换协议（入口掩码链 + 出口星型合成）", ha="center", fontsize=13, fontweight="bold", color=INK)
lanes = {"P0": 0.6, "P1": 3.0, "P2": 5.4, "P3": 7.6}
for name, x in lanes.items():
    ax.add_patch(FancyBboxPatch((x-0.55, 6.0), 1.1, 0.5, boxstyle="round,pad=0.02", fc="#1E293B", ec=INK))
    ax.text(x, 6.25, name, ha="center", va="center", fontsize=11, color="white", fontweight="bold")
    ax.plot([x, x], [0.7, 6.0], color="#CBD5E1", lw=1, ls="--")
def msg(y, x1, x2, text, color=ACC):
    arrow(ax, x1, y, x2, y, text=text, color=color, tx=0, ty=0.14)
msg(5.55, lanes["P2"], lanes["P3"], "① Enc(r2)")
msg(5.05, lanes["P3"], lanes["P1"], "② Enc(r2+r3)")
msg(4.55, lanes["P1"], lanes["P0"], "③ 掩码链末跳 Enc(r2+r3+r1)")
msg(4.05, lanes["P0"], lanes["P0"], "④ 白名单①解密 y=x+Σr（OTP）")
ax.add_patch(FancyBboxPatch((lanes["P0"]-0.75, 3.55), 1.5, 0.42, boxstyle="round,pad=0.02", fc="#DBEAFE", ec=ACC))
ax.text(lanes["P0"], 3.76, "⑤ y 分发锚点 P1", ha="center", va="center", fontsize=9.5, color=INK)
arrow(ax, lanes["P0"], 3.5, lanes["P1"], 3.5, "⑥ 分发 y", color=SUB)
msg(3.0, lanes["P2"], lanes["P1"], "⑦ Enc(s2)→离线库存取")
msg(2.5, lanes["P3"], lanes["P2"], "⑧ z3=a3+s3, Enc(s3)")
ax.add_patch(FancyBboxPatch((lanes["P2"]-0.95, 1.45), 1.9, 0.85, boxstyle="round,pad=0.02", fc="#FEF9C3", ec="#CA8A04"))
ax.text(lanes["P2"], 1.87, "⑨ 出口合成\nw=Σz=v+Σs\nEnc(v)=Enc(w)−ΣEnc(si)", ha="center", va="center", fontsize=9, color=INK)
msg(1.0, lanes["P2"], lanes["P0"], "⑩ Enc(v) fresh → 白名单②解密结果", color="#059669")
ax.text(0.15, 0.35, "安全语义：任意 2 方合谋至少缺一片均匀掩码 ri 或 si ⇒ 一次一密隐藏（t=m−1）",
        fontsize=10, color=INK)
plt.tight_layout(); plt.savefig(OUT + "/fig_3_1_d5.png", dpi=200); plt.close()

# ---------- 图 2-2 端到端推理流程 ----------
fig, ax = plt.subplots(figsize=(9.0, 8.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 11.6); ax.axis("off")
ax.text(5, 11.35, "端到端密文推理流程（一次分类请求，12 层循环）", ha="center", fontsize=13, fontweight="bold", color=INK)
steps = [
    ("① 会话建立：证书认证 → SM2DH → SM4-GCM 通道 + ratchet", "#1E293B", "white"),
    ("② P0：分词 → 嵌入 → 定点化 Q16 → CKKS 加密", "#DBEAFE", INK),
    ("③ P0→P2：INFER_REQUEST（GCM 帧）", "#F8FAFC", INK),
    ("④ P2：CKKS 线性层 QKV/FFN（对角线打包+BSGS）", "#ECFDF5", INK),
    ("⑤ P2→P3→P1：掩码链逐跳叠加 Enc(ri)", "#FFFBEB", INK),
    ("⑥ P0：白名单①解密 y=x+Σr（OTP 域）", "#DBEAFE", INK),
    ("⑦ P0→P1 分发 y；各方形成 m-of-m 分享", "#F8FAFC", INK),
    ("⑧ MPC 非线性：GELU/Softmax（Beaver，分享域）", "#ECFDF5", INK),
    ("⑨ 出口星型合成 Enc(v) fresh → 残差+LN → 下一层", "#ECFDF5", INK),
    ("⑩ P2→P0：INFER_RESULT（SM2 签名）", "#F8FAFC", INK),
    ("⑪ P0：白名单②解密 → 分类标签", "#DCFCE7", INK),
]
y = 10.6
for i, (t, fc, tc) in enumerate(steps):
    box(ax, 0.7, y, 8.6, 0.62, t, fc=fc, ec=ACC if i else "#1E293B", fs=10)
    if i < len(steps) - 1:
        arrow(ax, 5.0, y - 0.02, 5.0, y - 0.4, color=SUB)
    y -= 0.92
ax.add_patch(FancyBboxPatch((0.4, 0.18), 9.2, 0.55, boxstyle="round,pad=0.02",
             fc="#F1F5F9", ec=SUB))
ax.text(5.0, 0.45, "循环标注：④~⑨ 共执行 12 层，每层 10 次域转换；全程 SM3 审计链留痕",
        ha="center", va="center", fontsize=9.5, color=SUB)
plt.tight_layout(); plt.savefig(OUT + "/fig_2_2_flow.png", dpi=200); plt.close()

# ---------- 图 5-1 通信优化对照 ----------
fig, ax = plt.subplots(figsize=(8.6, 4.4))
ms = ["m=2", "m=3", "m=4", "m=5"]
base = [6.65, 9.29, 11.91, 14.55]
opt = [4.14, 5.52, 6.90, 8.28]
x = range(4)
w = 0.36
b1 = ax.bar([i - w/2 for i in x], base, w, label="优化前", color="#94A3B8")
b2 = ax.bar([i + w/2 for i in x], opt, w, label="库存+二进制帧优化后", color="#2563EB")
for i, (bv, ov) in enumerate(zip(base, opt)):
    ax.text(i - w/2, bv + 0.15, f"{bv:.2f}", ha="center", fontsize=9, color=SUB)
    ax.text(i + w/2, ov + 0.15, f"{ov:.2f}", ha="center", fontsize=9, color=ACC)
    ax.text(i, max(bv, ov) + 0.75, f"−{(1-ov/bv)*100:.0f}%", ha="center", fontsize=10,
            color="#DC2626", fontweight="bold")
ax.set_xticks(list(x)); ax.set_xticklabels(ms)
ax.set_ylabel("每转换往返通信（MiB）", fontsize=10)
ax.set_xlabel("计算方数量 m（mode-b 全槽，3 轮均值）", fontsize=10)
ax.set_ylim(0, max(base) * 1.22)
ax.legend(fontsize=9.5, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT + "/fig_5_1_comm.png", dpi=200); plt.close()
print("figures ->", OUT, os.listdir(OUT))
