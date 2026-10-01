#!/usr/bin/env bash
# =============================================================================
# A1-22 复现脚本（P6：可复现实验包）
# 从零跑通：装环境 → 明文基线 → 密态推理（生命周期 e2e）→ 攻击测试
#
# 用法（Git Bash / Linux）：
#   bash scripts/reproduce.sh                 # 全流程（干净目录一次通过）
#   A122_SKIP_ENV=1 bash scripts/reproduce.sh # 复用当前 .venv（调试用）
#   A122_MODELS_DIR=/abs/path/data/models bash scripts/reproduce.sh
#                                             # 复用本地模型目录（免下载；
#                                             # 模型是数据不是代码，环境洁净
#                                             # 性不受影响——docs/06 §2）
#
# 预期总耗时：全新 venv ~15-25 min（含 pip 缓存命中的依赖安装与模型下载）；
# 模型已就位时 ~6-10 min。逐步预期输出见 docs/06-复现手册.md。
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
LOG_DIR="$REPO_ROOT/reproduce_logs"
mkdir -p "$LOG_DIR"
STEP=0
ok() { STEP=$((STEP+1)); echo "[STEP $STEP][OK] $1"; }
fail() { echo "[FAIL] $1"; echo "复现失败——日志见 $LOG_DIR"; exit 1; }

echo "=== A1-22 复现开始 $(date -u +%FT%TZ) ==="
echo "repo: $REPO_ROOT"

# ---- STEP 1: Python 环境 ----------------------------------------------------
VENV_DIR="${REPO_ROOT}/.venv-repro"
if [ "${A122_SKIP_ENV:-0}" = "1" ]; then
  PY="$(command -v python)"
  ok "复用现有解释器（A122_SKIP_ENV=1）: $PY"
else
  command -v python >/dev/null 2>&1 || fail "未找到 python（需 3.10+）"
  python -m venv "$VENV_DIR" 2>&1 | tee "$LOG_DIR/1_venv.log" || fail "venv 创建失败"
  # shellcheck disable=SC1091
  source "$VENV_DIR/Scripts/activate" 2>/dev/null || source "$VENV_DIR/bin/activate"
  PY="$(command -v python)"
  ok "venv 创建: $VENV_DIR"
fi

# ---- STEP 2: 依赖安装 --------------------------------------------------------
if [ "${A122_SKIP_ENV:-0}" != "1" ]; then
  echo "pip install（约 3-10 分钟，取决于缓存）..."
  "$PY" -m pip install --quiet --upgrade pip 2>&1 | tail -1
  "$PY" -m pip install --quiet -r requirements.txt 2>&1 | tee "$LOG_DIR/2_pip.log" \
    || fail "依赖安装失败（requirements.txt 锁定版）"
  "$PY" -c "import torch, tenseal, transformers, gmssl, pytest" \
    || fail "依赖导入自检失败"
  ok "依赖安装 + 导入自检"
else
  ok "跳过依赖安装（A122_SKIP_ENV=1）"
fi

# ---- STEP 3: 模型三件套 ------------------------------------------------------
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"   # 国内镜像
if [ -n "${A122_MODELS_DIR:-}" ]; then
  ok "复用本地模型目录: $A122_MODELS_DIR（含情感微调检查点——数据复用，见 docs/06 §2）"
else
  echo "下载模型三件套（GPT-2 / BERT-base-chinese / 情感微调检查点）..."
  "$PY" -m benchmarks.download_models --models gpt2 bert-base-chinese 2>&1 | tee "$LOG_DIR/3_models.log" \
    || fail "模型下载失败（检查网络/镜像）"
  ok "模型三件套就绪"
fi

# ---- STEP 4: 明文基线（快速口径）---------------------------------------------
"$PY" -m benchmarks.run_full_bench --quick --parts plaintext \
  2>&1 | tee "$LOG_DIR/4_plaintext.log" || fail "明文基线失败"
"$PY" -c "
import json
d = json.load(open('benchmarks/results/p6_full_bench.json', encoding='utf-8'))
pt = d['plaintext']
assert 'bert_latency_full12' in pt and 'gpt2_latency' in pt, '明文基线缺项'
print('BERT 明文 P50 = %.1f ms；GPT-2 明文 P50 = %.1f ms' % (
    pt['bert_latency_full12']['round_total']['p50'],
    pt['gpt2_latency']['round_total']['p50']))
" || fail "明文基线结果校验失败"
ok "明文基线（冒烟口径，1 轮/任务）"

# ---- STEP 5: 密态推理（三方生命周期 e2e，玩具 CKKS 参数）----------------------
DEMO_DIR="$(mktemp -d /tmp/a122_repro_XXXX)"
"$PY" -m src.nodes.provision --dir "$DEMO_DIR" --toy 2>&1 | tee "$LOG_DIR/5_provision.log" \
  || fail "离线供给失败"
"$PY" -m src.nodes.orchestrator --prov-dir "$DEMO_DIR" --toy \
  --out report_repro.json 2>&1 | tee "$LOG_DIR/5_lifecycle.log" \
  || fail "密态推理生命周期 e2e 失败"
"$PY" -c "
import json
r = json.load(open('report_repro.json', encoding='utf-8'))
assert r['audit_chain_verified'] == {'p0_client': True, 'p1_keynode': True, 'p2_infernode': True}
assert r['phases']['inference']['max_abs_err'] < 1e-5
print('生命周期 e2e：审计链三节点核验通过；密态往返 max_abs_err =',
      r['phases']['inference']['max_abs_err'])
" || fail "生命周期 e2e 结果校验失败"
ok "密态推理（认证→会话→GCM 推理往返→ratchet→销毁→审计核验）"

# ---- STEP 6: 攻击测试套件 -----------------------------------------------------
"$PY" -m pytest tests/attack -q 2>&1 | tee "$LOG_DIR/6_attacks.log" \
  || fail "攻击测试套件失败"
ok "攻击测试（默认口径含合谋双模式——窃听/中间人/篡改/重放/合谋，38 项）"

# ---- STEP 7: 结论 -------------------------------------------------------------
cat <<BANNER
===============================================================
复现成功 $(date -u +%FT%TZ)——共 $STEP 步全部通过
- 明文基线：benchmarks/results/p6_full_bench.json :: plaintext
- 密态推理：report_repro.json（audit_chain_verified 全 True）
- 攻击判定：benchmarks/results/attack_verdicts.json
扩展：python -m benchmarks.run_full_bench --parts cipher_matrix \\
  --cipher-rounds 20 --segmented   # 密文矩阵（--segmented 必带：单进程形态
  在 32GB 机确定性 OOM；总时长 ~4h + 最长轮 33min）
      python -m pytest tests/attack -o addopts=  # 攻击全套（含模式 B 合谋）
逐步日志：$LOG_DIR
===============================================================
BANNER
