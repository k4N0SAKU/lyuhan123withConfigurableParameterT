"""演示场景定义（P7 任务 1；陷阱 1：全部模拟数据并在页面标注）。

两个场景共用底座 = BERT 情感二分类微调模型（满配 acc 99.0%，docs/04 §6）；
业务语义为"情感极性 → 处置结论"的映射示例（UI 如实标注映射关系，不冒充
专用业务模型）。姓名/证件号为虚构模拟字段——真实部署中随文本在 P0 加密前
合入，P1/P2 及线缆全程不可见。
"""
from __future__ import annotations

SIMULATED_NOTE = ("模拟数据：姓名/证件号/内容均为虚构，真实个人信息绝不入系统；"
                  "演示为同机回环模拟（S8），敏感字段仅 P0 本地处理。")

SCENARIOS = {
    "gov": {
        "id": "gov",
        "name": "政务咨询·市民诉求情绪研判",
        "mode_switch_label": "密文流程（模式 B）",
        "fields": [
            {"key": "name", "label": "姓名", "simulated": "张伟"},
            {"key": "id_no", "label": "证件号", "simulated": "110101********1234"},
            {"key": "content", "label": "咨询/投诉内容", "simulated": ""},
        ],
        "samples": [
            "窗口办事效率很高，工作人员态度热情，一次就办好了",
            "系统频繁崩溃，表单提交三次都失败，体验很差",
            "咨询电话一直打不通，问题拖了两周没人处理",
            "材料预审服务贴心，当天就拿到了批复，非常满意",
        ],
        "label_map": {"正面": "诉求已办结·满意（常规归档）",
                      "负面": "不满意·需人工跟进（升级工单）"},
    },
    "enterprise": {
        "id": "enterprise",
        "name": "企业工单·敏感级分类",
        "mode_switch_label": "密文流程（模式 B）",
        "fields": [
            {"key": "name", "label": "报单人", "simulated": "李娜"},
            {"key": "id_no", "label": "工号/证件号", "simulated": "E-2026****0871"},
            {"key": "content", "label": "工单描述", "simulated": ""},
        ],
        "samples": [
            "生产环境接口恢复正常，压力测试全部通过，可以按期上线",
            "客户数据在导出时出错，涉及隐私字段，需要立刻排查",
            "报表数字对不上，怀疑同步任务重复执行，影响结算",
            "文档清晰、部署顺利，按手册半小时完成上线",
        ],
        "label_map": {"正面": "常规工单·低优先级",
                      "负面": "敏感/异常·高优先级（需安全介入）"},
    },
}


def scenario_list() -> list:
    """供 /api/scenarios（去掉默认 content 仿真值的占位）。"""
    out = []
    for s in SCENARIOS.values():
        out.append({
            "id": s["id"], "name": s["name"],
            "fields": [{"key": f["key"], "label": f["label"],
                        "simulated": f["simulated"]} for f in s["fields"]],
            "samples": s["samples"],
            "label_map": s["label_map"],
            "note": SIMULATED_NOTE,
        })
    return out


def build_input(scenario_id: str, sample_index: int, overrides: dict | None = None) -> dict:
    """组装一次演示输入：返回 {text, display_fields, scenario}。

    text = 内容字段（可被 overrides['content'] 覆盖）；display_fields 为
    页面展示用的模拟敏感字段（不进模型——如实标注）。"""
    sc = SCENARIOS[scenario_id]
    ov = overrides or {}
    content = ov.get("content") or sc["samples"][sample_index % len(sc["samples"])]
    display = []
    for f in sc["fields"]:
        val = ov.get(f["key"], f["simulated"] or content)
        display.append({"label": f["label"], "value": val, "key": f["key"]})
    return {"text": content, "display_fields": display, "scenario": sc}
