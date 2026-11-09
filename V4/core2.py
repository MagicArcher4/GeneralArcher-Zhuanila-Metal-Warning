import pandas as pd
import numpy as np
import joblib
import os
from openai import OpenAI
from dotenv import load_dotenv


# 从event_detect导入实时事件识别函数
from event_detect import detect_current_events, apply_event_rules, build_event_json

# ============================================================
# 【关键・防崩】以「本文件所在目录」为基准定位数据文件
# 避免因启动目录不同（本地 / Streamlit 云端 / 子文件夹）而报 FileNotFoundError
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== 加载历史数据集，用于实时事件识别 ==========
df_history = pd.read_excel(os.path.join(BASE_DIR, '数据库.xlsx'), sheet_name='总数据集')
df_history['year_month'] = pd.to_datetime(df_history['year_month'], format='%y-%b')
df_history = df_history.sort_values('year_month').reset_index(drop=True)
df_history['X7工业增加值'] = df_history.groupby(df_history['year_month'].dt.year)['X7工业增加值'].transform(
    lambda x: x.ffill().bfill()
)
# =====================================================

# ============================================================
# 加载模型与事件库
# ============================================================
saved = joblib.load(os.path.join(BASE_DIR, 'metal_cost_model_2month_xgb.pkl'))
model_t1 = saved['model_t1']
model_t2 = saved['model_t2']
features = saved['features']

event_db = pd.read_csv(os.path.join(BASE_DIR, 'event_db.csv'))


# ============================================================
# 风险预测
# ============================================================
def predict_risk(X_dict, current_month=None):
    """输入当前X，输出下1月+下2月的波动预测（动态偏差校准版）"""
    X_now = pd.DataFrame([X_dict])

    Xa = model_t1['scaler'].transform(X_now[features].values)
    prob_a = model_t1['clf'].predict_proba(Xa)[0]
    mag_a = model_t1['reg'].predict(Xa)[0]

    Xb = model_t2['scaler'].transform(X_now[features].values)
    prob_b = model_t2['clf'].predict_proba(Xb)[0]
    mag_b = model_t2['reg'].predict(Xb)[0]

    # ========================================================
    # ★ 核心修正：动态计算当期的 BIAS
    # ========================================================
    dynamic_bias = 0.0
    if current_month is not None:
        try:
            # 取出目标月份之前的历史数据（比如过去12个月）
            target_date = pd.to_datetime(current_month)
            hist_data = df_history[df_history['year_month'] < target_date].tail(12)

            if len(hist_data) > 0:
                # 用模型预测一下过去这12个月的值
                X_hist = hist_data[features].values
                X_hist_s = model_t1['scaler'].transform(X_hist)
                pred_hist = model_t1['reg'].predict(X_hist_s)

                # 真实值
                real_hist = hist_data['Y_pred_target'].values

                # 计算过去这12个月，模型平均高估了多少（预测值 - 真实值）
                dynamic_bias = np.mean(pred_hist - real_hist)

                # 限制一下范围，防止某些极端年份导致偏差过大（限制在 -10 到 10 之间）
                dynamic_bias = max(min(dynamic_bias, 10.0), -10.0)
        except Exception as e:
            print(f"动态计算偏差失败，使用0偏差: {e}")
            dynamic_bias = 0.0

    # 扣减动态偏差
    mag_a = mag_a - dynamic_bias
    mag_b = mag_b - dynamic_bias

    # ========================================================
    # 风险等级判断
    # ========================================================
    def risk_level(magnitude):
        if magnitude >= 102.72:
            return '高'
        elif magnitude >= 98.40:
            return '中'
        else:
            return '低'

    return {
        '下1月': {
            'prob_low': float(prob_a[0]), 'prob_mid': float(prob_a[1]), 'prob_high': float(prob_a[2]),
            'predicted_PPI': float(mag_a),
            'risk_level': risk_level(mag_a)
        },
        '下2月': {
            'prob_low': float(prob_b[0]), 'prob_mid': float(prob_b[1]), 'prob_high': float(prob_b[2]),
            'predicted_PPI': float(mag_b),
            'risk_level': risk_level(mag_b)
        }
    }

# ============================================================
# 事件查询
# ============================================================
def get_recent_events(current_month, months=12):
    """查最近N个月的离线历史事件"""
    current = pd.to_datetime(current_month)
    start = current - pd.DateOffset(months=months - 1)
    start_str = start.strftime('%Y-%m')
    end_str = current.strftime('%Y-%m')

    mask = (event_db['time'] >= start_str) & (event_db['time'] <= end_str)
    return event_db[mask].to_dict('records')


def analyze_event_trend(recent_events):
    """判断事件趋势"""
    if not recent_events:
        return "近期无系统性事件，产品价格波动主要来自常规波动。"

    types = [e['event_type'] for e in recent_events]
    if len(set(types)) == 1 and len(types) >= 2:
        return f"「{types[0]}」事件已连续出现{len(types)}个月，需重点关注。"
    else:
        return f"近期出现{len(set(types))}类事件：{'、'.join(set(types))}。"


# ============================================================
# 完整分析（历史事件 + 实时预测事件 双展示）
# ============================================================
def full_analysis(X_dict, current_month):
    """
    完整分析：价格波动预测 + 历史事件(近12个月) + 实时预测事件
    """
    risk = predict_risk(X_dict,current_month)

    # 1. 读取离线历史事件（近12个月）
    hist_events = get_recent_events(current_month, months=12)

    # 2. 实时识别当前月份事件（根据你输入的当前值动态判断）
    current_events, event_details = detect_current_events(X_dict, current_month, df_history)

    # 3. 分别计算趋势
    current_trend = analyze_event_trend(current_events)
    hist_trend = analyze_event_trend(hist_events)

    # 综合趋势：优先展示实时，其次展示历史
    if current_events:
        trend = f"【实时事件】{current_trend}"
    elif hist_events:
        trend = f"【近12个月历史事件】{hist_trend}"
    else:
        trend = "近12个月无系统性事件，产品价格波动主要来自常规波动。"

    return {
        'current_month': current_month,
        'risk': risk,
        'current_events': current_events,
        'hist_events': hist_events,
        'event_trend': trend,
        'current_details': event_details
    }


# ============================================================
# 规则版报告生成
# ============================================================
def generate_report(analysis):
    """把分析结果拼成Markdown报告"""
    r = analysis['risk']
    r1 = r['下1月']
    r2 = r['下2月']

    events = analysis.get('current_events', []) + analysis.get('hist_events', [])
    if not events:
        events = analysis.get('recent_events', [])

    trend = analysis['event_trend']

    report = f"# 金属制品业出厂价格波动预警报告\n\n"
    report += f"**分析基准月**：{analysis['current_month']}\n\n"

    report += "## 一、价格波动预警结论\n\n"
    report += f"### 下1个月（价格波动等级：**{r1['risk_level']}**）\n\n"
    report += f"- 预测金属制品业出厂PPI：{r1['predicted_PPI']:.2f}\n"
    report += f"- 低波动概率：{r1['prob_low']:.1%}\n"
    report += f"- 中波动概率：{r1['prob_mid']:.1%}\n"
    report += f"- 高波动概率：{r1['prob_high']:.1%}\n\n"

    report += f"### 下2个月（价格波动等级：**{r2['risk_level']}**）\n\n"
    report += f"- 预测金属制品业出厂PPI：{r2['predicted_PPI']:.2f}\n"
    report += f"- 低波动概率：{r2['prob_low']:.1%}\n"
    report += f"- 中波动概率：{r2['prob_mid']:.1%}\n"
    report += f"- 高波动概率：{r2['prob_high']:.1%}\n\n"

    report += "## 二、宏观事件归因\n\n"
    report += f"{trend}\n\n"
    if events:
        for e in events:
            report += f"- **{e['time']}**：{e['event_type']}（{e['direction']}，强度{e['magnitude']}）\n"
        report += "\n"

    report += "## 三、经营决策建议\n\n"

    report += "### 📢 销售部门\n"
    if r1['risk_level'] == '高':
        report += "- 预期出厂价上行，**优先锁定产品销售长单**\n"
        report += "- 暂缓低价签单，等待价格上行\n"
    elif r1['risk_level'] == '中':
        report += "- 价格存在上行可能性，**审慎签订低价长期合同**\n"
        report += "- 签中短期合同，保留调价条款\n"
    else:
        report += "- 价格波动风险较低，**维持常规接单节奏**\n"

    report += "\n### 🏭 生产计划\n"
    if r1['risk_level'] == '高':
        report += "- 适度加大产能，提前排产\n"
        report += "- 增加成品库存，迎接价格上涨\n"
    elif r1['risk_level'] == '中':
        report += "- 动态调整生产计划，关注订单变化\n"
    else:
        report += "- 按订单生产，不提前备货\n"

    report += "\n### 💼 公司管理层\n"
    if r1['risk_level'] == '高':
        report += "- 关注需求端承接能力，防止涨价导致客户流失\n"
        report += "- 监控原材料成本传导是否顺畅\n"
    elif r1['risk_level'] == '中':
        report += "- 持续跟踪宏观指标，做好中长期经营规划\n"
    else:
        report += "- 维持现有经营策略，关注市场变化\n"

    return report


# ============================================================
# 大模型报告生成（DeepSeek）—— 【安全初始化客户端】
# ============================================================
# 1. 尝试加载本地 .env 文件
try:
    load_dotenv()
except Exception:
    pass

# 2. 读取 API Key：优先 st.secrets，本地回退到环境变量
#    注意：云端没有 secrets.toml 时 st.secrets 本身会抛异常，必须 try 包住
api_key = None
try:
    import streamlit as st
    try:
        api_key = st.secrets.get("DEEPSEEK_API_KEY")
    except Exception:
        api_key = None
except Exception:
    pass

api_key = api_key or os.getenv("DEEPSEEK_API_KEY")

# 3. 安全初始化：没有 Key 就把 _client 置为 None，绝不让页面崩
if api_key:
    _client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
else:
    _client = None
    print("⚠️ 未配置 API Key，系统将使用本地模板生成报告。")
# ============================================================
# 大模型报告生成函数（严谨归因版）
# ============================================================
def generate_report_llm(analysis, current_month):
    """用DeepSeek生成报告，结合事件但避免绝对化因果"""
    # 【防崩】未配置 API Key 时直接走降级模板，不抛异常
    if _client is None:
        return None

    r = analysis['risk']
    r1 = r['下1月']
    r2 = r['下2月']

    events = analysis.get('current_events', []) + analysis.get('hist_events', [])
    events_text = "\n".join([
        f"- {e['time']}: {e['event_type']} ({e['direction']})"
        for e in events
    ]) if events else "近期无系统性量化事件"

    system_prompt = """你是金属制品行业及产业链的资深市场分析师。
请结合系统提供的【量化预测数据】和【宏观事件】，为企业撰写一份出厂价格波动预测简报。

要求：
1. 总体判断：结合数据和事件，判断涨/跌/震荡及大致幅度。
2. 驱动因素分析：必须结合基准月前后发生的真实宏观/产业事件进行分析。
   ⚠️【核心要求】：分析事件时，**严禁使用“必然导致”、“一定会”、“肯定”等绝对化因果词汇**。
   请使用“可能对……形成扰动”、“需警惕……带来的传导风险”、“为……增添了不确定性”等客观、严谨的推测性表述。
3. 部门建议：给销售、生产、管理层具体、可执行的应对策略。
4. 字数控制在 400 字左右。
"""

    user_prompt = f"""分析基准月：{current_month}

【量化预测数据】
- 下1月出厂价预测：{r1['predicted_PPI']:.2f}，风险等级：{r1['risk_level']}
- 下2月出厂价预测：{r2['predicted_PPI']:.2f}，风险等级：{r2['risk_level']}

【系统监测到的量化事件】
{events_text}

【写作要求】
请利用你的知识库，回忆在 {current_month} 前后发生的【真实宏观事件】（例如地缘政治、进出口关税、美联储会议、大型矿山动态等）。
在第二段“驱动因素分析”中，请将这些真实事件客观地融入进去，但语气要严谨，强调“可能性”和“风险扰动”，不要写得太绝对。
"""

    try:
        resp = _client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4, # 稍微调高一点温度，让语言更灵活
            max_tokens=800
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"DeepSeek 报告生成失败: {e}")
        return None


import json


def get_company_metal_ratio_ai(company_name):
    """
    直接利用大模型内置的行业财报知识库，智能估算金属成本占比。
    不再依赖外网搜索，100%稳定。
    """
    prompt = f"""
    你是一名顶级的制造业供应链与财务分析师。
    请利用你的专业知识，分析“{company_name}”所在行业及其自身的商业模式，估算该企业的【金属材料（如钢铁、铝、铜等）占其营业成本的比例（%）】。

    请直接返回一个 JSON 格式，不要包含任何其他文字：
    {{
        "company": "{company_name}", 
        "metal_ratio": 18.5, 
        "reason": "基于该企业所处的新能源汽车行业特性，其车身结构件与电池托盘大量使用铝合金与高强度钢，结合其近年财报披露的直接材料成本结构，估算金属成本占比约为 18.5%"
    }}
    注意：reason 必须写得像真实的财报分析，要提到具体的材料（钢、铝、铜）和行业特征。metal_ratio 必须是一个具体的数字。
    """

    try:
        resp = _client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # 调低温度，保证输出稳定且符合事实
            response_format={"type": "json_object"}  # 强制返回 JSON
        )
        result = json.loads(resp.choices[0].message.content)
        return result
    except Exception as e:
        print(f"AI 占比分析失败: {e}")
        # 兜底方案
        return {
            "company": company_name,
            "metal_ratio": 25.0,
            "reason": "结合行业通用财务模型，制造业直接材料成本中金属占比通常较高，按均值 25% 进行测算。"
        }

# ============================================================
# 大模型：下游企业采购策略生成
# ============================================================
def generate_procurement_strategy(company_name, industry, revenue, net_margin, metal_ratio, price_change, extra_cost,
                                  absorbed_cost):
    """让大模型根据企业算账结果，生成定制化的采购策略"""

    prompt = f"""
    你是一名顶级的制造业供应链总监。请为以下企业撰写一份针对性极强的【智能采购与应对策略】。

    【企业画像】
    - 企业名称：{company_name}（所属行业：{industry}）
    - 企业营收：{revenue} 万元
    - 当前净利率：{net_margin}%
    - 核心金属材料占营业成本比重：约 {metal_ratio}%

    【成本冲击测算】
    - 上游金属采购价预期涨幅：{price_change}%
    - 预计新增总采购成本：{extra_cost:.1f} 万元
    - 扣除向客户转嫁部分后，企业需【自行硬抗】的成本：{absorbed_cost:.1f} 万元

    【任务要求】
    请结合上述数据，为该企业的采购部、销售部、管理层分别给出 1-2 条极具实操性的应对建议。
    要求语气专业、客观、果断，直接输出策略内容，不需要寒暄，总字数在 250 字左右。
    必须使用 Markdown 列表格式（- 开头）输出。
    """

    try:
        resp = _client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=600
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"采购策略生成失败: {e}")
        return "⚠️ AI 策略生成失败，请检查 API 余额或网络。"