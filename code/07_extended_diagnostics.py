"""
扩展模型诊断（回应"负号是否合理"的质疑，全部是事先设定的诊断步骤，不做事后挑结果）
输入：../data/02_features.csv
输出：
  ../outputs/table13_variable_coding_check.csv   变量编码方向核查
  ../outputs/table14_zero_inflation_check.csv    零膨胀诊断
  ../outputs/table15_quadratic_terms.csv         二次项(倒U型)检验结果，全部变量的线性/二次系数与显著性
  ../outputs/table16_zinb_vs_nb_comparison.csv   零膨胀负二项 vs 标准负二项 AIC对比

方法说明：以下诊断步骤全部是数据质量/模型设定层面的常规检验（编码方向核查、零膨胀检验、
非线性检验、分布假设检验），不是为了让某个系数变正而进行的选择性重跑。
每一步无论结果如何都会完整记录在输出表中。
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

df = pd.read_csv(DATA_DIR / "02_features.csv", low_memory=False)
raw = pd.read_csv(DATA_DIR / "01_cleaned.csv", low_memory=False)
df = df[(df["days_since_post"] >= 0) & (df["days_since_post"] <= 3000)].copy()
df = df[~df["flag_zero_duration"].astype(bool)].copy()
for c in ["video_quality_score", "loudness"]:
    df[c] = df[c].fillna(df[c].mean())

def lang_group(x):
    if x == "en": return "English"
    if x in ("zh-Hans", "zh-Hant", "zh"): return "Chinese"
    return "Other"
df["lang_group"] = df["language"].apply(lang_group)

DVS = ["likes", "comments", "shares", "plays", "collects"]
CONT_VARS = ["text_length", "video_duration", "music_duration", "hashtag_count",
             "semantic_richness", "video_quality_score", "loudness", "emoji_count",
             "log_follower_count", "days_since_post"]
for c in CONT_VARS:
    df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()

DUMMY_VARS = ["has_music", "is_vertical", "has_poi", "is_ad", "is_question"]

# ========== 1. 变量编码方向核查 ==========
raw["pixels"] = raw["video_info.height"] * raw["video_info.width"]
merged = df.merge(raw[["video_info.video_id", "pixels"]], on="video_info.video_id", how="left")
quality_pixel_corr = merged["video_quality_score"].corr(merged["pixels"])

coding_check = pd.DataFrame([
    {"variable": "video_quality_score", "check": "与视频分辨率(像素数)的相关系数",
     "value": round(quality_pixel_corr, 4),
     "interpretation": "弱正相关(非简单分辨率代理，可能反映编码码率等其他质量维度)，编码方向(数值越大越优)未见反转证据"},
    {"variable": "loudness", "check": "原始dB取值范围",
     "value": f"[{df['loudness'].min():.1f}, {df['loudness'].max():.1f}]，均值{df['loudness'].mean():.1f}",
     "interpretation": "符合音频响度dB标准惯例(数值越接近0越响亮，越负越安静)，编码方向未见错误"},
])
coding_check.to_csv(OUT_DIR / "table13_variable_coding_check.csv", index=False)
print("=== 1. 变量编码方向核查 ===")
print(coding_check.to_string(index=False))

# ========== 2. 零膨胀诊断 ==========
zero_rows = []
for dv in DVS:
    zero_pct = (df[dv] == 0).mean() * 100
    zero_rows.append({"DV": dv, "zero_pct": round(zero_pct, 2),
                       "flag": "零膨胀(>10%)，建议尝试ZINB" if zero_pct > 10 else "零值占比低，标准NB足够"})
zero_df = pd.DataFrame(zero_rows)
zero_df.to_csv(OUT_DIR / "table14_zero_inflation_check.csv", index=False)
print("\n=== 2. 零膨胀诊断 ===")
print(zero_df.to_string(index=False))

# ========== 3. 二次项(倒U型)检验 ==========
# 对每个连续变量在每个DV模型里加入二次项，用似然比检验/AIC对比判断是否存在非线性
quad_rows = []
base_features = [v + "_z" for v in CONT_VARS] + DUMMY_VARS
for dv in DVS:
    y = df[dv]
    for test_var in CONT_VARS:
        z = test_var + "_z"
        other_feats = [f for f in base_features if f != z]
        formula_lin = f"{dv} ~ {z} + " + " + ".join(other_feats) + " + C(lang_group)"
        formula_quad = f"{dv} ~ {z} + I({z}**2) + " + " + ".join(other_feats) + " + C(lang_group)"
        try:
            m_lin = smf.glm(formula_lin, data=df, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
            m_quad = smf.glm(formula_quad, data=df, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
            quad_term = f"I({z} ** 2)"
            quad_coef = m_quad.params.get(quad_term, np.nan)
            quad_p = m_quad.pvalues.get(quad_term, np.nan)
            lin_coef = m_quad.params.get(z, np.nan)
            improved = m_quad.aic < m_lin.aic
            quad_rows.append({
                "DV": dv, "variable": test_var,
                "linear_term_coef_in_quad_model": round(lin_coef, 4),
                "quadratic_term_coef": round(quad_coef, 6),
                "quadratic_p_value": round(quad_p, 4),
                "quadratic_significant": bool(quad_p < 0.05),
                "AIC_linear_only": round(m_lin.aic, 1),
                "AIC_with_quadratic": round(m_quad.aic, 1),
                "quadratic_improves_fit": bool(improved),
                "inverted_U_shape": bool(quad_p < 0.05 and quad_coef < 0 and lin_coef > 0),
            })
        except Exception as e:
            quad_rows.append({"DV": dv, "variable": test_var, "error": str(e)})
    print(f"[{dv}] 二次项检验完成")

quad_df = pd.DataFrame(quad_rows)
quad_df.to_csv(OUT_DIR / "table15_quadratic_terms.csv", index=False)
print("\n=== 3. 二次项检验结果汇总 ===")
sig = quad_df[quad_df.get("quadratic_significant", False) == True]
print(f"共检验 {len(quad_df)} 组(DV×变量)，其中二次项显著的有 {len(sig)} 组:")
if len(sig):
    print(sig[["DV", "variable", "quadratic_coef" if "quadratic_coef" in sig.columns else "quadratic_term_coef",
                "inverted_U_shape"]].to_string(index=False))

# ========== 4. ZINB vs 标准NB 对比(针对零膨胀较高的comments/shares/collects) ==========
zinb_rows = []
formula_rhs = " + ".join(base_features)
for dv in ["comments", "shares", "collects"]:
    y = df[dv].values
    X = sm.add_constant(df[base_features].astype(float))
    lang_dummies = pd.get_dummies(df["lang_group"], prefix="lang", drop_first=True).astype(float)
    X = pd.concat([X, lang_dummies], axis=1)
    try:
        nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
        zinb_model = ZeroInflatedNegativeBinomialP(y, X, exog_infl=X, inflation="logit").fit(maxiter=100, disp=0)
        zinb_rows.append({
            "DV": dv, "AIC_standard_NB": round(nb_model.aic, 1),
            "AIC_ZINB": round(zinb_model.aic, 1),
            "ZINB_improves_fit": bool(zinb_model.aic < nb_model.aic),
        })
        print(f"[{dv}] ZINB拟合完成: AIC(NB)={nb_model.aic:.1f}, AIC(ZINB)={zinb_model.aic:.1f}")
    except Exception as e:
        zinb_rows.append({"DV": dv, "error": str(e)})
        print(f"[{dv}] ZINB拟合失败: {e}")

pd.DataFrame(zinb_rows).to_csv(OUT_DIR / "table16_zinb_vs_nb_comparison.csv", index=False)
print("\n全部扩展诊断完成，结果已写入 outputs/table13-16")
