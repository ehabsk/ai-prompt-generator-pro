"""Core logic for AI Prompt Generator Pro - Enhanced.

هذا الملف يحتوي على منطق التوليد، التقييم، التاريخ، والتصدير بدون أي كود خاص بالواجهة.
"""

from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
from pathlib import Path
import logging
from functools import wraps
import time

import ollama

from templates import TEMPLATES, MASTER_TEMPLATE_ENHANCED, FIELD_LABELS

# =========================
# إعدادات عامة + تحميل الإعدادات من ملف خارجي
# =========================

DEFAULT_CONFIG: Dict = {
    "available_models": [
        "qwen3:30b",
        "dolphin-mixtral:8x7b",
        "gpt-oss:20b",
        "dolphin3:8b",
    ],
    "history_file": "prompt_history.json",
}

CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config() -> Dict:
    try:
        if CONFIG_PATH.exists():
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logging.getLogger(__name__).warning(f"تعذر قراءة config.json: {e}")
    return DEFAULT_CONFIG.copy()


_config = load_config()

AVAILABLE_MODELS: List[str] = _config.get("available_models", DEFAULT_CONFIG["available_models"])
HISTORY_PATH: Path = Path(_config.get("history_file", DEFAULT_CONFIG["history_file"]))

# إعداد السجلات
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ai_prompt_generator")

# =========================
# نظام التحقق المحسّن
# =========================

def validate_input_quality(text: str, min_length: int = 10, field_name: str = "") -> Optional[str]:
    """التحقق من جودة المحتوى"""
    if not text or not text.strip():
        return f"⚠️ حقل '{field_name}' فارغ"
    
    text = text.strip()
    
    if len(text) < min_length:
        return f"⚠️ '{field_name}' قصير جداً (أقل من {min_length} حرف). أضف مزيداً من التفاصيل"
    
    # تحقق من وجود محتوى ذي معنى (أكثر من كلمتين)
    if text.count(' ') < 2:
        return f"⚠️ '{field_name}' يحتاج لمزيد من الوضوح والتفاصيل"
    
    return None

def validate_required_fields_enhanced(
    style_choice: str,
    goal: str,
    context: str,
    instructions: str
) -> Optional[str]:
    """تحقق محسّن من الحقول الأساسية"""
    errors = []
    
    # الحقول الإلزامية
    required_fields = [
        (goal, "الهدف", 20),
        (context, "السياق", 20),
        (instructions, "التعليمات", 15),
    ]
    
    for field, label, min_len in required_fields:
        error = validate_input_quality(field, min_length=min_len, field_name=label)
        if error:
            errors.append(error)
    
    # نصائح إضافية بناءً على نوع القالب
    suggestions = []
    if style_choice == "تقني" and not any(word in goal.lower() for word in ["تطوير", "برمجة", "نظام", "api"]):
        suggestions.append("💡 للقوالب التقنية، حدد التقنية أو النظام المطلوب في الهدف")
    
    if style_choice == "إبداعي" and not any(word in goal.lower() for word in ["محتوى", "كتابة", "مقال", "منشور"]):
        suggestions.append("💡 للقوالب الإبداعية، حدد نوع المحتوى المطلوب (مقال، منشور، إعلان...)")
    
    if errors:
        result = "❌ **يرجى تحسين الحقول التالية:**\n" + "\n".join(f"• {e}" for e in errors)
        if suggestions:
            result += "\n\n" + "\n".join(suggestions)
        return result
    
    if suggestions:
        return "✅ الحقول مكتملة\n\n" + "\n".join(suggestions)
    
    return None

# =========================
# نظام الاتصال المحسّن
# =========================

def retry_on_failure(max_attempts: int = 3, delay: int = 2):
    """Decorator لإعادة المحاولة عند الفشل"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"المحاولة {attempt + 1}/{max_attempts} فشلت: {str(e)}")
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
                    else:
                        raise
            return None
        return wrapper
    return decorator

@retry_on_failure(max_attempts=3, delay=2)
def call_ollama_robust(model_name: str, prompt: str, temperature: float) -> str:
    """اتصال محسّن مع إعادة محاولة وتسجيل الأخطاء"""
    try:
        logger.info(f"استدعاء النموذج: {model_name}")
        
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options={
                "temperature": float(temperature),
                "num_predict": 3000,
                "top_p": 0.9,
            },
        )
        
        result = response.get("response", "").strip()
        
        if not result:
            raise ValueError("النموذج أرجع استجابة فارغة")
        
        logger.info(f"تم التوليد بنجاح: {len(result)} حرف")
        return result
        
    except Exception as e:
        logger.error(f"خطأ في Ollama: {type(e).__name__} - {str(e)}")
        raise Exception(
            f"❌ خطأ في الاتصال بـ {model_name}:\n"
            f"{str(e)}\n\n"
            f"تأكد من:\n"
            f"1. تشغيل Ollama: ollama serve\n"
            f"2. تحميل النموذج: ollama pull {model_name}\n"
            f"3. النموذج يعمل: ollama list"
        )

# =========================
# نظام تقييم الجودة
# =========================

def assess_prompt_quality(prompt: str) -> Dict[str, any]:
    """تقييم جودة البرومبت بناءً على معايير متعددة"""
    scores = {}
    
    # 1. الطول الأمثل (500-1500 كلمة)
    word_count = len(prompt.split())
    if word_count < 300:
        scores['length'] = (word_count / 300) * 5
    elif word_count <= 1500:
        scores['length'] = 10
    else:
        scores['length'] = max(5, 10 - (word_count - 1500) / 500)
    
    # 2. وجود عناصر أساسية
    essential_elements = {
        "الدور": ["أنت", "خبير", "متخصص", "مهندس", "محلل", "كاتب"],
        "القيود": ["لا تختلق", "لا ت", "ممنوع", "يجب", "التزم"],
        "KPIs": ["مؤشر", "قياس", "نسبة", "%", "معدل", "عدد"],
        "التنسيق": ["تنسيق", "صيغة", "شكل", "بنية", "هيكل"],
        "الأمثلة": ["مثال", "مثل", "كـ", "على سبيل"]
    }
    
    elements_found = sum(
        1 for keywords in essential_elements.values()
        if any(k in prompt.lower() for k in keywords)
    )
    scores['completeness'] = (elements_found / len(essential_elements)) * 10
    
    # 3. الوضوح (نسبة العناوين والنقاط)
    structure_markers = prompt.count('#') + prompt.count('-') + prompt.count('•') + prompt.count('**')
    scores['clarity'] = min(10, (structure_markers / 10) * 10)
    
    # 4. التحديد (وجود أرقام ونسب)
    numbers = len([c for c in prompt if c.isdigit()])
    scores['specificity'] = min(10, (numbers / 10) * 10)
    
    # 5. التنظيم (أقسام واضحة)
    sections = prompt.count('\n#')
    scores['organization'] = min(10, (sections / 5) * 10)
    
    # 6. الأمان (قواعد ضد الاختلاق)
    safety_keywords = ["لا تختلق", "لا تخترع", "التزم", "تحقق", "تأكد"]
    safety_score = sum(1 for kw in safety_keywords if kw in prompt)
    scores['safety'] = min(10, (safety_score / 2) * 10)
    
    # الدرجة الإجمالية
    total = sum(scores.values()) / len(scores)
    
    # تحديد التقدير
    if total >= 9:
        grade = "ممتاز جداً 🏆"
    elif total >= 8:
        grade = "ممتاز ⭐"
    elif total >= 7:
        grade = "جيد جداً ✅"
    elif total >= 6:
        grade = "جيد ✓"
    else:
        grade = "يحتاج تحسين ⚠️"
    
    return {
        'total_score': round(total, 1),
        'breakdown': {k: round(v, 1) for k, v in scores.items()},
        'word_count': word_count,
        'grade': grade,
        'char_count': len(prompt),
        'sections': sections
    }

def suggest_improvements(prompt: str, quality_scores: Dict) -> List[str]:
    """اقتراحات لتحسين البرومبت بناءً على التقييم"""
    suggestions = []
    
    scores = quality_scores['breakdown']
    
    if scores.get('completeness', 10) < 7:
        suggestions.append("💡 أضف المزيد من العناصر الأساسية: KPIs واضحة، قيود محددة، أمثلة عملية")
    
    if scores.get('safety', 10) < 7:
        suggestions.append("💡 عزز الأمان: أضف قواعد صريحة لمنع اختلاق المعلومات")
    
    if scores.get('clarity', 10) < 7:
        suggestions.append("💡 حسّن التنظيم: استخدم عناوين فرعية ونقاط أكثر وضوحاً")
    
    if scores.get('specificity', 10) < 7:
        suggestions.append("💡 كن أكثر تحديداً: أضف أرقاماً ونسباً ومعايير قابلة للقياس")
    
    word_count = quality_scores['word_count']
    if word_count < 300:
        suggestions.append(f"💡 البرومبت قصير ({word_count} كلمة) - أضف مزيداً من التفاصيل والسياق")
    elif word_count > 2000:
        suggestions.append(f"💡 البرومبت طويل جداً ({word_count} كلمة) - حاول التركيز على الأساسيات")
    
    if quality_scores['sections'] < 3:
        suggestions.append("💡 قسّم البرومبت لأقسام أكثر (الدور، الهدف، القيود، KPIs، التنسيق)")
    
    return suggestions

# =========================
# نظام الذاكرة
# =========================

class PromptHistory:
    """حفظ واسترجاع تاريخ البرومبتات الناجحة"""
    
    def __init__(self, filepath: str = "prompt_history.json"):
        self.filepath = Path(filepath)
        self.history = self._load()
    
    def _load(self) -> List[Dict]:
        """تحميل التاريخ من الملف"""
        try:
            if self.filepath.exists():
                return json.loads(self.filepath.read_text(encoding='utf-8'))
        except Exception as e:
            logger.error(f"خطأ في تحميل التاريخ: {e}")
        return []
    
    def save_prompt(self, prompt_data: Dict):
        """حفظ برومبت جديد"""
        try:
            self.history.append({
                **prompt_data,
                'timestamp': datetime.now().isoformat(),
                'id': len(self.history) + 1
            })
            
            # حفظ آخر 100 برومبت فقط
            self.history = self.history[-100:]
            
            self.filepath.write_text(
                json.dumps(self.history, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            logger.info(f"تم حفظ البرومبت في التاريخ")
        except Exception as e:
            logger.error(f"خطأ في حفظ التاريخ: {e}")
    
    def get_similar_prompts(self, keywords: List[str], limit: int = 3) -> List[Dict]:
        """استرجاع برومبتات مشابهة"""
        scored = []
        for item in self.history:
            prompt_text = item.get('prompt', '').lower()
            score = sum(k.lower() in prompt_text for k in keywords)
            if score > 0:
                scored.append((score, item))
        
        return [item for _, item in sorted(scored, reverse=True)[:limit]]
    
    def get_stats(self) -> Dict:
        """إحصائيات التاريخ"""
        if not self.history:
            return {"total": 0}
        
        styles = {}
        for item in self.history:
            style = item.get('style', 'غير محدد')
            styles[style] = styles.get(style, 0) + 1
        
        avg_score = sum(item.get('quality_score', 0) for item in self.history) / len(self.history)
        
        return {
            "total": len(self.history),
            "styles": styles,
            "avg_quality": round(avg_score, 1),
            "last_used": self.history[-1].get('timestamp', 'غير معروف')
        }

# إنشاء كائن التاريخ العام
history = PromptHistory(str(HISTORY_PATH))

# =========================
# دوال المساعدة الكاملة
# =========================

def build_user_request(
    language_pref: str,
    vegetarian_mode: bool,
    goal: str,
    context: str,
    instructions: str,
    role: str,
    data: str,
    audience: str,
    tone: str,
    output_format: str,
    extra: str,
    constraints: str,
    kpis: str,
    budget: str,
    timeline: str,
) -> str:
    """تجميع المدخلات في طلب موحد"""
    parts: List[str] = []
    
    # الحقول الأساسية
    if goal.strip():
        parts.append(f"🎯 **{FIELD_LABELS['goal']}**:\n{goal.strip()}")
    
    if context.strip():
        parts.append(f"📋 **{FIELD_LABELS['context']}**:\n{context.strip()}")
    
    if instructions.strip():
        parts.append(f"⚙️ **{FIELD_LABELS['instructions']}**:\n{instructions.strip()}")
    
    # الحقول الاختيارية
    if role.strip():
        parts.append(f"👤 **{FIELD_LABELS['role']}**:\n{role.strip()}")
    
    if data.strip():
        parts.append(f"📊 **{FIELD_LABELS['data']}**:\n{data.strip()}")
    
    if audience.strip():
        parts.append(f"👥 **{FIELD_LABELS['audience']}**:\n{audience.strip()}")
    
    if tone.strip():
        parts.append(f"🎨 **{FIELD_LABELS['tone']}**:\n{tone.strip()}")
    
    if output_format.strip():
        parts.append(f"📄 **{FIELD_LABELS['format']}**:\n{output_format.strip()}")
    
    # الحقول المحسّنة 10/10
    if constraints.strip():
        parts.append(f"🚫 **{FIELD_LABELS['constraints']}**:\n{constraints.strip()}")
    
    if kpis.strip():
        parts.append(f"📈 **{FIELD_LABELS['kpis']}**:\n{kpis.strip()}")
    
    if budget.strip():
        parts.append(f"💰 **{FIELD_LABELS['budget']}**:\n{budget.strip()}")
    
    if timeline.strip():
        parts.append(f"⏰ **{FIELD_LABELS['timeline']}**:\n{timeline.strip()}")
    
    if extra.strip():
        parts.append(f"ℹ️ **{FIELD_LABELS['extra']}**:\n{extra.strip()}")
    
    # التفضيلات
    if language_pref.strip():
        parts.append(f"🌍 **اللغة المفضلة**: {language_pref.strip()}")
    
    if vegetarian_mode:
        parts.append("🥗 **ملاحظة**: الوصفات نباتية (إن كان ذا صلة)")
    
    return "\n\n---\n\n".join(parts) if parts else "لا يوجد محتوى."

def generate_prompt(
    model_name: str,
    style_choice: str,
    language_pref: str,
    temperature: float,
    vegetarian_mode: bool,
    goal: str,
    context: str,
    instructions: str,
    role: str,
    data: str,
    audience: str,
    tone: str,
    output_format: str,
    extra: str,
    constraints: str,
    kpis: str,
    budget: str,
    timeline: str,
) -> Tuple[str, str, str, str]:
    """الدالة الرئيسية لتوليد البرومبت - محسّنة"""
    
    try:
        # 1. التحقق من الحقول
        error = validate_required_fields_enhanced(
            style_choice, goal=goal, context=context, instructions=instructions
        )
        if error and "يرجى تحسين" in error:
            return "", error, "", ""
        
        # 2. بناء طلب المستخدم
        user_request = build_user_request(
            language_pref, vegetarian_mode, goal, context, instructions,
            role, data, audience, tone, output_format, extra,
            constraints, kpis, budget, timeline
        )
        
        # 3. اختيار القالب المناسب
        template = TEMPLATES.get(style_choice, MASTER_TEMPLATE_ENHANCED)
        meta_prompt = template.format(user_request=user_request)
        
        # 4. التوليد (مرة واحدة فقط - إصلاح المشكلة الحرجة!)
        logger.info(f"بدء التوليد: النموذج={model_name}, القالب={style_choice}")
        final_prompt = call_ollama_robust(model_name, meta_prompt, temperature)
        
        if not final_prompt:
            return "", "❌ فشل التوليد - حاول مرة أخرى", "", ""
        
        # 5. تقييم الجودة
        quality_assessment = assess_prompt_quality(final_prompt)
        improvements = suggest_improvements(final_prompt, quality_assessment)
        
        # 6. إنشاء تقرير الجودة
        quality_report = f"""
### 📊 تقييم الجودة التلقائي

**الدرجة الإجمالية**: {quality_assessment['total_score']}/10 - {quality_assessment['grade']}

**التفاصيل**:
- 📏 الطول: {quality_assessment['breakdown']['length']}/10 ({quality_assessment['word_count']} كلمة)
- ✅ الاكتمال: {quality_assessment['breakdown']['completeness']}/10
- 🔍 الوضوح: {quality_assessment['breakdown']['clarity']}/10
- 🎯 التحديد: {quality_assessment['breakdown']['specificity']}/10
- 📑 التنظيم: {quality_assessment['breakdown']['organization']}/10
- 🔒 الأمان: {quality_assessment['breakdown']['safety']}/10

**الإحصائيات**:
- عدد الكلمات: {quality_assessment['word_count']}
- عدد الأحرف: {quality_assessment['char_count']}
- عدد الأقسام: {quality_assessment['sections']}
"""
        
        if improvements:
            quality_report += "\n\n### 💡 اقتراحات للتحسين\n"
            quality_report += "\n".join(improvements)
        else:
            quality_report += "\n\n### 🎉 ممتاز! البرومبت بجودة عالية جداً"
        
        # 7. إنشاء الملخص
        summary = f"""
### 📋 ملخص البرومبت المثالي

- **الإطار المستخدم**: {style_choice}
- **النموذج**: {model_name}
- **درجة الجودة**: {quality_assessment['total_score']}/10 {quality_assessment['grade']}
- **الطول**: {quality_assessment['word_count']} كلمة ({quality_assessment['char_count']} حرف)
- **عدد الأقسام**: {quality_assessment['sections']}
- **الوقت**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

**المميزات المضافة**:
✅ قيود واضحة ضد الاختلاق
✅ مؤشرات أداء قابلة للقياس (KPIs)
✅ تنظيم احترافي بالعناوين
✅ أمثلة وسيناريوهات عملية
✅ معايير جودة محددة
"""
        
        # 8. حفظ في التاريخ
        try:
            history.save_prompt({
                'prompt': final_prompt,
                'style': style_choice,
                'model': model_name,
                'quality_score': quality_assessment['total_score'],
                'word_count': quality_assessment['word_count'],
                'goal': goal[:100],  # أول 100 حرف
            })
        except Exception as e:
            logger.error(f"خطأ في حفظ التاريخ: {e}")
        
        # 9. رسالة نجاح
        success_msg = f"✅ تم التوليد بنجاح! الجودة: {quality_assessment['total_score']}/10"
        
        logger.info(f"اكتمل التوليد بنجاح: {quality_assessment['word_count']} كلمة، درجة {quality_assessment['total_score']}")
        
        return final_prompt, success_msg, summary, quality_report
        
    except Exception as e:
        logger.error(f"خطأ في generate_prompt: {e}", exc_info=True)
        return "", f"❌ حدث خطأ: {str(e)}", "", ""

def export_prompt(prompt: str, format_type: str) -> Tuple[str, str]:
    """تصدير البرومبت بصيغ متعددة"""
    if not prompt or not prompt.strip():
        return "", "⚠️ لا يوجد برومبت للتصدير"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        if format_type == "JSON":
            data = {
                "prompt": prompt,
                "created_at": timestamp,
                "version": "Enhanced 10/10",
                "quality": "professional",
                "word_count": len(prompt.split()),
                "char_count": len(prompt)
            }
            result = json.dumps(data, ensure_ascii=False, indent=2)
            return result, f"✅ تم التصدير بصيغة JSON ({len(result)} حرف)"
        
        elif format_type == "Markdown":
            md = f"""---
title: AI Prompt Professional
date: {timestamp}
version: Enhanced 10/10
quality: professional
---

{prompt}

---
*تم التوليد بواسطة AI Prompt Generator Pro*
"""
            return md, f"✅ تم التصدير بصيغة Markdown ({len(md)} حرف)"
        
        else:  # نص عادي
            header = f"=" * 60 + f"\nبرومبت احترافي 10/10\nالتاريخ: {timestamp}\n" + "=" * 60 + "\n\n"
            result = header + prompt
            return result, f"✅ تم التصدير بصيغة نص ({len(result)} حرف)"
    
    except Exception as e:
        logger.error(f"خطأ في التصدير: {e}")
        return "", f"❌ فشل التصدير: {str(e)}"

def clear_all_fields():
    """مسح جميع الحقول"""
    return [
        "",  # goal
        "",  # context
        "",  # instructions
        "",  # role
        "",  # data
        "",  # audience
        "",  # tone
        "",  # output_format
        "",  # extra
        "",  # constraints
        "",  # kpis
        "",  # budget
        "",  # timeline
        "العربية الفصحى المبسطة",  # language_pref
        False,  # vegetarian_checkbox
        0.3,  # temperature
        "MASTER_10_10",  # style
        "",  # output_prompt
        "جاهز للبدء! املأ الحقول الأساسية",  # success_msg
        "### 📋 ملخص\nلم يتم التوليد بعد",  # summary
        "### 📊 تقييم الجودة\nلم يتم التقييم بعد",  # quality
    ]

def load_example(example_type: str):
    """تحميل أمثلة جاهزة"""
    examples = {
        "تقني": {
            "goal": "بناء نظام إدارة مخزون متكامل لمتجر إلكتروني",
            "context": "شركة تجارة إلكترونية متوسطة الحجم تحتاج لنظام آلي لتتبع المخزون",
            "instructions": "صمم معمارية نظام كامل مع APIs وقاعدة بيانات ومؤشرات أداء",
            "constraints": "لا تستخدم تقنيات قديمة، التزم بأفضل الممارسات الأمنية",
            "kpis": "زمن استجابة API < 200ms، دعم 1000 طلب/دقيقة، Uptime 99.9%",
        },
        "إبداعي": {
            "goal": "كتابة منشور تسويقي جذاب لمنتج تقني جديد",
            "context": "إطلاق تطبيق ذكاء اصطناعي للطلاب يساعدهم في التعلم",
            "instructions": "اكتب منشور 300 كلمة بأسلوب حماسي وملهم يستهدف الطلاب",
            "audience": "طلاب جامعيون 18-25 سنة، مهتمون بالتكنولوجيا",
            "tone": "حماسي، ودود، ملهم، يركز على الفوائد العملية",
            "kpis": "معدل تفاعل 5%، 100 مشاركة خلال أسبوع، CTR 3%",
        },
        "تحليلي": {
            "goal": "تحليل بيانات مبيعات الربع الأخير واستخراج رؤى قابلة للتنفيذ",
            "context": "شركة B2B لديها بيانات مبيعات 10,000 عميل على مدى 3 أشهر",
            "instructions": "حلل الاتجاهات، حدد العملاء الأكثر ربحية، واقترح استراتيجيات نمو",
            "data": "ملفات CSV تحتوي على: تاريخ، منتج، عميل، مبلغ، منطقة",
            "kpis": "زيادة متوسط قيمة الطلب 15%، معدل احتفاظ 85%، نمو شهري 10%",
        },
        "أعمال": {
            "goal": "إعداد خطة عمل لإطلاق منصة توصيل طعام صحي",
            "context": "سوق ناشئ مع منافسة متوسطة، ميزانية بدء 500,000 ريال",
            "instructions": "أنشئ خطة شاملة: نموذج العمل، التسعير، التسويق، الخطة المالية",
            "audience": "مستثمرون محتملون وشركاء استراتيجيون",
            "budget": "500,000 ريال رأس مال أولي، 50,000 ريال تسويق شهري",
            "timeline": "3 أشهر للإطلاق، الوصول للربحية خلال 18 شهر",
            "kpis": "1000 عميل نشط في 6 أشهر، معدل نمو شهري 20%، هامش ربح 25%",
        }
    }
    
    example = examples.get(example_type, {})
    
    return [
        example.get("goal", ""),
        example.get("context", ""),
        example.get("instructions", ""),
        example.get("role", ""),
        example.get("data", ""),
        example.get("audience", ""),
        example.get("tone", ""),
        example.get("output_format", ""),
        "",  # extra
        example.get("constraints", ""),
        example.get("kpis", ""),
        example.get("budget", ""),
        example.get("timeline", ""),
    ]

def get_history_stats():
    """الحصول على إحصائيات الاستخدام"""
    stats = history.get_stats()
    
    if stats['total'] == 0:
        return "### 📊 إحصائيات الاستخدام\n\nلم يتم توليد أي برومبتات بعد"
    
    styles_text = "\n".join(f"  - {style}: {count} برومبت" for style, count in stats['styles'].items())
    
    return f"""
### 📊 إحصائيات الاستخدام

**إجمالي البرومبتات**: {stats['total']}

**التوزيع حسب النوع**:
{styles_text}

**متوسط الجودة**: {stats['avg_quality']}/10

**آخر استخدام**: {stats['last_used']}
"""

# =========================
# بناء الواجهة المحسّنة
# =========================

