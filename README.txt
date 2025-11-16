AI Prompt Generator Pro - Modular Version
===========================================

الهيكل الجديد:
- core.py      -> منطق التوليد والتقييم والتاريخ
- templates.py -> القوالب النصية وتعريف الحقول
- ui.py        -> واجهة Gradio + CSS غامق
- config.json  -> النماذج المتاحة ومسار ملف التاريخ

لتشغيل الواجهة:
1) تأكد من تفعيل البيئة (venv) وتثبيت gradio و ollama
2) شغل:
   pip install -r requirements.txt
   python ui.py
إنشاء وتفعيل  venv
    لينكس/ماك:
        إنشاء: python3 -m venv .venv
        تفعيل: source .venv/bin/activate
    ويندوز (PowerShell):
      إنشاء: py -m venv .venv

## 🎥 فيديو توضيحي
![واجهة الأداة](assets/demo.png)

<video src="assets/demo.mp4" controls width="900">
  متصفحك لا يدعم تشغيل الفيديو، يمكنك تحميله من الملفات في المستودع.
</video>

assets/demoمشاهدة الفيديو

## 👤 المؤلف

**Ehab Khafagy (ehabsk)**  
مطوّر أداة AI Prompt Generator Pro

- GitHub: [@ehabsk](https://github.com/ehabsk)
- البريد: ehab.alforat@gmail.com

## 🙏 شكر وتقدير

تم تطوير الأداة بمساعدة نماذج الذكاء الاصطناعي (cloude, Kime, gemeni, ChatGPT / Maria) في كتابة الكود
وتحسين القوالب وتصميم الواجهة.
