import streamlit as st
import cv2
import numpy as np
import os
import base64  
from datetime import datetime  
import urllib.parse 

# 1. إعداد الصفحة
st.set_page_config(layout="wide")

# تنسيق CSS للمحاذاة من اليمين لليسار، وتوسيط العنوان بالكامل
st.markdown("""
    <style>
    body, .stApp, p, div, label {
        direction: rtl !important;
        text-align: right !important;
    }
    .main-title {
        text-align: center !important;
        font-size: 32px !important;
        font-weight: bold !important;
        margin-bottom: 20px;
    }
    .report-box {
        padding: 15px; 
        border-radius: 6px; 
        margin-bottom: 15px;
        border-right: 5px solid #166534;
        background-color: #f0fdf4;
        color: #166534;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# العنوان الرئيسي ممركز تماماً في المنتصف
st.markdown('<div class="main-title">نظام رصد الأطفال المفقودين الذكي</div>', unsafe_allow_html=True)
st.write("") 

# دالة تشغيل صوت الإنذار
def play_alarm_sound(sound_file):
    if os.path.exists(sound_file):
        with open(sound_file, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            audio_html = f"""
                <audio autoplay="true" style="display:none;">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(audio_html, unsafe_allow_html=True)

# 2. استدعاء وتهيئة خوارزميات الذكاء الاصطناعي
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces_data = []
labels_data = []

def load_folder_images(folder_name, label_id):
    if os.path.exists(folder_name):
        for file in os.listdir(folder_name):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(folder_name, file)
                img = cv2.imread(path)
                if img is not None:  
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    detected = face_cascade.detectMultiScale(gray_img, 1.3, 5)
                    for (x, y, w, h) in detected:
                        faces_data.append(gray_img[y:y+h, x:x+w])
                        labels_data.append(label_id)

# تحميل صور المستهدف والأشخاص الآخرين
if os.path.exists("target"): load_folder_images("target", 1)
else: load_folder_images("Target", 1)

if os.path.exists("others"): load_folder_images("others", 2)
else: load_folder_images("Others", 2)

model_ready = False
if len(labels_data) > 0 and (1 in labels_data) and (2 in labels_data):
    recognizer.train(faces_data, np.array(labels_data))
    model_ready = True

# 3. إدارة ذاكرة الحالة (Session State) لمنع اختفاء البيانات
if "is_target_found" not in st.session_state: st.session_state.is_target_found = False
if "saved_frame" not in st.session_state: st.session_state.saved_frame = None
if "saved_time" not in st.session_state: st.session_state.saved_time = ""
if "saved_date" not in st.session_state: st.session_state.saved_date = ""

# القائمة الجانبية لتسجيل رقم الهاتف
st.sidebar.markdown("### ⚙️ الاتصال الرقمي")
parent_phone = st.sidebar.text_input("رقم هاتف ولي الأمر (مثال: 9665xxxxxxxx):", value="966500000000")

# 4. واجهة التحكم وبث الكاميرا
check_col1, check_col2, check_col3 = st.columns([1.3, 1, 1])
with check_col2:
    run_camera = st.checkbox("تفعيل كاميرا المراقبة", value=not st.session_state.is_target_found, disabled=st.session_state.is_target_found)

video_col1, video_col2, video_col3 = st.columns([1, 3, 1])
with video_col2:
    if st.session_state.is_target_found and st.session_state.saved_frame is not None:
        st.image(st.session_state.saved_frame)
    else:
        FRAME_WINDOW = st.image([])

# تشغيل بث الكاميرا وفحص الوجوه
if run_camera and model_ready and not st.session_state.is_target_found:
    camera = cv2.VideoCapture(0)
    while run_camera:
        success, frame = camera.read()
        if not success: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        for (x, y, w, h) in faces:
            label, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            
            # 🌟 تم إعادتها إلى 60 بناءً على طلبك لالتقاط الطفلة بنجاح
            if label == 1 and confidence < 60:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "MATCHED", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                now = datetime.now()
                st.session_state.saved_time = now.strftime("%I:%M:%S %p")
                st.session_state.saved_date = now.strftime("%Y-%m-%d")
                st.session_state.saved_frame = frame
                st.session_state.is_target_found = True
                
                camera.release()
                st.rerun()
            else:
                # أي وجه آخر يظهر باللون الأزرق العادي
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
        FRAME_WINDOW.image(frame)
    camera.release()

# 5. شاشة عرض النتيجة النهائية بعد رصد طائرة الدرون للطفل
if st.session_state.is_target_found:
    
    # تشغيل صوت الإنذار المستقر بعد الـ Rerun مباشرة لضمان عدم اختفائه
    play_alarm_sound("alarm.mp3")
    
    # البيانات الجغرافية لموقع طائرة الدرون الحالية
    geo_name = "المنطقة المركزية - مكة المكرمة"
    geo_map_url = "https://maps.google.com/?q=21.4225,39.8262"
    
    # صيغة رسالة الطوارئ المختصرة والموجهة للواتساب
    whatsapp_text = f"""🟢 إشعار طوارئ: تم رصد طفلكم المفقود
👤 الحالة: حصل تطابق (MATCHED)
📍 الموقع: {geo_name}
🗺️ رابط الموقع الجغرافي: {geo_map_url}
⏱️ الوقت: {st.session_state.saved_time} | التاريخ: {st.session_state.saved_date}
💡 يرجى الضغط على رابط الموقع أعلاه والتوجه فوراً لاستلام الطفل."""
    
    st.error("🚨 إنذار: تم رصد وتطابق الطفل المستهدف!")
    
    # عرض صندوق الرسالة الأنيق
    st.markdown(f'<div class="report-box"><b>🟢 الإشعار الجاهز للإرسال (WhatsApp):</b><br>{whatsapp_text}</div>', unsafe_allow_html=True)
    
    # تطهير رقم الهاتف تلقائياً من المسافات وعلامات الزائد لضمان توافق الـ API
    clean_phone = parent_phone.replace(" ", "").replace("+", "")
    
    # تشفير وتجهيز الرابط الفعلي للإرسال
    encoded_message = urllib.parse.quote(whatsapp_text)
    whatsapp_api_url = f"https://wa.me/{clean_phone}?text={encoded_message}"
    
    # زر الإرسال التفاعلي الفعلي
    st.write("")
    col_a, col_b, col_c = st.columns([1, 1.5, 1])
    with col_b:
        st.markdown(f'<a href="{whatsapp_api_url}" target="_blank"><button style="background-color: #25D366; color: white; border: none; padding: 12px 15px; border-radius: 5px; font-size: 16px; font-weight: bold; cursor: pointer; width: 100%;">🔗 إرسال رسالة لولي الأمر الآن</button></a>', unsafe_allow_html=True)
    
    # زر إعادة تصفير النظام وبدء جولة جديدة
    st.write("")
    col_b1, col_b2, col_b3 = st.columns([1.2, 1, 1])
    with col_b2:
        if st.button("🔄 بدء رصد جديد"):
            st.session_state.is_target_found = False
            st.session_state.saved_frame = None
            st.session_state.saved_time = ""
            st.session_state.saved_date = ""
            st.rerun()
else:
    if not run_camera:
        st.markdown("<div style='text-align: center; color: #0c5460; background-color: #d1ecf1; padding: 12px; border-radius: 5px;'>🔍 يرجى تفعيل الكاميرا لبدء عملية المسح والرصد الذكي</div>", unsafe_allow_html=True)