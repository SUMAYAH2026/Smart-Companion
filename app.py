import streamlit as st
import cv2
import numpy as np
import os
import base64  
from datetime import datetime  

# 1. واجهة البرنامج (ممركزة في الوسط وبدون عنوان فرعي)
st.markdown("<h1 style='text-align: center;'>نظام رصد الأطفال المفقودين</h1>", unsafe_allow_html=True)
st.write("") 

# دالة ذكية لتشغيل الصوت تلقائياً داخل المتصفح
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

# 2. استدعاء أدوات الذكاء الاصطناعي الأساسية
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces_data = []
labels_data = []

# دالة قراءة الصور المحدثة
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

# 3. مرحلة تدريب نموذج تعلم الآلة (التعديل هنا لمنع ظهور None)
if os.path.exists("target"):
    load_folder_images("target", 1)
else:
    load_folder_images("Target", 1)

if os.path.exists("others"):
    load_folder_images("others", 2)
else:
    load_folder_images("Others", 2)

model_ready = False
if len(labels_data) > 0 and (1 in labels_data) and (2 in labels_data):
    recognizer.train(faces_data, np.array(labels_data))
    model_ready = True
    st.markdown("<div style='text-align: center; color: #155724; background-color: #d4edda; border-color: #c3e6cb; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>✅ تم تدريب النظام على تمييز المستهدف ضد الآخرين</div>", unsafe_allow_html=True)
else:
    st.markdown("<div style='text-align: center; color: #856404; background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>⚠️ تأكدي من وجود مجلد target ومجلد others وبداخلهما صور واضحة للوجوه</div>", unsafe_allow_html=True)

# 4. تشغيل كاميرا البث المباشر والمحاكاة
check_col1, check_col2, check_col3 = st.columns([1.3, 1, 1])
with check_col2:
    run_camera = st.checkbox("تفعيل كاميرا المراقبة")

video_col1, video_col2, video_col3 = st.columns([1, 3, 1])
with video_col2:
    FRAME_WINDOW = st.image([])

alert_area = st.empty()
sms_area = st.empty()

if run_camera and model_ready:
    camera = cv2.VideoCapture(0)
    
    while run_camera:
        success, frame = camera.read()
        if not success:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        child_spotted = False
        
        for (x, y, w, h) in faces:
            box_color = (255, 0, 0) 
            text_label = "Unknown"
            
            label, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            
            if label == 1 and confidence < 95:
                box_color = (0, 255, 0) 
                text_label = "MATCHED"
                child_spotted = True
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(frame, text_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
            
        FRAME_WINDOW.image(frame)
        
        if child_spotted:
            play_alarm_sound("alarm.mp3")  
            
            now = datetime.now()
            current_time = now.strftime("%I:%M:%S %p")
            current_date = now.strftime("%Y-%m-%d")
            
            sms_text = f"""
            📥 **الرسالة النصية الصادرة (SMS):**
            
            "عزيزي ولي الأمر، تم رصد طفلكم المفقود وهو بصحة جيدة.
            🗓️ التاريخ: {current_date}
            ⏱️ وقت الرصد الدقيق: {current_time}
            📍 الموقع الجغرافي: بوابة المراقبة رقم 4"
            """
            
            with alert_area.container():
                st.error("🚨 [إنذار أمني]: تم رصد وتطابق الطفل المفقود المستهدف!")
                
            with sms_area.container():
                st.info(sms_text)  
        else:
            alert_area.empty()
            sms_area.empty()
else:
    st.markdown("<div style='text-align: center; color: #0c5460; background-color: #d1ecf1; padding: 15px; border-radius: 5px;'>🔍 قم بتفعيل الكاميرا لبدء عملية البحث والرصد</div>", unsafe_allow_html=True)