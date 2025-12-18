import os
import google.generativeai as genai
from dotenv import load_dotenv

# โหลด API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: ไม่พบ API Key ในไฟล์ .env")
else:
    print(f"Key found: {api_key[:5]}...********")
    
    # ตั้งค่าและดึงรายชื่อ Model
    genai.configure(api_key=api_key)
    
    print("\nกำลังดึงรายชื่อ Model ที่บัญชีคุณใช้ได้... (รอสักครู่)")
    try:
        print("--- SUPPORTED MODELS ---")
        for m in genai.list_models():
            # กรองเฉพาะ Model ที่คุยแชทได้ (generateContent)
            if 'generateContent' in m.supported_generation_methods:
                print(f"ชื่อ Model: {m.name}")
        print("------------------------")
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")