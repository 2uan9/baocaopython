import os
import cv2
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from collections import defaultdict
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# Cấu hình
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['FONT_FOLDER'] = 'static/fonts'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Tạo thư mục
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['FONT_FOLDER'], exist_ok=True)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Bảng dịch Anh-Việt
EN_TO_VI = {
    "apple": "táo", "banana": "chuối", "cat": "mèo", "dog": "chó",
    "person": "người", "car": "xe hơi", "chair": "ghế", "bottle": "chai",
    "book": "sách", "bird": "chim", "horse": "ngựa", "cup": "cốc"
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def get_vietnamese_name(obj_name):
    return EN_TO_VI.get(obj_name.lower(), obj_name)


def draw_vietnamese_text(img, text, position, font_size=20):
    """Hàm vẽ text tiếng Việt lên ảnh"""
    try:
        # Chuyển OpenCV -> PIL
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # Load font (ưu tiên font có dấu)
        font_path = os.path.join(app.config['FONT_FOLDER'], 'arial-unicode-ms.ttf')
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()

        # Vẽ text
        draw.text(position, text, font=font, fill=(0, 255, 0))

        # Chuyển lại OpenCV
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Lỗi khi vẽ text: {e}")
        return img


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'Không có file được chọn'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Không có file được chọn'})

        if file and allowed_file(file.filename):
            # Lưu ảnh
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{secure_filename(file.filename)}"
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            # Nhận diện
            results = model(upload_path)
            detections = results.pandas().xyxy[0]

            # Xử lý ảnh
            img = cv2.imread(upload_path)
            for _, row in detections.iterrows():
                x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                label = get_vietnamese_name(row['name'])

                # Vẽ bounding box và text
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                img = draw_vietnamese_text(img, label, (x1, y1 - 30))

            # Lưu ảnh kết quả
            result_filename = f"result_{filename}"
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            cv2.imwrite(result_path, img)

            # Chuẩn bị kết quả
            letter_counter = defaultdict(int)
            details = []
            for _, row in detections.iterrows():
                vi_name = get_vietnamese_name(row['name'])
                letter = vi_name[0].upper()
                letter_counter[letter] += 1
                details.append({
                    'object': vi_name,
                    'letter': letter,
                    'confidence': round(float(row['confidence']) * 100, 2)
                })

            return jsonify({
                'status': 'success',
                'uploaded': filename,
                'processed': result_filename,
                'letter': max(letter_counter.items(), key=lambda x: x[1])[0] if letter_counter else None,
                'details': details
            })

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)