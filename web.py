from flask import Flask, render_template_string, request, jsonify
from PIL import Image
import numpy as np
from io import BytesIO
import base64
from model import RBM
import torch
import torch.nn as nn
from process import get_image_tensor

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    global rbm_model, clf_model
    try:
        device = torch.device('cuda')
        if 'rbm_model' not in globals() or rbm_model is None:
            rbm_model = RBM(n_visible=784, n_hidden=200)
            rbm_model.load_state_dict(torch.load('./models/rbm.pth', map_location=device))
            rbm_model.to(device)
            rbm_model.eval()

        if 'clf_model' not in globals() or clf_model is None:
            clf_model = nn.Linear(rbm_model.n_hidden, 10)
            clf_model.load_state_dict(torch.load('./models/clf.pth', map_location=device))
            clf_model.to(device)
            clf_model.eval()

        data = request.get_json()
        image_data = data.get('image', '')
        image_data = image_data.split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image.save('./images/received_raw.png')

        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
            bg.paste(image, mask=image.convert('RGBA').split()[-1])
            image = bg.convert('RGB')

        img_array = np.array(image)
        img_array = 255 - img_array
        image = Image.fromarray(img_array.astype(np.uint8))
        image.save('./images/received_processed.png')

        img_tensor = get_image_tensor(image)
        img_show = img_tensor.squeeze().numpy().astype('uint8')
        Image.fromarray(img_show).save('./images/adjusted_image.png')

        v = img_tensor.view(1, -1).to(device) / 255.0
        v = torch.clamp(v, 0, 1)
        v = torch.bernoulli(v)

        with torch.no_grad():
            p_h, _ = rbm_model.visible_to_hidden(v)
            logits = clf_model(p_h)
            probs = torch.softmax(logits, dim=1)
            pred = int(probs.argmax(dim=1).item())
            confidence = float(probs[0, pred].item())

        return jsonify({'success': True, 'prediction': pred, 'confidence': confidence})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <title>RBM 手写数字识别</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            .drawing-area {
                text-align: center;
                margin: 20px 0;
            }

            #canvas {
                border: 2px solid #333;
                border-radius: 5px;
                cursor: crosshair;
                background: white;
            }
            .controls {
                text-align: center;
                margin-bottom: 20px;
            }
            .btn-primary, .btn-secondary {
                margin: 0 10px;
                padding: 10px 20px;
                font-size: 16px;
            }
            .status {
                text-align: center;
                margin: 10px 0;
                font-size: 18px;
            }
            .status.success {
                color: green;
            }
            .status.error {
                color: red;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="drawing-area">
                <canvas id="canvas" width="280" height="280"></canvas>
            </div>

            <div class="controls">
                <button class="btn-primary" onclick="recognize()"> 识别</button>
                <button class="btn-secondary" onclick="clearCanvas()"> 清空</button>
            </div>

            <div id="status" class="status" style="display: none;"></div>

            <div class="result" id="result" style="display: none;">
                <div class="prediction">预测结果：<span id="prediction"></span></div>
                <div class="confidence">置信度：<span id="confidence"></span></div>
            </div>
        </div>
        <script>
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            let isDrawing = false;

            // 设置画布样式
            ctx.lineWidth = 15;
            ctx.lineCap = 'round';
            ctx.strokeStyle = '#000000';

            // 鼠标事件
            canvas.addEventListener('mousedown', startDrawing);
            canvas.addEventListener('mousemove', draw);
            canvas.addEventListener('mouseup', stopDrawing);
            canvas.addEventListener('mouseout', stopDrawing);

            // 触摸事件（移动设备）
            canvas.addEventListener('touchstart', handleTouch);
            canvas.addEventListener('touchmove', handleTouch);
            canvas.addEventListener('touchend', stopDrawing);

            function startDrawing(e) {
                isDrawing = true;
                draw(e);
            }

            function draw(e) {
                if (!isDrawing) return;

                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;

                ctx.lineTo(x, y);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(x, y);
            }

            function handleTouch(e) {
                e.preventDefault();
                const touch = e.touches[0];
                const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' :
                    e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
                    clientX: touch.clientX,
                    clientY: touch.clientY
                });

                canvas.dispatchEvent(mouseEvent);
            }

            function stopDrawing() {
                isDrawing = false;
                ctx.beginPath();
            }

            function clearCanvas() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                document.getElementById('result').style.display = 'none';
            }

            function showStatus(message, isError = false) {
                const status = document.getElementById('status');
                status.textContent = message;
                status.className = `status ${isError ? 'error' : 'success'}`;
                status.style.display = 'block';
            }

            function hideStatus() {
                document.getElementById('status').style.display = 'none';
            }

            async function recognize() {
                showStatus('正在识别...');
                
                try {
                    // 获取画布图像
                    const imageData = canvas.toDataURL('image/png');

                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({image: imageData})
                    });

                    const result = await response.json();

                    if (result.success) {
                        document.getElementById('prediction').textContent = result.prediction;
                        document.getElementById('confidence').textContent = result.confidence.toFixed(4);
                        document.getElementById('result').style.display = 'block';
                        showStatus('识别完成！');
                    } else {
                        showStatus('识别失败：' + result.error, true);
                    }
                } catch (error) {
                    showStatus('识别出错：' + error.message, true);
                }

                setTimeout(hideStatus, 3000);
            }

            clearCanvas();
        </script>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True)