from flask import Flask, render_template, Response
import cv2
import imutils

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    cap = cv2.VideoCapture('http://<IP_ADDRESS>/video')

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = imutils.resize(frame, width=480)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)


