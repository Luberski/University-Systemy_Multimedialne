from flask import Flask, render_template, Response, request
import cv2

app = Flask(__name__)

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier()
face_cascade.load(cv2.samples.findFile("haarcascade_frontalface_alt.xml"))

imgprocess = 0

def gen_frames(video):
    while True:
        success, image = video.read()
        if success:
            frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.equalizeHist(frame_gray)

            faces = face_cascade.detectMultiScale(frame_gray)

            if(imgprocess > 0):
                if(imgprocess == 1):
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if(imgprocess == 2):
                    image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)

            for (x, y, w, h) in faces:
                center = (x + w // 2, y + h // 2)
                cv2.putText(
                    image,
                    "X: " + str(center[0]) + " Y: " + str(center[1]),
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    3,
                )
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            ret, jpeg = cv2.imencode(".jpg", image)

            frame = jpeg.tobytes()

            yield (
                b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n"
            )
        else:
            break


@app.route("/video_feed", methods=["GET", "POST"])
def video_feed():
    global imgprocess
    if request.method == 'POST':
        if request.form['submit_button'] == 'Grayscale':
            imgprocess = 1
            index()
        elif request.form['submit_button'] == 'Normal':
            imgprocess = 0
        elif request.form['submit_button'] == 'Denoise':
            imgprocess = 2
        else:
            pass # unknownszs
    return Response(
        gen_frames(video), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
