import cv2 as cv
import faulthandler

faulthandler.enable()

cap = cv.VideoCapture('./giraffe.mp4')

n = 0
while True:
    # capture the next frame
    ret, frame = cap.read()

    # if frame is read correctly then ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    resized = cv.resize(frame, (224, 224))

    cv.imwrite(f"frames/frame_{n:04}.png", resized)

    n += 1

# release the capture once done with it
cap.release()
cv.destroyAllWindows()
