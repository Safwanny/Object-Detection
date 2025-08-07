import cv2

print("Scanning for working cameras...")
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera index {i} is working!")
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
        cap.release()