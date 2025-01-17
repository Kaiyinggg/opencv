import cv2
import numpy as np
from imutils.perspective import four_point_transform
import pytesseract
from datetime import datetime


cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

WIDTH, HEIGHT = 800,600
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
count = 0
def image_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    return threshold



def scan_detection(image):
    global document_countour


    document_countour = np.array([[0,0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _= cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                document_countour = approx
                max_area = area
    cv2.drawContours(frame, [document_countour], -1, (0,255,0), 3)

# while True:
#     _, frame = cap.read()
#     frame = cv2.rotate(frame, cv2.ROTATE_180)
#     frame_copy = frame.copy()
#     scan_detection(frame_copy)
#     cv2.imshow("input: ", frame)

#     warped = four_point_transform(frame_copy, document_countour.reshape(4,2))
#     cv2.imshow("Warped: ", warped)

#     processed = image_processing(warped)
#     processed = processed[10:processed.shape[0] - 10, 10:processed.shape[1]-10]
#     cv2.imshow("Processed", processed)
#     text = pytesseract.image_to_string(warped)
#     print("Detected Text:", text)

#     pressed_key = cv2.waitKey(1) & 0xFF
#     if pressed_key == 27:
#         break
#     elif pressed_key == ord('s'):
#         cv2.imwrite(f"output/scanned{count}.jpg", processed)
#         count += 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame_copy = frame.copy()

    scan_detection(frame_copy)

    if document_countour is not None:
        warped = four_point_transform(frame_copy, document_countour.reshape(4, 2))
        processed = image_processing(warped)
        processed = processed[10:processed.shape[0] - 10, 10:processed.shape[1] - 10]
        
        text = pytesseract.image_to_string(processed)
        print("Detected Text:", text)

        cv2.imshow("Warped", warped)
        cv2.imshow("Processed", processed)
    else:
        cv2.putText(frame, "No document detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Input", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Escape key
        break
    elif key == ord('s'):
        save_path = f"output/scanned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(save_path, processed)
        print(f"Saved: {save_path}")

cap.release()
cv2.destroyAllWindows()

