import cv2

# Open the default camera (0 = first camera on the device)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open the camera")
    exit()

while True:
    # Read one frame from the camera
    success, frame = cap.read()
    if not success:
        print("Could not read a frame")
        break

    # Flip horizontally so it behaves like a mirror
    frame = cv2.flip(frame, 1)

    cv2.imshow("Hand Gesture Control", frame)

    # Wait 1ms for a key press; quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
