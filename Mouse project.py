import pyautogui

import mediapipe as mp

import time

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(max_num_hands=1)

wCam, hCam = 640, 480

frameR = 100  # Frame Reduction

smoothening = 5

wScr, hScr = pyautogui.size()

pTime = 0

plocX, plocY = 0, 0

clocX, clocY = 0, 0

while True:

    success, img = pyautogui.screenshot().numpy(),  # Get screenshot

    # Convert RGB to BGR for compatibility with OpenCV

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Flip the image horizontally for correct handedness output

    img = cv2.flip(img, 1)

    # Convert to grayscale

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Process the image with the hands detection model

    results = hands.process(img)

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]  # Use only the first hand detected

        index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        middle_finger = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        index_finger_x, index_finger_y = int(index_finger.x * wCam), int(index_finger.y * hCam)

        middle_finger_x, middle_finger_y = int(middle_finger.x * wCam), int(middle_finger.y * hCam)

        # Check which fingers are up

        is_index_finger_up = index_finger_y < middle_finger_y

        is_middle_finger_up = middle_finger_y < index_finger_y

        # Draw a rectangle around the screen

        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # Moving mode - only index finger is up

        if is_index_finger_up and not is_middle_finger_up:

            # Convert coordinates to screen coordinates

            x3 = np.interp(index_finger_x, (frameR, wCam - frameR), (0, wScr))

            y3 = np.interp(index_finger_y, (frameR, hCam - frameR), (0, hScr))

            # Smoothen values

            clocX = plocX + (x3 - plocX) / smoothening

            clocY = plocY + (y3 - plocY) / smoothening

            # Move the mouse

            pyautogui.moveTo(wScr - clocX, clocY)

            # Draw a circle around the index finger

            cv2.circle(img, (index_finger_x, index_finger_y), 15, (255, 0, 0), cv2.FILLED)

            # Update previous position

            plocX, plocY = clocX, clocY

        # Clicking mode - both index and middle fingers are up

        if is_index_finger_up and is_middle_finger_up:

            # Find the distance between the index and middle fingers

            distance = abs(index_finger_y - middle_finger_y)

            # Click themouse if the distance between the index and middle fingers is small

if distance < 30:

# Click the left mouse button

pyautogui.click(button='left')
# Display the FPS

fps = 1.0 / (time.time() - prev_frame_time)

cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Display the frame
Release the video capture device and close all windows

cap.release()

cv2.destroyAllWindows()
cv2.imshow("Hand Gesture Recognition", frame)

# Check if the user has pressed the 'q' key to quit

if cv2.waitKey(1) & 0xFF == ord('q'):

    break


