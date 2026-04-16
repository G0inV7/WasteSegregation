import cv2
import datetime
from PIL import Image # Optional: used for saving actual RGB files

# 1. Initialize the webcam
cam = cv2.VideoCapture(0)

print("Press 'SPACE' to capture/save with timestamp, 'ESC' to exit.")

while True:
    ret, frame = cam.read()
    if not ret:
        break


    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC pressed
        break
    elif k % 256 == 32:  # SPACE pressed
        # 2. Convert to RGB for processing/correct saving
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 3. Generate filename using Year-Month-Day_Hour-Minute-Second
        # Format: YYYYMMDD_HHMMSS (Best for chronological sorting)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name = f"capture_{timestamp}.jpg"
        
        # 4. Save the file
        # Note: cv2.imwrite expects BGR. To save the RGB version correctly:
        img_to_save = Image.fromarray(rgb_frame)
        img_to_save.save(img_name)
        
        print(f"Saved: {img_name}")

cam.release()
cv2.destroyAllWindows()
