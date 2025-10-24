import cv2
import time
import numpy as np
import argparse
import os

FLAG_SAVE_VIDEO = False

def check_file_naming(file_name: str) -> bool:
    """Check if filename ends with .mp4"""
    return file_name.lower().endswith('.mp4')

parser = argparse.ArgumentParser()
parser.add_argument('--save_video_path', type=str, default='', help='Directory to save video')
parser.add_argument('--name', type=str, default='', help='Video file name (.mp4)')
args = parser.parse_args()

save_video_path = args.save_video_path
save_video_name = args.name

if save_video_path:
    if not save_video_name:
        save_video_name = f"{int(time.time())}.mp4"
    if not os.path.isdir(save_video_path):
        try:
            os.makedirs(save_video_path, exist_ok=True)
            print(f"New folder has been created: {save_video_path}")
        except OSError as e:
            print(f"Error creating folder: {e}")
            exit(1)
    FLAG_SAVE_VIDEO = True

if save_video_name:
    if not save_video_path:
        save_video_path = './'
    full_path = os.path.join(save_video_path, save_video_name)
    if os.path.isfile(full_path):
        print(f"File already exists: {full_path}")
        raise FileExistsError(full_path)
    if not check_file_naming(save_video_name):
        print("Error in file name: file name must end with .mp4")
        raise ValueError("Invalid file extension")
    FLAG_SAVE_VIDEO = True
else:
    full_path = ''


ddepth = cv2.CV_16S
erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 840)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cam.isOpened():
    print("Could not open camera.")
    exit(1)

cv2.namedWindow('main')
cv2.createTrackbar('canny_lower', 'main', 50, 1000, lambda x: None)
cv2.createTrackbar('canny_upper', 'main', 150, 1000, lambda x: None)

font = cv2.FONT_HERSHEY_SIMPLEX


def stack_images(img_array, cols=3, scale=0.3):
    """Stack images in a grid for display"""
    rows = []
    for i in range(0, len(img_array), cols):
        row_imgs = img_array[i:i + cols]
        row_imgs = [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img for img in row_imgs]
        h_min = min(img.shape[0] for img in row_imgs)
        scaled = [cv2.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min)) for img in row_imgs]
        w_min = min(img.shape[1] for img in scaled)
        resized = [cv2.resize(img, (w_min, h_min)) for img in scaled]
        while len(resized) < cols:
            resized.append(np.zeros_like(resized[0]))
        rows.append(np.hstack(resized))
    max_width = max(r.shape[1] for r in rows)
    for i, r in enumerate(rows):
        if r.shape[1] < max_width:
            pad = np.zeros((r.shape[0], max_width - r.shape[1], 3), dtype=np.uint8)
            rows[i] = np.hstack([r, pad])
    collage = np.vstack(rows)
    if scale != 1.0:
        collage = cv2.resize(collage, None, fx=scale, fy=scale)
    return collage


writer = None
if FLAG_SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(full_path, fourcc, 20.0, (840, 480))

prev_tick = cv2.getTickCount()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Camera read failed.")
        break

    canny_lower = cv2.getTrackbarPos('canny_lower', 'main')
    canny_upper = cv2.getTrackbarPos('canny_upper', 'main')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    image = {
        'original': frame.copy(),
        'gaussianblur_3x3': cv2.GaussianBlur(frame, (3, 3), 0),
        'gaussianblur_55x3': cv2.GaussianBlur(frame, (55, 3), 0),
        'medianblur_3x3': cv2.medianBlur(frame, 3),
        'laplace': cv2.normalize(cv2.Laplacian(gray, ddepth, ksize=3),
                                 None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
        'sobelx': cv2.normalize(cv2.Sobel(gray, ddepth, 1, 0, ksize=3),
                                None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
        'sobely': cv2.normalize(cv2.Sobel(gray, ddepth, 0, 1, ksize=3),
                                None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
        'dilate': cv2.dilate(frame, dilate_kernel, iterations=1),
        'erode': cv2.erode(frame, erode_kernel, iterations=1),
        'binaryotsu': cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        'eqhist': cv2.equalizeHist(gray),
        'canny': cv2.Canny(frame, canny_lower, canny_upper)
    }

    # Calculate FPS
    new_tick = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (new_tick - prev_tick)
    prev_tick = new_tick

    # Annotate images
    fps_display = f"{fps:.1f} FPS"
    cv2.putText(image['original'], fps_display, (10, 70), font, 2, (100, 255, 0), 3, cv2.LINE_AA)
    for key in image.keys():
        cv2.putText(image[key], key, (80, 450), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

    collage = stack_images(list(image.values()), cols=3, scale=0.35)
    cv2.imshow('main', collage)

    if FLAG_SAVE_VIDEO and writer is not None:
        # Resize collage to match writer frame size
        frame_to_write = cv2.resize(collage, (840, 480))
        writer.write(frame_to_write)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cam.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()

