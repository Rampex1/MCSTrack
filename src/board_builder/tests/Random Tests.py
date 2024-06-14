import cv2
import cv2.aruco as aruco
import numpy as np

def read_camera_params_from_command_line(parser):
    cam_matrix = np.loadtxt(parser.get('cam_matrix'), delimiter=',')
    dist_coeffs = np.loadtxt(parser.get('dist_coeffs'), delimiter=',')
    return cam_matrix, dist_coeffs

def read_dictionary_from_command_line(parser):
    dictionary_id = parser.get('dict')
    return aruco.Dictionary_get(dictionary_id)

def read_detector_params_from_command_line(parser):
    detector_params = aruco.DetectorParameters_create()
    # Add additional detector parameters from the command line parser if needed
    return detector_params

markers_x = args.w
markers_y = args.h
marker_length = args.l
marker_separation = args.s
show_rejected = args.r
refind_strategy = args.rs
cam_id = args.ci
video = args.v if 'v' in args else ''

cam_matrix, dist_coeffs = read_camera_params_from_command_line(args)
dictionary = read_dictionary_from_command_line(args)
detector_params = read_detector_params_from_command_line(args)

detector = aruco.ArucoDetector(dictionary, detector_params)
input_video = cv2.VideoCapture(video if video else cam_id)
wait_time = 0 if video else 10

axis_length = 0.5 * (min(markers_x, markers_y) * (marker_length + marker_separation) + marker_separation)

# Create GridBoard object
board = aruco.GridBoard_create(markers_x, markers_y, marker_length, marker_separation, dictionary)

total_time = 0
total_iterations = 0

while input_video.grab():
    ret, image = input_video.retrieve()
    if not ret:
        break

    image_copy = image.copy()
    tick = cv2.getTickCount()

    ids = []
    corners = []
    rejected = []
    rvec, tvec = None, None

    # Detect markers
    detector.detectMarkers(image, corners, ids, rejected)

    # Refind strategy to detect more markers
    if refind_strategy:
        detector.refineDetectedMarkers(image, board, corners, ids, rejected, cam_matrix, dist_coeffs)

    # Estimate board pose
    markers_of_board_detected = 0
    if ids:
        obj_points, img_points = board.matchImagePoints(corners, ids)
        if len(obj_points) > 0:
            _, rvec, tvec = cv2.solvePnP(obj_points, img_points, cam_matrix, dist_coeffs)
            markers_of_board_detected = int(obj_points.total() / 4)

    current_time = (cv2.getTickCount() - tick) / cv2.getTickFrequency()
    total_time += current_time
    total_iterations += 1
    if total_iterations % 30 == 0:
        print(f"Detection Time = {current_time * 1000:.2f} ms (Mean = {1000 * total_time / total_iterations:.2f} ms)")

    # Draw results
    if ids:
        aruco.drawDetectedMarkers(image_copy, corners, ids)

    if show_rejected and rejected:
        aruco.drawDetectedMarkers(image_copy, rejected, np.array([]), (100, 0, 255))

    if markers_of_board_detected > 0 and rvec is not None and tvec is not None:
        cv2.drawFrameAxes(image_copy, cam_matrix, dist_coeffs, rvec, tvec, axis_length)

    cv2.imshow("out", image_copy)
    key = cv2.waitKey(wait_time)
    if key == 27:  # ESC key to break
        break

input_video.release()
cv2.destroyAllWindows()
