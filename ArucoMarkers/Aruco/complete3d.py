import cv2
import cv2.aruco as aruco
import numpy as np
import pywavefront

def load_mtl_file(mtl_path):
    materials = {}
    current_material = None
    with open(mtl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('newmtl '):
                current_material = line.split()[1]
            elif line.startswith('Kd ') and current_material:

                r, g, b = map(float, line.split()[1:4])
                materials[current_material] = (int(r * 255), int(g * 255), int(b * 255))
    print(f"Loaded materials: {materials}")
    return materials

def load_obj_model(obj_path, mtl_path):
    try:

        materials = load_mtl_file(mtl_path)

        scene = pywavefront.Wavefront(obj_path, collect_faces=True, parse=True)
        vertices = []
        faces = []
        face_materials = []
        vertex_offset = 0

        current_material = None
        for mesh_name, mesh in scene.meshes.items():
            mesh_vertices = np.array(mesh.vertices, dtype=np.float32)
            vertices.extend(mesh_vertices)
            mesh_faces = [list(f) for f in mesh.faces]

            adjusted_faces = [[i + vertex_offset for i in face] for face in mesh_faces]
            faces.extend(adjusted_faces)

            face_materials.extend([current_material] * len(mesh_faces))
            vertex_offset += len(mesh_vertices)
        vertices = np.array(vertices, dtype=np.float32)

        current_material = None
        face_idx = 0
        with open(obj_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('usemtl '):
                    current_material = line.split()[1]
                elif line.startswith('f '):
                    if face_idx < len(face_materials):
                        face_materials[face_idx] = current_material
                        face_idx += 1
        print(f"Loaded {len(vertices)} vertices and {len(faces)} faces using pywavefront.")
        print(f"First few face indices: {faces[:5]}")
        print(f"Face materials: {face_materials[:5]}")
        return vertices, faces, face_materials, materials
    except Exception as e:
        print(f"pywavefront error: {e}. Attempting manual load...")

        vertices = []
        faces = []
        face_materials = []
        vertex_offset = 0
        current_vertices = []
        current_material = None
        with open(obj_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    coords = list(map(float, line.split()[1:4]))
                    current_vertices.append(coords)
                elif line.startswith('f '):
                    face = [int(x.split('/')[0]) - 1 for x in line.strip().split()[1:]]
                    faces.append([i + vertex_offset for i in face])
                    face_materials.append(current_material)
                elif line.startswith('o '):
                    if current_vertices:
                        vertices.extend(current_vertices)
                        vertex_offset += len(current_vertices)
                        current_vertices = []
                elif line.startswith('usemtl '):
                    current_material = line.split()[1]
            if current_vertices:
                vertices.extend(current_vertices)
        if vertices and faces:
            vertices = np.array(vertices, dtype=np.float32)
            print(f"Loaded {len(vertices)} vertices and {len(faces)} faces manually.")
            print(f"First few face indices: {faces[:5]}")
            print(f"Face materials: {face_materials[:5]}")
            return vertices, faces, face_materials, materials
        else:
            print("Failed to load vertices or faces manually.")
            return None, None, None, None

def draw_model(frame, vertices, faces, face_materials, materials, rvec, tvec, camera_matrix, dist_coeffs, solid=False):

    # match the marker size
    scale_factor = 0.01
    scaled_vertices = vertices * scale_factor

    # Convert vector to Matrix
    rmat, _ = cv2.Rodrigues(rvec)
    # Ensure tvec is shaped correctly (3,) for broadcasting
    tvec = tvec.ravel()  # Flatten to (3,)
    # Transform vertices to camera space
    transformed_vertices = np.dot(scaled_vertices, rmat.T) + tvec


    img_points, _ = cv2.projectPoints(scaled_vertices, rvec, tvec, camera_matrix, dist_coeffs)
    img_points = np.int32(img_points).reshape(-1, 2)

    # Debug: Print projected point ranges
    frame_height, frame_width = frame.shape[:2]
    print(f"Min X: {img_points[:, 0].min()}, Max X: {img_points[:, 0].max()}")
    print(f"Min Y: {img_points[:, 1].min()}, Max Y: {img_points[:, 1].max()}")

    # Compute face depths (average Z in camera space) for sorting
    face_depths = []
    for i, face in enumerate(faces):
        if all(0 <= idx < len(transformed_vertices) for idx in face):
            avg_z = np.mean([transformed_vertices[idx][2] for idx in face])
            face_depths.append((i, avg_z))
    # Sort faces by depth (descending, so farthest faces are drawn first)
    face_depths.sort(key=lambda x: x[1], reverse=True)

    # Draw faces
    visible_faces = 0
    for face_idx, _ in face_depths:
        face = faces[face_idx]
        pts = [img_points[i] for i in face if 0 <= i < len(img_points)]
        if len(pts) == 3:  # Ensure valid triangle
            all_in_frame = all(0 <= x < frame_width and 0 <= y < frame_height for x, y in pts)
            if all_in_frame:
                # Get material color
                material_name = face_materials[face_idx]
                color = materials.get(material_name, (0, 255, 0))  # Default to green if material not found
                if solid:
                    cv2.fillPoly(frame, [np.array(pts)], color)
                else:
                    cv2.polylines(frame, [np.array(pts)], True, color, 5)
                visible_faces += 1
    print(f"Visible faces: {visible_faces} out of {len(faces)}")

    return frame

def main():
    # Camera parameters (replace with actual calibration)
    camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # Load .obj and .mtl files
    obj_path = r'E:\ALL PYTHON PROJECTS\ArucoMarkers\Aruco\Solid\cube.obj'
    mtl_path = r'E:\ALL PYTHON PROJECTS\ArucoMarkers\Aruco\Solid\cube.mtl'
    vertices, faces, face_materials, materials = load_obj_model(obj_path, mtl_path)
    if vertices is None or faces is None:
        print("Failed to load .obj file. Exiting.")
        return

    # Choose rendering style
    render_solid = True  # Set to True for solid faces with material colors

    # Initialize ArUco dictionary and detector parameters
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    parameters = aruco.DetectorParameters()

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # Process detected markers
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            marker_size = 0.01  # Marker size in meters
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)

            for i, corner in enumerate(corners):
                print(f"Detected marker ID: {ids[i][0]}")
                frame = draw_model(frame, vertices, faces, face_materials, materials, rvecs[i], tvecs[i], camera_matrix, dist_coeffs, render_solid)

        # Display the frame
        cv2.imshow("Sample Augmented Reality through OBJ", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()