# Augmented Reality with ArUco Markers and 3D Model Rendering

This project demonstrates an augmented reality (AR) application using Python, OpenCV, and ArUco markers to overlay a 3D model onto a webcam feed. 
The model is loaded from `.obj` and `.mtl` files and rendered in real-time based on detected ArUco markers.

## Features
- Detects ArUco markers in a webcam feed using OpenCV's ArUco module.
- Loads and parses `.obj` and `.mtl` files to render a 3D model with material colors.
- Projects the 3D model onto the detected marker's position in the video feed.
- Supports both wireframe and solid rendering modes.
- Handles multiple meshes and materials from the `.obj` file.

## Requirements
- Python 3.6+
- OpenCV (`opencv-contrib-python`) for ArUco marker detection and image processing
- NumPy for numerical computations
- PyWavefront for parsing `.obj` and `.mtl` files
- A webcam for capturing video
- ArUco markers (DICT_4X4_1000 dictionary) printed and visible to the webcam
- Sample `.obj` and `.mtl` files (e.g., `cube.obj` and `cube.mtl`)

----------------------------------------------------------------------------------------------------------------------------------------------------------------

USAGE

1. Ensure you have a webcam connected and ArUco markers printed (DICT_4X4_50 dictionary).
2. Place your .obj and .mtl files in the appropriate directory (e.g., Solid/cube.obj and Solid/cube.mtl).
3. Update the obj_path and mtl_path variables in complete3d.py to point to your .obj and .mtl files:
	obj_path = r'path/to/your/cube.obj'
 	mtl_path = r'path/to/your/cube.mtl'
4. Run the Script 
	```bash 
 	python complete3d.py

5. Point your webcam at the ArUco marker. The 3D model should appear overlaid on the marker.
6. Press q to exit the application.


----------------------------------------------------------------------------------------------------------------------------------------------------------------
	
MIT License

Copyright (c) [year] [author]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software... to deal in the Software without restriction...

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND...
