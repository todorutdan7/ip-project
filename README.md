# ip-project — Rubik's Cube Face Detection

Image Processing project. The program detects a single face of a Rubik's Cube
in an image or a live webcam feed and recognizes the color of each of its nine
stickers, then shows a virtual reconstruction of the face.



## Documentation

The full project documentation (description, related work, architecture,
implementation, user manual, results and analysis) is in
[`documentation.pdf`](documentation.pdf).

## Source

- `OpenCVApplication.cpp` — main source file (the whole detector)
- `common.cpp`, `common.h`, `stdafx.*`, `targetver.h` — lab template helpers
- `OpenCVApplication.sln` / `OpenCVApplication.vcxproj` — Visual Studio project

## Building & running

1. Open `OpenCVApplication.sln` in Visual Studio.
2. Make sure OpenCV (the build the project was made with) is installed and the
   include/library paths and DLLs are configured. The OpenCV SDK itself is
   **not** committed to this repo because of its size.
3. Build and run. A menu appears:
   - `1` — use the webcam (hold a cube face to the camera; `ESC`/`q` to quit)
   - `2` — choose an image; a file dialog opens so you can test on the sample
     images in the [`Images/`](Images) folder or your own.

## Sample images

The [`Images/`](Images) folder contains sample captures that can be loaded with
option `2` to test the detector without a webcam.
