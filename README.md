# Real-Time CCTV Analysis with Object Detection and Analytics

Real-Time CCTV Analysis with Object Detection and Analytics is a powerful software application that enables intelligent analysis of CCTV footage in real-time. The program utilizes computer vision techniques and the OpenCV library to process video frames, detect objects of interest, and provide valuable insights through advanced analytics features.

## Features

- **Real-time object detection**: The program uses state-of-the-art object detection models, including face detection and pedestrian detection, to accurately identify and track objects within the CCTV video feed.
- **Multi-threaded processing**: It employs multi-threading to ensure smooth and responsive analysis, enabling simultaneous video capture and object detection.
- **Flexible and customizable**: The program allows for the integration of additional object detection models, such as vehicle detection, for a wider range of applications and requirements.
- **Analytics capabilities**: Real-time statistics, such as object counts and tracking, provide valuable insights for efficient monitoring, surveillance, and decision-making.
- **User-friendly interface**: The program provides a user-friendly interface that displays the processed video feed with overlaid bounding boxes around detected objects.

## Getting Started

### Prerequisites

- Python 3.x
- OpenCV library

### Installation

1. Clone the repository:

```bash
https://github.com/captain-n3m0/Watchman.git
```

2. Install the required dependencies:

```bash
pip install opencv-python
```

3. Download the pre-trained object detection models and place them in the appropriate directory:

- Haar cascade classifier for face detection: [Download here](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
- Other object detection models: (optional)

### Usage

1. Run the main Python script:

```bash
python main.py
```

2. The program will capture video frames from the default camera or a specified video file and perform object detection in real-time. Detected objects will be displayed with bounding boxes on the video feed.

3. Press 'q' to exit the program.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug fixes, please open an issue or submit a pull request.

## License

This project is licensed under the [GNU General Public License](LICENSE).

## Acknowledgments

- The OpenCV project for providing the Haar cascade classifiers and other pre-trained models.
- The developers and contributors of the OpenCV library for their invaluable work in computer vision and image processing.

## Contact

For any questions, comments, or collaborations, please feel free to reach out to us at [debjitnaskar@icloud.com](mailto:debjitnaskar@icloud.com).
