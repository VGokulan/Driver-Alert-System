# SafeDrive: Cell Phone Detection for Road Safety

This project, SafeDrive, aims to contribute to the fight against distracted driving by detecting cell phone usage in video streams. It leverages the power of computer vision and machine learning to identify phones in real-time, potentially paving the way for advanced driver monitoring systems (future implementation).

## Key Features
SafeDrive offers several functionalities to enhance road safety by detecting cell phone usage:

Real-time Cell Phone Detection: Utilizes a pre-trained object detection model to identify cell phones within video streams.
Visual Feedback: Detected phones are highlighted with bounding boxes, accompanied by confidence scores indicating the detection accuracy.
Driver Monitoring Potential: This technology can serve as a foundation for developing advanced driver monitoring systems in the future.
Optional SMS Alerts (Twilio Integration): (Commented out in current implementation) With Twilio integration, the system can send SMS alerts when phone usage is detected for a prolonged period (potential functionality).

## Installation and Usage
To get started with SafeDrive, follow these steps:

### 1. Dependencies:

This project relies on several Python libraries:

1. OpenCV-Python (for computer vision tasks)
2. TensorFlow (for machine learning and object detection)
3. NumPy (for numerical computations)
4. Twilio (optional, for SMS alerts - currently commented out in the script)

Installation:

There are two main options for installing these dependencies:

Using pip: If you have Python with pip package manager installed, you can run the following command in your terminal to install all required libraries:
```ruby
pip install opencv-python tensorflow numpy twilio
```
> [!CAUTION]
>Using conda (Anaconda/Miniconda): If you're using Anaconda or Miniconda, you can create a new environment with the required libraries and activate it before running the script. Refer to the official >documentation for creating environments: https://conda.io/activation

### 2. Downloading the Pre-trained Model:

(Currently commented out in the script)

The script utilizes a pre-trained object detection model for cell phone detection. By default, the script downloads the model automatically (commented out section). However, you can download it manually if needed.

Download Link: [TensorFlow Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

### 3. Running the Script:

Once you have the dependencies installed and (potentially) downloaded the model manually, navigate to the project directory in your terminal and run the script with the desired video path as an argument:
```ruby
python your_script_name.py path/to/your/video.mp4
```
This will process the video at the specified path and display the results with bounding boxes and confidence scores for detected cell phones.

Note: If you plan to enable the SMS alert functionality using Twilio, you'll need to create a Twilio account and obtain your account credentials (SID and Auth Token). These credentials will need to be added to the script (currently commented out sections) and your environment variables before running the script.

## Future Enhancements
SafeDrive has the potential to be further developed and expanded upon in several exciting ways:

1. Custom Phone Detection Model: Training a custom deep learning model specifically optimized for detecting cell phones in various orientations and positions within a car environment. This could improve accuracy and reliability compared to the current pre-trained model.
2. Real-time In-vehicle Monitoring: Integrating SafeDrive with a driver-facing camera would enable real-time monitoring within a vehicle. This could provide continuous feedback and potentially trigger alerts for distracted driving behavior.
3. Multi-distraction Detection: Expanding the system beyond cell phone detection to encompass other forms of driver distraction. This could involve incorporating facial recognition for drowsiness detection, analyzing steering wheel movements, or detecting objects held near the driver's face.
By implementing these enhancements, SafeDrive could evolve into a comprehensive driver monitoring system, promoting safer roads and reducing distracted driving accidents.
