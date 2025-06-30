# Real-Time Multimodal System for Nonverbal Communication Through Vision-Based Methods

This repository contains the official implementation of the paper:

**"Real-Time Multimodal System for Nonverbal Communication Through Vision-Based Methods"**.

This project presents a real-time, vision-based coaching system designed for virtual interviews. It integrates multiple modalities of nonverbal behavior analysis and provides live feedback based on user behavior.

---

## Technologies Used

- Python 3.8+
- [MediaPipe](https://chuoling.github.io/mediapipe/) (Pose, Face Mesh, Iris, Hands)
- [EmoNet](https://github.com/face-analysis/emonet)
- [face-alignment](https://github.com/1adrianb/face-alignment)
- OpenCV (cv2)
- PyTorch
- OpenAI API (ChatGPT integration)

---
## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/IsiPaz/CoachingAI.git
cd CoachingAI
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate       # On macOS/Linux
venv\Scripts\activate          # On Windows
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the System

To launch the real-time multimodal system using your webcam:

```bash
cd src
python main.py
```

## Optional Command-Line Arguments

| Argument             | Type   | Default  | Description                                                              |
|----------------------|--------|----------|--------------------------------------------------------------------------|
| `--n_expression`     | int    | 8        | Number of emotion classes (5 or 8)                                       |
| `--device`           | str    | cuda:0   | Device to run models on (`cuda:0`, `cpu`, etc.)                          |
| `--image_size`       | int    | 256      | Image input size for EmoNet                                              |
| `--camera_id`        | int    | 0        | Webcam device ID                                                         |
| `--target_fps`       | int    | 30       | Target frame rate                                                        |
| `--show_fps`         | flag   | False    | Show real-time FPS overlay                                               |
| `--show_circumplex`  | flag   | False    | Show valence-arousal (circumplex) plot                                   |
| `--debug`            | flag   | False    | Enable verbose output and detailed visual overlays                       |
| `--openai_api_key`   | str    | None     | OpenAI API key to enable ChatGPT feedback                                |

## Example Usage

## Credits and Acknowledgments

This project builds upon the work of several open-source tools and research efforts. We would like to acknowledge the following:

- The **EmoNet** model used for facial emotion recognition is based on the work by Toisoul et al., published in *Nature Machine Intelligence*:

> Toisoul, A., Kossaifi, J., Bulat, A., Tzimiropoulos, G., & Pantic, M. (2021).  
> *Estimation of continuous valence and arousal levels from faces in naturalistic conditions*.  
> Nature Machine Intelligence.  
> [https://www.nature.com/articles/s42256-020-00280-0](https://www.nature.com/articles/s42256-020-00280-0)

- **Face Detection (SFD)** — The EmoNet pipeline uses the SFD face detector introduced by Bulat & Tzimiropoulos:

> Bulat, A., & Tzimiropoulos, G. (2017).  
> *How far are we from solving the 2D & 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)*.  
> *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.

- **MediaPipe** — Real-time face, pose, hand, and iris tracking is powered by [MediaPipe](https://github.com/google-ai-edge/mediapipe), an open-source framework developed by Google for building cross-platform multimodal applied ML pipelines.

Please cite these tools appropriately if you use this system in your own research or development.

## Contact
For questions, suggestions, or collaborations, please contact:
- Nicolás Torres - [nicolas.torresr@usm.cl]
- Isidora Ubilla — [isidora.ubilla17@gmail.com]
