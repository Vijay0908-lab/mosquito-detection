# Mosquito and Vector Detection Model - Mosha

Developed for Tula's Hackverse 2024, a hackathon organized by Tula Institute, Dehradun, India.
Lead AI Developer: Ujjwal Paliwal

---

## Project Overview
Mosha is an AI-powered mosquito and vector detection system built using YOLOv8. The model detects mosquitoes and other vectors in images and videos, helping in vector surveillance and control efforts. This project was developed as part of a hackathon challenge to create impactful solutions for public health.

## Features
- Detects mosquitoes and vectors in images and videos
- Pre-trained YOLOv8 model included for instant inference
- Easy-to-use scripts for both image and video analysis
- Sample data and results provided

## Quick Start

1. **Clone the repository**
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
3. **Run inference on a video:**
   - Place your video in the `videos/` folder (e.g., `videos/mos0.mp4`)
   - Run:
   ```
   python video.py
   ```
   - The annotated result will be saved as `result.mp4`
4. **Run inference on test images:**
   - Place images in `test/images/`
   - Run:
   ```
   python test.py
   ```
   - Results will be saved in the current directory

## Project Structure
- `video.py` : Run detection on videos
- `test.py`  : Run detection on images
- `train.py` : (Optional) Retrain the model if needed
- `runs/`    : Contains model weights (e.g., `best.pt`)
- `videos/`  : Sample and input videos
- `test/`, `train/`, `valid/` : Data folders

## Notes
- The repository includes the pre-trained model weights for immediate use.
- No need to retrain unless you want to improve or adapt the model.
- For best results, use clear images/videos with visible vectors.

## Credits
- Developed during Tula's Hackverse 2024, Tula Institute, Dehradun, India.
- Lead AI Developer: Ujjwal Paliwal

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

