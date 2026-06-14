# AI-Driven Noise Reduction

AI-based automatic noise classification and adaptive image/video denoising system with an additional SAR denoising module.

This repository contains the source code, sample inputs, trained checkpoints, and utility scripts used for the Capstone Project final submission.

## Requirements

- Python 3.10 or 3.11 is recommended.
- The commands below should be run from the project root directory.
- For normal image/video testing, the files under `models/` must remain in their current paths.
- SAR scripts require `rasterio`. If `rasterio` installation fails on a machine, the RGB image/video pipeline can still be tested without running SAR scripts.

## Installation

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If Python 3.11 is not installed, use the available Python launcher version, for example:

```powershell
py -m venv .venv
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick Test

Run the end-to-end image pipeline on a sample image:

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe -m src.main --image data\samples\gaussian.png
```

### Linux / macOS

```bash
python -m src.main --image data/samples/gaussian.png
```

Expected behavior:

- The classifier predicts one of: `gaussian`, `salt_pepper`, `speckle`, `periodic`.
- The matching denoiser checkpoint is selected automatically.
- A denoised image is written to `outputs/<input_name>_denoised.png`.

Example output format:

```text
Noise Reduction Result
Predicted noise type : gaussian
Classifier confidence: 99.00%
Selected model type  : UNet
Selected denoiser    : models/denoisers/gaussian/gaussian_unet_best.pt
Output path          : outputs/gaussian_denoised.png
```

The exact predicted label and confidence may vary depending on the input image.

## Streamlit Interface

The main UI supports image upload and video upload.

### Windows PowerShell

```powershell
.\.venv\Scripts\streamlit.exe run src\web\app.py
```

Alternative:

```powershell
.\.venv\Scripts\python.exe scripts\apps\run_ui.py
```

### Linux / macOS

```bash
streamlit run src/web/app.py
```

After the command starts, open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Main Project Structure

```text
src/
  architectures/      Model architectures: DnCNN, U-Net, periodic dual-branch U-Net
  inference/          Classifier and denoiser checkpoint loading
  models/             Additional model implementations, including periodic NAFNet
  noise_classifier/   Noise classifier dataset, transforms, and helper code
  pipeline/           End-to-end automatic classification and denoising pipeline
  preprocessing/      Periodic FFT filtering utilities
  metrics/            PSNR and SSIM helpers
  sar/                SAR dataset/filter/model helper modules
  video/              Video frame denoising pipeline
  web/                Streamlit application

scripts/
  apps/               App launch wrappers
  dataset/            Dataset download and synthetic-noise generation
  evaluation/         Evaluation and testing scripts
  sar/                SAR preprocessing, training, inference, and evaluation scripts
  training/           Training scripts for classifier and denoisers
  video/              Video utility scripts

models/
  classifiers/        Trained noise classifier checkpoint
  denoisers/          Trained denoiser checkpoints

data/
  samples/            Small sample images for testing
  synthetic/          Synthetic noisy datasets
  clean/              Clean reference images
```

## Active Checkpoints

The default runtime pipeline expects these checkpoint files:

```text
models/classifiers/noise_classifier_best.pt
models/denoisers/gaussian/gaussian_unet_best.pt
models/denoisers/salt_pepper/salt_pepper_unet_best.pt
models/denoisers/speckle/speckle_unet_residual_hybrid_best.pt
models/denoisers/periodic/periodic_fft_guided_nafnet_best.pt
models/denoisers/sar/best_noise_map_sar_model.pth
```

Do not rename or move these files unless the corresponding paths in `src/pipeline/run_pipeline.py` are updated.

## Archived Model Checkpoints

Additional archived checkpoints from earlier experiments and alternative model variants are available on Google Drive:

```text
https://drive.google.com/drive/folders/1aS0zhJi3q_7pA_lezm6jyXD5BsmMv9Tx?usp=sharing
```

These archived files are optional and are not required for the standard quick test. See `ARCHIVE_MODELS_GUIDE.txt` for the contents and purpose of each archived model group.

## Useful Commands

Evaluate the noise classifier:

```powershell
.\.venv\Scripts\python.exe scripts\evaluation\eval_noise_classifier.py --checkpoint models\classifiers\noise_classifier_best.pt --data_dir data\synthetic --device cpu
```

Run Gaussian denoiser evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluation\test_gaussian_unet.py --device cpu
```

Generate synthetic datasets from clean BSD images:

```powershell
.\.venv\Scripts\python.exe scripts\dataset\generate_synthetic_dataset.py
```

Process a video:

```powershell
.\.venv\Scripts\python.exe scripts\video\process_video_denoising.py --video data\videos\demo_1_yatay.mp4 --output outputs\demo_1_yatay_denoised.mp4
```

## Troubleshooting

If imports fail, make sure dependencies were installed into the active virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If checkpoint errors occur, verify that the `models/` directory was included in the submitted archive and that the checkpoint paths match the list above.

If output files are not created, make sure the project is not opened from a read-only location. The pipeline writes results to `outputs/`.

If CUDA is unavailable, the code automatically falls back to CPU. CPU inference can be slower, especially for video and SAR processing.

If `rasterio` fails to install on Windows, test the main RGB image/video system first. `rasterio` is only required for SAR ENVI/TIFF workflows.
