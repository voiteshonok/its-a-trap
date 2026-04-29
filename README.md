## Video picker (PyQt + uv)
 
Simple PyQt app that lets you pick a video file, choose **batch size** and **confidence**, then run MegaDetector and write results to `./output.json`.
 
 ### Setup (uv)
 
 Create a virtual environment and install deps:
 
 ```bash
 uv venv
 uv pip install -e .
 ```
 
 ### Run
 
 ```bash
 uv run video-picker
 ```
 
 (Alternative)
 
 ```bash
 uv run python -m video_picker.app
 ```
 

 to run the script 
 ```bash
 uv run python srctips/run_md_over_data_frames.py -b 8 -c 0.0
 ```
