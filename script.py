import cv2
import sys
import os
import subprocess
import pandas as pd
import base64
import google.generativeai as genai

# ---------- Helpers ----------

def extract_frames(video_path, output_folder, interval=0.5):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)

    frame_count, saved_count = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            filename = os.path.join(output_folder, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f"[Frames] Extracted {saved_count} frames at {interval}s intervals")

def run_openface(openface_bin, frames_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [openface_bin, "-fdir", frames_dir, "-out_dir", output_dir]
    subprocess.run(cmd, check=True)
    print(f"[OpenFace] Processing complete! Results in: {output_dir}")

def find_peak_frame(openface_csv):
    print("Reading CSV from:", openface_csv)
    df = pd.read_csv(openface_csv)
    print("CSV Columns:", list(df.columns)[:20]) 
    df.columns = df.columns.str.strip() 
    df = df[df['confidence'] > 0.8]   # filter low-confidence
    au_cols = [c for c in df.columns if "_r" in c]
    df["expression_strength"] = df[au_cols].sum(axis=1)
    peak_row = df.loc[df["expression_strength"].idxmax()]
    return int(peak_row["frame"])

def extract_frame(video_path, frame_number, output_path):

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not extract frame {frame_number}")
    cv2.imwrite(output_path, frame)
    return output_path

def analyze_with_gemini(image_path, api_key, prompt=None):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
            
    # If analyzing an image
    if image_path is not None:
        with open(image_path, "rb") as f:
            image_data = {
                "mime_type": "image/jpeg",
                "data": f.read()
            }
        response = model.generate_content([prompt or "Analyze this image", image_data])
    else:
        # Pure text prompt (no image)
        text_model = genai.GenerativeModel("gemini-1.5-flash")
        response = text_model.generate_content(prompt or "Summarize")

    return response.text

# ---------- Main pipeline ----------

def process_video(video_path, openface_bin, interval=0.5, base_output="outputs"):
    # Setup output dirs
    api_key = "AIzaSyABLSxBAXLzn-6PZaZAoY2nf-L_VRwelcE"
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(base_output, video_name)
    frames_dir = os.path.join(output_folder, "frames")
    openface_out = os.path.join(output_folder, "openface")
    os.makedirs(output_folder, exist_ok=True)

    print(f"▶️ Processing video: {video_name}")

    # Step 1: Extract frames
    extract_frames(video_path, frames_dir, interval)

    # Step 2: Run OpenFace
    run_openface(openface_bin, frames_dir, openface_out)

    # Step 3: Find peak frame from OpenFace CSV
    csv_path = os.path.join(openface_out, "frames.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"frames.csv not found in {openface_out}")
    peak_frame = find_peak_frame(csv_path)
    print(f"[Peak Frame] Found frame #{peak_frame}")
    
    # csv_path = r"C:\Users\Avinash\Desktop\openFace\outputs\input_video\openface\frames.csv"

    # Step 4: Extract the peak frame
    peak_frame_path = os.path.join(output_folder, "peak_frame.jpg")
    extract_frame(video_path, peak_frame, peak_frame_path)

    # Step 5: Analyze with Gemini
    gemini_result = analyze_with_gemini(peak_frame_path, api_key)

    # Save Gemini result
    result_file = os.path.join(output_folder, "gemini_result.txt")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(gemini_result)

    print(f"[Done] Results saved in {output_folder}")
    return {"peak_frame": peak_frame_path, "gemini_result": gemini_result}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py input_video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    # Path to OpenFace binary
    openface_bin = r"OpenFace_2.2.0_win_x64\FeatureExtraction.exe"

    results = process_video(video_path, openface_bin)

    print("\n✅ Pipeline complete!")
    print("Peak frame saved at:", results["peak_frame"])
    print("Gemini result:\n", results["gemini_result"])