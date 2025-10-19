from script import process_video, analyze_with_gemini
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def main(clips_dir, openface_bin, api_key, output_file="outputs/overall_summary.txt"):
    # collect mp4 clips
    video_files = [f for f in os.listdir(clips_dir) if f.endswith(".mp4")]
    video_files.sort()  # keep them in order: clip_0.mp4, clip_1.mp4, ...

    all_results = []

    for video in video_files:
        video_path = os.path.join(clips_dir, video)
        print(f"\n Processing {video} ...")
        
        results = process_video(video_path, openface_bin)  # your pipeline
        
        all_results.append({
            "clip": video,
            "peak_frame": results["peak_frame"],
            "summary": results["gemini_result"]
        })

    # write results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== Individual Summaries ===\n\n")
        for res in all_results:
            f.write(f"{res['clip']} (peak frame: {res['peak_frame']}):\n{res['summary']}\n\n")
        
        f.write("\n=== Combined Overview ===\n")
        # send all summaries to Gemini/OpenAI again for a scene-level summary
        combined_prompt = "Summarize the following scene based on clip summaries:\n\n"
        combined_prompt += "\n\n".join([f"{r['clip']}: {r['summary']}" for r in all_results])
        
        overall_summary = analyze_with_gemini(None, api_key, prompt=combined_prompt)
        f.write(overall_summary)

if __name__ == "__main__":
    clips_dir = r"Harry Potter and the Sorcerer's Stone (2001) - Harry's Warning"
    openface_bin = r"OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
    api_key = "AIzaSyABLSxBAXLzn-6PZaZAoY2nf-L_VRwelcE"

    main(clips_dir, openface_bin, api_key)

