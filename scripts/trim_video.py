import os
import sys
import subprocess
import time

def get_video_duration(video_path):
    """Returns the duration of the video in seconds using ffprobe."""
    try:
        result = subprocess.run(
             ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
            , stdout=subprocess.PIPE
            , stderr=subprocess.PIPE
            , text=True
            , check=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting duration for {video_path}: {e}")
        return None

def trim_video(input_file: str, front_trim_sec: float, last_trim_sec: float) -> bool:
    """
    Trims the video by the specified amount of seconds from the front and back.
    Replaces the original video with the trimmed version.
    Returns True if successful, False otherwise.
    """
    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' does not exist.")
        return False
        
    if front_trim_sec == 0 and last_trim_sec == 0:
        print("Nothing to trim. Exiting.")
        return True

    print("\nAnalyzing video duration...")
    duration_sec = get_video_duration(input_file)
    
    if duration_sec is None:
        print("Could not determine video duration. Cannot proceed.")
        return False
        
    print(f"Original video duration: {duration_sec:.2f} seconds ({duration_sec/60:.2f} minutes).")
    
    start_time = front_trim_sec
    end_time = duration_sec - last_trim_sec
    
    if start_time >= end_time or start_time >= duration_sec:
        print("Error: Trim amounts are too large. The resulting video would have a negative or zero duration.")
        return False
        
    # Duration of the new clipped video
    duration_to_keep = end_time - start_time
    
    # Generate temporary output filename
    base_name, ext = os.path.splitext(input_file)
    dir_name = os.path.dirname(input_file)
    temp_output_file = os.path.join(dir_name, f"{os.path.basename(base_name)}_temp_trim{ext}")
    
    print(f"\nTrimming video...")
    print(f"Start time:  {start_time:.2f}s")
    print(f"End time:    {end_time:.2f}s")
    
    # Run ffmpeg command
    # using -ss before -i for fast seek, and -t for duration.
    # -c copy so we don't re-encode (super fast, but cuts at nearest keyframe)
    command = [
        "ffmpeg", 
        "-ss", str(start_time), 
        "-i", input_file, 
        "-t", str(duration_to_keep), 
        "-c:v", "copy",
        "-c:a", "copy",
        "-y",  # Overwrite output if it exists
        temp_output_file
    ]
    
    try:
        # Run command, pipe output to terminal so user can see progress
        subprocess.run(command, check=True)
        print(f"\n✅ Successfully generated trimmed video. Replacing original...")
        
        # Give OS a tiny moment to release file handles
        time.sleep(0.5)
        
        # Replace the original file with the new temporary file
        os.replace(temp_output_file, input_file)
        print(f"✅ Original file '{input_file}' has been successfully overwritten with the trimmed version.")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during ffmpeg processing: {e}")
        if os.path.exists(temp_output_file):
            try:
                os.remove(temp_output_file)
            except:
                pass
        return False
    except FileNotFoundError:
        print("\n❌ Error: 'ffmpeg' command not found. Please ensure FFmpeg is installed and added to your system PATH.")
        return False
    except Exception as e:
        print(f"\n❌ Error replacing the original file: {e}")
        return False

def trim_video_exact(input_file: str, start_sec: float, end_sec: float) -> bool:
    """
    Trims the video using exact start and end timestamps.
    Replaces the original video with the trimmed version.
    Returns True if successful, False otherwise.
    """
    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' does not exist.")
        return False
        
    duration_to_keep = end_sec - start_sec
    if duration_to_keep <= 0:
        print("Error: End time must be greater than start time.")
        return False
        
    # Generate temporary output filename
    base_name, ext = os.path.splitext(input_file)
    dir_name = os.path.dirname(input_file)
    temp_output_file = os.path.join(dir_name, f"{os.path.basename(base_name)}_temp_trim{ext}")
    
    # Run ffmpeg command
    command = [
        "ffmpeg", 
        "-ss", str(start_sec), 
        "-i", input_file, 
        "-t", str(duration_to_keep), 
        "-c:v", "copy",
        "-c:a", "copy",
        "-y",
        temp_output_file
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
        os.replace(temp_output_file, input_file)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during ffmpeg processing: {e}")
        if os.path.exists(temp_output_file):
            try:
                os.remove(temp_output_file)
            except:
                pass
        return False
    except FileNotFoundError:
        print("\n❌ Error: 'ffmpeg' command not found. Please ensure FFmpeg is installed and added to your system PATH.")
        return False
    except Exception as e:
        print(f"\n❌ Error replacing the original file: {e}")
        return False

def main():
    print("-----------------------------------------------------")
    print("          Video Trimmer (Front and Last)             ")
    print("-----------------------------------------------------")
    print("Note: Fast trimming is used (-c copy), which cuts at keyframes.")
    print("      Requires FFmpeg to be installed on your system.\n")
    print("WARNING: This will overwrite the original video file!\n")
    
    # 1. Ask for input video file
    input_file = input("Enter the path to the video file (e.g., input.mp4): ").strip()
    
    # Remove quotes if the user dragged and dropped the file into terminal
    if input_file.startswith('"') and input_file.endswith('"'):
        input_file = input_file[1:-1]
        
    # 2. Ask how much to trim from the front (in minutes)
    try:
        front_trim_min = float(input("How many minutes to trim from the FRONT? (Enter 0 if none): "))
    except ValueError:
        print("Invalid input. Defaulting to 0.")
        front_trim_min = 0.0

    # 3. Ask how much to trim from the last part (in minutes)
    try:
        last_trim_min = float(input("How many minutes to trim from the LAST PART? (Enter 0 if none): "))
    except ValueError:
        print("Invalid input. Defaulting to 0.")
        last_trim_min = 0.0
        
    front_trim_sec = front_trim_min * 60
    last_trim_sec = last_trim_min * 60

    trim_video(input_file, front_trim_sec, last_trim_sec)

def run_exact_cli(argv):
    """
    Non-interactive entry point used by the GUI subprocess.
    Usage: trim_video.py exact <video_path> <start_sec> <end_sec>
    """
    if len(argv) != 3:
        print("Usage: trim_video.py exact <video_path> <start_sec> <end_sec>")
        sys.exit(2)

    video_path = argv[0]
    try:
        start_sec = float(argv[1])
        end_sec = float(argv[2])
    except ValueError:
        print("Error: start and end times must be numeric (seconds).")
        sys.exit(2)

    print(f"Trimming '{video_path}' to {start_sec:.2f}s -> {end_sec:.2f}s...")
    success = trim_video_exact(video_path, start_sec, end_sec)
    if success:
        print("✅ Trim completed successfully.")
        sys.exit(0)
    else:
        print("❌ Trim failed.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        # The GUI invokes: trim_video.py exact <video> <start> <end>
        if len(sys.argv) > 1 and sys.argv[1] == "exact":
            run_exact_cli(sys.argv[2:])
        else:
            main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
