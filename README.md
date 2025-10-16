!pip install moviepy groq

from moviepy.video.io.VideoFileClip import VideoFileClip
#define the paths
video_path="/videoplayback.mp4"
audio_path="output_audio.mp3"
#load and covert
clip=VideoFileClip(video_path)
clip.audio.write_audiofile(audio_path)
print("Audio Extracted:", audio_path)

import os
import json
from groq import Groq
os.environ["GROQ_API_KEY"]="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"
#initailize the groq client
client=Groq()
filename="output_audio.mp3"
#open the audio file
with open(filename,"rb") as file:
  transcription = client.audio.transcriptions.create(
      file=file,
      model="whisper-large-v3-turbo",
      response_format="verbose_json",
      #timestamp_granularities=["word","segment"],
      language="en",
      temperature="0.0"
  )
  print(transcription.text)
  
  print(json.dumps(transcription,indent=2,default=str))
  
  from moviepy.video.io.VideoFileClip import VideoFileClip
def extract_audio(video_path:str,audio_path:str) ->str:
  """
  Extracts audio from a video file and saves it as an MP3 file.
  Args:
    video_path: The path to the video file.
    audio_path: The path to save the audio file.
  Returns:
   str: The path to the saved audio file.
  """
  clip=VideoFileClip(video_path)
  clip.audio.
  
  write_audiofile(audio_path)
  print("Audio Extracted:", audio_path)
  return audio_pathvideo_path="/videoplayback.mp4"
audio_path="output_audio.mp3"
saved_audio_path=extract_audio(video_path,audio_path)

import os
from dotenv import load_dotenv
from groq import Groq
#load envirnoment variables
load_dotenv()
# GROQ_API_KEY="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" # Remove explicit key assignment here
def transcribe_audio(audio_path:str) ->str:
  """
  Transcribes an audio file using the Groq API.
  Args:
    audio_path: The path to the audio file.
  Returns:
    str:The transcribed text.
  """
  # client=Groq(api_key=GROQ_API_KEY) # Remove client initialization from here
  with open(audio_path,"rb") as file:
    transcription=client.audio.transcriptions.create( # Use the global client
        file=file,
        model="whisper-large-v3-turbo",
        prompt="Specify context or spelling",
        response_format="verbose_json",
        timestamp_granularities=["word","segment"],
        language="en",
        temperature="0.0"
    )
    print("Transcription completed.")
    return transcription.text
transcription_text=transcribe_audio(audio_path)
print(transcription_text)

pip install git+https://github.com/openai/whisper.git

import whisper

model = whisper.load_model("base")  # or "small", "medium", "large"
result = model.transcribe(audio_path) # Use the variable audio_path
transcription_text = result["text"]
print(transcription_text)

pip install transformers

from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_meeting(transcript):
    chunks = [transcript[i:i+1000] for i in range(0, len(transcript), 1000)]
    summaries = [summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return "\n".join(summaries)

summary_text = summarize_meeting(transcription_text)
print("📝 Summary:\n", summary_text)

def extract_action_items(summary):
    lines = summary.split(".")
    actions = [line.strip() for line in lines if any(kw in line.lower() for kw in ["will", "should", "must", "decided", "agreed", "assigned"])]
    return actions

action_items = extract_action_items(summary_text)
print("📌 Action Items:")
for item in action_items:
    print("-", item)
import gradio as gr
from moviepy.video.io.VideoFileClip import VideoFileClip
import whisper # Corrected import
from transformers import pipeline

# Load models
whisper_model = whisper.load_model("base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def process_file(file):
    # Extract audio
    clip = VideoFileClip(file)
    audio_path = "temp_audio.mp3"
    clip.audio.write_audiofile(audio_path, logger=None)

    # Transcribe
    result = whisper_model.transcribe(audio_path)
    transcription_text = result["text"]

    # Summarize
    chunks = [transcription_text[i:i+1000] for i in range(0, len(transcription_text), 1000)]
    summaries = [summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text'] for chunk in chunks]
    summary_text = "\n".join(summaries)

    # Extract action items
    lines = summary_text.split(".")
    actions = [line.strip() for line in lines if any(kw in line.lower() for kw in ["will", "should", "must", "decided", "agreed", "assigned"])]

    return transcription_text, summary_text, "\n".join(actions)

# Gradio Interface
demo = gr.Interface(
    fn=process_file,
    inputs=gr.File(type="filepath", label="Upload Audio/Video"),
    outputs=[
    gr.Textbox(label="Transcription", lines=15, max_lines=30),
        gr.Textbox(label="Summary", lines=10, max_lines=20),
        gr.Textbox(label="Action Items", lines=10, max_lines=20),
    ],
    title="🎥 Meeting Summarizer",
    description="Upload a video/audio file → Get transcript, summary & action items"
)

demo.launch()

def extract_action_items(summary):
    lines = summary.split(".")
    actions = [line.strip() for line in lines if any(kw in line.lower() for kw in ["will", "should", "must", "decided", "agreed", "assigned"])]
    return actions

action_items = extract_action_items(summary_text)

print("📌 Action Items:")
for item in action_items:
    print("-", item)


# ---------------------------
# ✅ Add transcription accuracy
# ---------------------------
!pip install jiwer

from jiwer import wer

def transcription_accuracy(reference_text, predicted_text):
    """
    Calculates transcription accuracy using Word Error Rate (WER).
    Args:
        reference_text (str): Ground truth transcript
        predicted_text (str): Model transcription
    Returns:
        float: accuracy percentage
    """
    error = wer(reference_text, predicted_text)
    accuracy = (1 - error) * 100
    return accuracy

# ✍️ Provide the TRUE transcript of your video here
reference_text = "Hello everyone, thank you guys for coming to our weekly student success meeting. And let's just get started. So I have our list of chronically absent students here and I've been noticing a troubling trend. A lot of students are skipping on Fridays. Does anyone have any idea what's going on? I've heard some of my mentees talking about how it's really hard to get out of bed on Fridays. It might be good if we did something like a pancake breakfast to encourage them to come. I think that's a great idea. Let's try that next week. It might also be because a lot of students have been getting sick now that it's getting colder outside. I've had a number of students come by my office with symptoms like sniffling and coughing. We should put up posters with tips for not getting sick since it's almost flu season. Like, you know, wash your hands after the bathroom. Stuff like that. I think that's a good idea and it'll be a good reminder for the teachers as well. So one other thing I wanted to talk about, there's a student I've noticed here, John Smith. He's missed seven days already and it's only November. Does anyone have an idea what's going on with him? I might be able to fill in the gaps there. I talked to John today and he's really stressed out. He's been dealing with helping his parents take care of his younger siblings during the day. It might actually be a good idea if he spoke to the guidance counselor a little bit. I can talk to John today if you want to send him to my office after you meet with him. It's a lot to deal with for middle schooler. Great thanks and I can help out with the family's childcare needs. I'll look for some free or low-cost resources in the community to share with John and he can share them with his family. Great, well some really good ideas here today. Thanks for coming and if no one has anything else I think we can wrap up."

predicted_text = transcription_text  # from Whisper/Groq

accuracy = transcription_accuracy(reference_text, predicted_text)
print(f"🎯 Transcription Accuracy: {accuracy:.2f}%")

