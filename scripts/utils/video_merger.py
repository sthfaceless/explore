import cv2

videos = ["D:\Avatar.Extended.2009.BDRip.1080p.Rus.mkv", "D:\Avatar.Extended.2009.BDRip.1080p.Rus.mkv"]
result = "D:\Avatar.Extended.2009.BDRip.1080p.Rus.mp4"

# read common video properties
reader = cv2.VideoCapture(videos[0])
video_fps = int(reader.get(cv2.CAP_PROP_FPS))
height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
reader.release()

# open mp4 writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(result, apiPreference=0, fourcc=fourcc, fps=video_fps, frameSize=(width, height))

# write all frames from videos to one video
for video in videos:
    reader = cv2.VideoCapture(video)
    while reader.isOpened():
        ret, frame = reader.read()
        if not ret:
            break
        writer.write(frame)
    reader.release()

writer.release()
