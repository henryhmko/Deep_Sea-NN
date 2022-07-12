import os
import cv2
import math


!apt install ffmpeg

video_file = '/content/original.mp4'
video_dir = '/content/'
original_dir = video_dir + 'original/'
corrected_dir = video_dir + 'corrected/'

if os.path.isfile(video_file):
    vidcap = cv2.VideoCapture(video_file)
else:
    print("No path to video! Recheck path :(")

#get that frame rate of your video
fps = int(math.floor(vidcap.get(cv2.CAP_PROP_FPS)))

#if success==False, it means vid is not read :/
success, vimg = vidcap.read()
count = 0

while success:
  success, vimg = vidcap.read()
  if not success: continue
  vimg = np.array(Image.fromarray(vimg).resize((512,512))) #512x512 is good but it can be changed too
  cv2.imwrite(video_dir+'/original/frame_%d.png'%count, vimg)
  count += 1



#MAKING VIDEO
# 1. video for original
cmd_orginal = 'ffmpeg -framerate ' +str(fps) + ' -i '+original_dir+'frame_%d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p /content/original_vid.mp4'
##os.system(cmd_original) #Uncomment to run


# 2. video for corrected
cmd_corrected = 'ffmpeg -framerate ' +str(fps) + ' -i '+corrected_dir+'frame_%d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p /content/corrected_vid.mp4'
##os.system(cmd_corrected) #Uncomment to run

# videos side by side
cmd_side = 'ffmpeg -i ' + '/content/original_vid.mp4 -i ' + '/content/corrected_vid.mp4 -filter_complex \'[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]\' -map [vid] -c:v libx264 -crf 23 -preset veryfast '+ '/content/final_comparison.mp4'
##os.system(cmd_side) #Uncomment to run

