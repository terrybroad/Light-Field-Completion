import pygame, glob, json, commands
from pprint import pprint

whichFile = "truck"
jpegFilenames = glob.glob(whichFile + "*.jpg")
shasumToJpeg = {}
for filename in jpegFilenames:
    shasum = commands.getstatusoutput("shasum " + filename)[1]
    shasum = "sha1-" + shasum.split(" ")[0]
    shasumToJpeg[shasum] = pygame.image.load(filename)
print "Loading filenames:", jpegFilenames
pprint(shasumToJpeg)

#print "Loading depth table"
#depths = [float(x.strip()) for x in open(whichFile + "_depth.txt")]
#print len(depths), "depth numbers found"

print "Loading jpeg -> depth table"
metadata = json.load(open(whichFile + "_table.json"))
imageArray = metadata["picture"]["accelerationArray"][0]["vendorContent"]["imageArray"]


        # find closest jpeg

li = []
for j, img in enumerate(imageArray):
  closestImg = None
  closestDepth = 100000000

  d = 0
  for i, img in enumerate(imageArray):
    delta = img["lambda"]

    if ((delta < closestDepth) and (delta not in li)):
      closestDepth = delta
      closestImg = img["imageRef"]
      closestImg = shasumToJpeg[closestImg]
      d = delta

  li.append(d)
  print li
  string = 'reorder/truck'
  string +=`j`
  string +='.jpg'
  jpegOut = pygame.Surface((1080, 1080))
  rect = pygame.Rect(0, 0, 1080, 1080)
  subSurf = closestImg.subsurface(rect)
  jpegOut.blit(subSurf,(0,0))
  pygame.image.save(jpegOut, string)
