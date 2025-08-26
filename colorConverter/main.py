# You can use this file to translate the colors of images to match the color space of wplace.
# This way the color codes can be detected by the overlay reliably.
# Run this script with a "source.png" image in the same folder, you will receive a converted "target.png"

from PIL import Image

palette = Image.open("palette.png").convert('RGBA')
palettepix = palette.load()

palettevals = []

for x in range(palette.width):
    color = palettepix[x,0]
    palettevals.append([color[0],color[1],color[2]])

source = Image.open("source.png").convert('RGBA')
sourcepix = source.load()
for x in range(source.width):
    for y in range(source.height):
        readcolor = sourcepix[x,y]
        if readcolor[3] == 0:
            continue
        best = [0,0,0]
        best_diff = 255*3
        for element in palettevals:
            new_diff = abs(element[0]-readcolor[0]) + abs(element[1]-readcolor[1]) + abs(element[2]-readcolor[2])
            if new_diff < best_diff:
                best_diff = new_diff
                best = element
        sourcepix[x,y] = (best[0],best[1],best[2],255)

source.save("target.png", 'PNG')
print(palettevals)
