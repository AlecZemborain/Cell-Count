"""
===================
Label image regions
===================
http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html#sphx-glr-download-auto-examples-segmentation-plot-label-py

This example shows how to segment an image with image labelling. The following
steps are applied:

1. Thresholding with automatic yen method
2. Close small holes with binary closing
3. Measure image regions to filter small objects

"""

import os, csv, glob, sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from skimage import data, io
from skimage.util import img_as_ubyte
from skimage.filters import threshold_yen, threshold_isodata, threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.color import label2rgb, rgb2gray
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk, square
from PIL import Image
from scipy.misc import imread
import cv2 as cv

""" Use a Picture class to simplify workflow and keep track of important
    information in a simple, effective manner. """
class Picture:
    def __init__(self, file):

        # convert file to a grayscale np array
        self.img = np.array(Image.open(file).convert('L'))

        # array used to keep track of pixels visited during search
        self.visited = np.zeros((len(self.img), len(self.img[0])))

        # number of pixels visited in search
        self.pixls = 0

        # number of nanowires in the image
        self.wires = 0

        # threshold used for filtering
        self.threshold = 0

        """ lists of row and col coordinates of what algorithm determines
            to be a wire """
        self.i = []
        self.j = []

    # set image to given array
    def renew(self, array):
        self.img = array

    # set specific location in image to given pixel value
    def update(self, i, j, val):
        self.img[i][j] = val

    # pixels visited by wiresearch
    def pixels(self):
        return self.pixls

    # pixels visited by wiresearch
    def total_pixels(self, number):
        self.pixls = number

    # tell whether the pixel location has already been visited by search
    def not_visited(self, i, j):
        return self.visited[i][j] == 0

    # mark location as visited
    def been_visited(self, i, j, x):
        self.visited[i][j] = x

    #increment wire count
    def inc_wire(self):
        self.wires += 1

    # return the number of wires
    def num_wires(self):
        return self.wires

    # set threshold for reducing image
    def set_threshold(self, x):
        self.threshold = x

    # get threshold for pooling image
    def get_threshold(self):
        return self.threshold


# returns if the pixel location is valid
def inbounds(img, i, j, dx = 1, dy = 1):
    return ((i >= 0) and (i < len(img)) and
            (j >= 0) and (j < len(img[0])) and not((dx==0) and (dy == 0)))


# search and count pixels neighboring cells
def adjacency(img, x, y, queue= []):
    queue.append((x, y))
    count = 0
    while queue:
        x, y = queue.pop()

        if (inbounds(img.img, x, y) and (img.img[x][y] == 255)
            and (img.not_visited(x, y))):

            count += 1
            img.i.append(x)
            img.j.append(y)
            img.been_visited(x, y, 2)
            img.total_pixels((img.pixels()+1))
            for i in range(-1,2):
                for j in range(-1,2):
                    queue.append((x+i,y+j))
    return count


# increments the number of nanowires in the image if the search found 90 pixels.
def adj(img, i, j):
    img.i = []
    img.j = []
    img.total_pixels(0)

    if (inbounds(img.img, i, j) and img.not_visited(i, j) and
        img.img[i][j] == 255):
        adjacency(img, i, j)

    if img.pixels() > 90:
        img.inc_wire()
        for i in range(0, len(img.i)):
            img.update(img.i[i], img.j[i], 100)
    return


""" reduces the image to black and white.
    sets each pixel to black or white, depending on whether the average of its
    surrounding pixels is greater than the threshold """
def reduce(img):
    newi=img.img.copy()
    filter = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(1, len(img.img) - 1):
        for j in range(1, len(img.img[0]) - 1):
            sum = 0
            for x in range(-1, 2):
                for z in range(-1, 2):
                    filter[x + 1][z + 1] = img.img[i + x][j + z]
                    sum += img.img[i+x][j+z]

            if (sum/9) > img.get_threshold():
                newi[i, j] = 255
            else:
                newi[i, j] = 0
    img.renew(newi)
    return


""" computes finds the 96th percentile of pixel values and sets filter
    theshold to the value """
def threshold(img):
    dist = np.zeros(256, dtype = int)

    #create distribution
    for i in range(0, len(img.img)):
        for j in range(0, len(img.img[0])):
            dist[img.img[i][j]] += 1

    # find 96th percentile of distribution
    tot = 0
    x=len(img.img) * len(img.img[0])
    for i in range(0, len(dist)):
        tot += dist[i]

        if (tot / x) * 100 >= 96:
            img.set_threshold(i)
            return



""" FULL DISCLOSURE: THE FOLLOWING 100 (APROXIMATELY) LINES OF CODE WERE NOT
    WRITTEN BY ME AND I AM NOT CLAIMING CREDIT FOR THEM. THEY COME FROM
    http://scikit-image.org OPEN SOURCE LIBRARIES """
###############################################################################


def get_labelled_image(image):
    # apply threshold
    thresh = threshold_yen(image)
    bw = closing(image > thresh, square(3))
    # label image regions
    return label(bw)


def count_regions(image):
    labelled_image = get_labelled_image(image)

    count = 0
    for region in regionprops(labelled_image):
        # take regions with large enough areas

        if region.area >= 1200:
            # and perform a second pass
            count += len(get_cropped_regions(image, region.bbox))

        elif region.area >= 200:
            count += 1

    return count


def crop_image(image, coords):
    """ given an image and coordinates forming a rectangle within the image
        return a cropped subimage.  images are numpy arrays """
    minr, minc, maxr, maxc = coords
    return image[minr:maxr+1, minc:maxc+1]


def get_cropped_regions(image, coords):
    """ given an image and a bounding box, crop the image, and do specific close-up
        filtering to separate cells. """

    # crop and filter the image
    cropped = crop_image(image, coords)
    cropped = erosion(cropped)
    thresh = threshold_yen(cropped)
    binary_cropped = opening(closing(cropped > thresh, square(3)), square(3))
    labelled_cropped = label(binary_cropped)

    # return the number of regions discovered with an area greater than 200 pixels
    regions = filter(lambda r: r.area >= 200, regionprops(labelled_cropped))
    return list(regions)


def label_regions(image):
    labelled_image = get_labelled_image(image)
    image_label_overlay = rgb2gray(label2rgb(labelled_image, image=image))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)
    regions = []
    for region in regionprops(labelled_image):

        if region.area >= 200:
            clr = 'green' if region.area < 1200 else 'red'
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor=clr, linewidth=1)
            regions.append([minr,minc,maxr,maxc])

            ax.add_patch(rect)

            if region.area >= 1200:
                for lilregion in get_cropped_regions(image, region.bbox):
                    bigminr, bigminc, _, _ = region.bbox
                    minr, minc, maxr, maxc = lilregion.bbox

                    rect = mpatches.Rectangle((bigminc + minc, bigminr + minr), maxc - minc, maxr - minr,
                                            fill=False, edgecolor='yellow', linewidth=1)
                    ax.add_patch(rect)
    return regions


def write_counts2(values, path):
    with open(path+"machined_counts.csv", 'w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['filename', 'wire count', 'cell count'])
        for i in range(0, len(values)):
            csv_out.writerow([values[i][0], values[i][1], values[i][2]])


def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')
    plt.show()

################################################################################
""" MY CODE RESUMES AFTER THIS """


""" marks regions containing cells as visited to avoid searching for
    nanowires in these regions """
def boundbox(img, regions):
    for item in regions:
        for i in range(item[0], item[2]):
            for j in range(item[1], item[3]):
                img.been_visited(i, j, 1)
    return


""" searches around the cells for nanowires """
def wiresearch(img, regions):
    for item in regions:
        for i in range(item[0]-1, item[2] + 1):

            if inbounds(img.img, i, item[1] - 1):
                adj(img, i, item[1] - 1)

            if inbounds(img.img, i, item[3]):
                adj(img, i, item[3])


        for i in range(item[1], item[3] + 1):

            if inbounds(img.img, item[0]-1, i):
                adj(img, item[0]-1, i)

            if inbounds(img.img, item[0], i):
                adj(img, item[2], i)
    return


# colors nanowires green, and shades regions containing cells red
def color(img, rgb, regions):
    for i in range(1, len(img.img) - 1):
        for j in range(1, len(img.img[0]) - 1):

            if img.img[i][j] > 50 and img.img[i][j] < 150:
                rgb[i][j] = [124, 252, 0]

    for item in regions:
        for i in range(item[0], item[2]):
            for j in range(item[1], item[3]):

                if img.img[i][j] > 125:
                    rgb[i][j] = [255, 182, 193]

                if img.img[i][j] < 125:
                    rgb[i][j] = [200, 0, 0]
    return rgb


""" This function takes an input folder and an output folder. It finds the
    numbers of cells and nanowire connections in each image in the folder,
    and produces a new image that clearly marks each component. It alsow writes
    to a csv file with the counts for each of the images. """
def count(input_file, output_file):
    values = []

    accepted = {'jpg': 0, 'jpeg': 0, 'png': 0, 'JPG': 0, 'PNG': 0,'JPEG':0}
    files = os.listdir(input_file)
    for filename in files:

        label = filename
        type = label.split('.')[1]

        if not(type in accepted):
            print("WRONG FROMAT, SKIPPIN " + filename)
            continue
        img = Picture(input_file + '/' + filename)

        #find and mark the cells
        regions = label_regions(img.img)
        boundbox(img, regions)

        # process the image for analysis
        img.renew(np.array(Image.fromarray(
            cv.fastNlMeansDenoisingColored(cv.cvtColor(
            img.img, cv.COLOR_GRAY2BGR), None, 10, 10, 7, 21)).convert('L')))
        threshold(img)
        reduce(img)

        # search for nanowires
        wiresearch(img, regions)

        # edit the image
        rgb = cv.cvtColor(img.img,cv.COLOR_GRAY2RGB)
        rgb = color(img, rgb, regions)

        # save the final image
        path = str(output_file + '/') + label
        rgb = Image.fromarray(rgb)
        rgb.save(path)

        print('wires =', img.num_wires())
        values.append((label, img.num_wires(), len(regions)))

    # write counts to csv
    write_counts2(values, str(output_file + '/'))

def main():

    count(sys.argv[1], sys.argv[2])

main()
