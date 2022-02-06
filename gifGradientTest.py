
# MotionVisualizer
# Main File
# (C) 2022 by Jacob (ahhhh6980@gmail.com)

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from operator import ne
import sys, os, cv2, urllib.request, PIL
import pickle, psutil, warnings, pyfftw
import imageio as iio
import numpy as np

warnings.simplefilter('error')

def gif_to_channels(gif):
    channels = [[],[],[]]
    for img in gif:
        tmp = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        for i in range(3): channels[i].append(tmp[i])
    return np.array(channels)

def channels_to_gif(channels):
    gif = []
    for b,g,r in zip(*channels):
        tmp = cv2.merge([b,g,r])
        gif.append(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
    return np.array(gif)

def channels_to_grayscale_gif(channels):
    gif = []
    for b,g,r in zip(*channels):
        tmp = cv2.merge([b,g,r])
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        gif.append(cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY))
    return np.array(gif)

def url_gif_to_cv2_array(url):
    fname = "../Gradient/Gifs/"+url[1].split("/")[-1]

    if not os.path.exists(fname):
        with urllib.request.urlopen(url[0]) as request:
            # Request data
            imdata = request.read()
            imbytes = bytearray(imdata)

            # Write data to file
            with open(fname, "wb+") as tmpgif:
                tmpgif.write(imdata)
    
    # Read from file and convert frames to BGR
    gif = iio.mimread(fname)
    
    # Get FPS information
    with PIL.Image.open(fname) as gifFile:
        try:    fps = 1000 / gifFile.info['duration']
        except: fps = 0

    return [gif_to_channels(gif),fps]

pyfftw.interfaces.cache.enable()
simd = pyfftw.simd_alignment
threadCount = psutil.cpu_count()
if threadCount < 6:
    threadCount = 2
else: threadCount = 6
print("SIMD Alignment =", simd)
print("Threads In Use =", threadCount)
global written_wisdom
written_wisdom = False
def apply_kernel(img, kernel):
    global written_wisdom
    print("Starting application of kernel...", end = "")
    
    # These are intialized to be empty, and aligned for SIMD utilization
    kernelF = pyfftw.empty_aligned(img.shape, dtype='complex64', n=simd)
    imgF = pyfftw.empty_aligned(img.shape, dtype='complex64', n=simd)
    newIMG = pyfftw.empty_aligned(img.shape, dtype='complex64', n=simd)
    kernel = pyfftw.byte_align(kernel.astype(np.complex64), n=simd)
    img = pyfftw.byte_align(img.astype(np.complex64), n=simd)

    if not written_wisdom:
        print("This time will be slower, computing best method of applying the fft...", end="")

    # Create Fourier Transform object, and then execute it!
    a = pyfftw.FFTW(kernel, kernelF, direction='FFTW_FORWARD', 
        axes=(0,1,2),threads=threadCount, flags=('FFTW_MEASURE',))
    a.execute()

    if not written_wisdom:
        print(" done!")
        written_wisdom = True
        with open("../Gradient/wisdom.txt", "wb+") as f:
            pickle.dump(pyfftw.export_wisdom(), f)

    # Create Fourier Transform object, and then execute it!
    b = pyfftw.FFTW(img, imgF, direction='FFTW_FORWARD', 
        axes=(0,1,2),threads=threadCount, flags=('FFTW_MEASURE',))
    b.execute()

    # Create Fourier Transform object, and then execute it!
    temp = pyfftw.byte_align((kernelF * imgF).astype(np.complex64), n=simd)
    c = pyfftw.FFTW(temp, newIMG, direction='FFTW_BACKWARD', 
        axes=(0,1,2),threads=threadCount, flags=('FFTW_MEASURE',))
    c.execute()

    print(" done!")
    # The real component is the part we want
    return newIMG.real 

def normalize_img(img):
    newImg = img
    newImg = newImg - newImg.min()
    newImg = newImg / newImg.max()
    return newImg

def save_grad_gif(channels, fname, fps, color=True):
    # Normalize range of values to 0-255
    newChannels = []
    for channel in channels:
        newChannel = []
        for img in channel:
            newChannel.append(normalize_img(img))
        newChannels.append(newChannel)

    # Convert back to gif format
    newChannels = (np.array(newChannels) * 255).astype(np.float32)
    if color:   gifGrad = channels_to_gif(newChannels).astype(np.uint8)[5:-5]
    else:       gifGrad = channels_to_grayscale_gif(newChannels).astype(np.uint8)[5:-2]

    # Same gif
    if fps != 0:    iio.mimsave(fname,gifGrad,fps=fps)
    else:           iio.mimsave(fname,gifGrad)
    print("Saved gif!")

def np_map(f,o, *arg):
    return np.array(list(map(f,o,*arg)))

def compute_motion(url, loopGif):
    # Grab gif from url, and then convert to array
    data = url_gif_to_cv2_array(url)

    fps = data[1]
    data = data[0]
    d,h,w = data[0].shape

    # This will compute the change in image over time
    kernelDerivativeTimeFrom = np.pad(np.array(
        [[[ 0, 0, 0],[ 0,-1, 0],[ 0, 0, 0]],
         [[ 0, 0, 0],[ 0, 1, 0],[ 0, 0, 0]],
         [[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0]]]
        ), [(0,d+3), (0,h+3), (0,w+3)])

    # This will compute the change in image over time
    kernelDerivativeTimeFromTiny = np.pad(np.array(
        [[[ 0, 0, 0],[ 0,-1, 0],[ 0, 0, 0]],
         [[ 0, 0, 0],[ 0, 1, 0],[ 0, 0, 0]],
         [[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0]]]
        ), [(0,d+3), (0,h+3), (0,w+3)])

    # This will compute the edges of the image
    kernelEdgeDetect = np.pad(np.array(
        [[[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0]],
         [[-2,-1, 0],[-2, 0, 2],[ 0, 2, 2]],
         [[ 0, 0, 0],[ 0, 0, 0],[ 0, 0, 0]]]
        ), [(0,d+3), (0,h+3), (0,w+3)])

    # Iterate over channels and apply processing
    channels = []
    for i in range(3):
        # If its a gif, change the padding from blank images, to actual frames
        if loopGif:
            channelData = np.pad(data[i], [(0,0),(3,3),(3,3)])
            for i in range(3):
                channelData = np.insert(channelData, 0, 
                                channelData[-i-1], axis=0)
            for i in range(3):
                channelData = np.insert(channelData,-1, 
                                channelData[3-i], axis=0)
        else:   channelData = np.pad(data[i], 3)
        ch = channelData

        # Normalize to range [0,1]
        ch = np_map(normalize_img, ch)

        # Compute edges
        edges = apply_kernel(ch, kernelEdgeDetect)
        edges = np_map(cv2.GaussianBlur, edges, [(3,3)]*len(ch), [3]*len(ch))
        edges = np_map(normalize_img, edges)

        # Compute df/dt (change in image over change in time)
        changes = apply_kernel(edges, kernelDerivativeTimeFrom)
        changes = np.power(np_map(cv2.GaussianBlur, changes, [(3,3)]*len(ch), [3]*len(ch)), 2)
        ch = np_map(normalize_img, changes)

        channels.append(ch)

    channels = [np.array(channels), fps]
    return channels

def main():
    global written_wisdom
    # A few of the gifs I used for testing:
    urls = [

        # Rage
        # [r'https://media2.giphy.com/media/2HWWvU3wXwZna/giphy.gif', 'rage.gif'],

        # Matrix Dodge
        # [r'https://c.tenor.com/Jw8I___MCdQAAAAC/matrix-dodge.gif', 'matrix.gif'],

        # Looping gif
        # [r'https://i.gifer.com/MXgu.gif', 'loop.gif'],

        # TV Smash
        [r'https://i.pinimg.com/originals/63/b9/1a/63b91abbe6fd23219e2ef0cb6af4f59e.gif', 'tv.gif'],

        # People Leaving Wembly Stadium
        # [r'https://i.imgur.com/RqttdvX.gif', 'stadium.gif'],

        # Very noisy/low res footage of an earthquake
        #   basically just an example of how this performs with bad footage
        # [r'https://j.gifs.com/vJ0OqV.gif', 'earthquake.gif'],
        
        # City Timelapses!
        # [r'https://i.imgur.com/cTAf6g7.gif', 'city1.gif'],
        # [r'https://64.media.tumblr.com/b8c05903e89a5eeb755764cad8263a9d/tumblr_ny6ynbYpUE1tchrkco1_500.gifv', 'city2.gif'],
        
        # Ants!
        # [r'https://i.gifer.com/IKQd.gif', 'ants1.gif'],
        # [r'https://i.gifer.com/RRvW.gif', 'ants2.gif'],
    ]

    # Import information for FFTW
    with open("../Gradient/wisdom.txt", "r+b") as f:
        pyfftw.import_wisdom(pickle.load(f))

    # Compute motion
    for url in urls:
        print("Processing", url[1])
        written_wisdom = False
        data = compute_motion(url, loopGif=True)
        channels = data[0]
        fname = url[1].split("/")[-1].split(".")
        
        # Save Gif
        save_grad_gif(channels, "../Gradient/Output/"+fname[0]+"_grad.gif", data[1], color=False)

    # Export information for FFTW
    with open("../Gradient/wisdom.txt", "wb+") as f:
        pickle.dump(pyfftw.export_wisdom(), f)

if __name__ == "__main__":
    sys.exit(main())