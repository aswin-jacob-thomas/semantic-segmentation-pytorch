# System libs
import os, csv, torch, numpy, scipy.io, PIL.Image, torchvision.transforms
# Our libs
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode
import cv2
import math
from math import atan2, degrees
import time

def initialize():
    colors = scipy.io.loadmat('data/color150.mat')['colors']
    names = {}
    with open('data/object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch='resnet50dilated',
        fc_dim=2048,
        weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
    net_decoder = ModelBuilder.build_decoder(
        arch='ppm_deepsup',
        fc_dim=2048,
        num_class=150,
        weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
        use_softmax=True)

    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module = segmentation_module.eval()

    x = torch.rand(3,672,504)
    output_size = x.shape

    # segmentation_module.cuda()
    with torch.no_grad():
        with torch.jit.optimized_execution(True, {'target_device': 'eia:0'}):
            segmentation_module = torch.jit.trace(segmentation_module, x[None])
#         torch.jit.save(segmentation_module, 'traced.pt')
    return segmentation_module

def leaning_angle(segmentation_module, image):
#     segmentation_module = torch.jit.load('traced.pt')
    pil_image = PIL.Image.open(image).convert('RGB')
    pil_image = PIL.Image.fromarray(numpy.rot90(numpy.array(pil_image), -1))
    pil_image = pil_image.resize((504, 672), PIL.Image.ANTIALIAS)
    # Load and normalize one image as a singleton tensor batch
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])

    img_original = numpy.array(pil_image)

    img_data = pil_to_tensor(pil_image)
    # singleton_batch = {'img_data': img_data[None].cuda()}
    singleton_batch = {'img_data': img_data[None]}
    output_size = img_data.shape[1:]
    start = time.time()
    with torch.no_grad():
#         scores = segmentation_module(singleton_batch, segSize=output_size)
        with torch.jit.optimized_execution(True, {'target_device': 'eia:0'}):
            scores = segmentation_module(img_data[None])
    end = time.time()
    print("The total time took - ", (end-start)*1000)
    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()
    # visualize_result(img_original, pred)
    poleBW = pred == 93
    binary_map = (poleBW > 0).astype(numpy.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, 8, cv2.CV_32S)
    largest_label = 1 + numpy.argmax(stats[1:, cv2.CC_STAT_AREA])
    stat = stats[largest_label]

    cropped = img_original[stat[1]: stat[1]+stat[3], stat[0]: stat[0]+stat[2], :]
    cropped_copy = cropped.copy()
    edges = cv2.Canny(cropped_copy,100,200,3)
    lines = cv2.HoughLinesP(edges,rho=1,theta=numpy.pi/180,threshold=80, minLineLength=20,maxLineGap=60)
    X = []
    x = []
    y = []
    for line in lines:
      for x1, y1, x2, y2 in line:
        X.append([x1, y1])
        X.append([x2, y2])
        x.extend([x1, x2])
        y.extend([y1, y2])
    X = numpy.vstack(X)

    (vx, vy, x0, y0) = cv2.fitLine(X, cv2.DIST_L12, 0, 0.01, 0.01)

    d = degrees(atan2(vy, vx))
    if d<0:
      return 90+d
    else:
      return 90-d

#
#     cropped_copy_hough = cropped.copy()
#     edges = cv2.Canny(cropped_copy_hough,100,200,3)
#     lines = cv2.HoughLines(edges, 1, numpy.pi / 180, 150)
#     X = []
#     if lines is not None:
#       for i in range(0, len(lines)):
#         rho = lines[i][0][0]
#         theta = lines[i][0][1]
#         a = math.cos(theta)
#         b = math.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#
#         X.append(list(pt1))
#         X.append(list(pt2))
#
#     X = numpy.vstack(X)
#
#     (vx, vy, x0, y0) = cv2.fitLine(X, cv2.DIST_L12, 0, 0.01, 0.01)
#
#     d = degrees(atan2(vy, vx))
#     if d < 0:
#       print(90+d)
#     else:
#       print(90-d)