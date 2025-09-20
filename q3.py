import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import helpers as h

image = cv.imread('images/pre_img.jpg')
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

median_denoised_img = cv.medianBlur(image, 5) # Kernel size 5x5

non_local_denoised_img = cv.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

gaussian_denoised_img = cv.GaussianBlur(image, (5, 5), 0)

median_and_gaussian_denoised_img = cv.GaussianBlur(median_denoised_img, (5, 5), 0)

#increase kernel size
increased_kernel_size = cv.GaussianBlur(median_denoised_img, (9, 9), 0)

median_and_non_local_denoised_img = cv.fastNlMeansDenoisingColored(median_denoised_img, None, 10, 10, 7, 21)


def contrast_stretch_rgb(img_rgb, low_pct=2, high_pct=98):
    # convert RGB → YCrCb
    ycc = cv.cvtColor(img_rgb, cv.COLOR_RGB2YCrCb)
    Y, Cr, Cb = cv.split(ycc)

    Yf = Y.astype(np.float32)
    p1, p2 = np.percentile(Yf, (low_pct, high_pct))
    a = 255.0 / max(p2 - p1, 1e-6)
    b = -a * p1

    Ys = np.clip(a * Yf + b, 0, 255).astype(np.uint8)
    out = cv.merge((Ys, Cr, Cb))
    # back to RGB
    return cv.cvtColor(out, cv.COLOR_YCrCb2RGB)

def boost_saturation_rgb(img_rgb, s_gain=1.25, v_gain=1.0):
    hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV).astype(np.float32)
    hsv[...,1] *= s_gain   # saturation
    hsv[...,2] *= v_gain   # optional: brightness
    hsv[...,1:] = np.clip(hsv[...,1:], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv.cvtColor(hsv, cv.COLOR_HSV2RGB)

def enhance_soft_rgb(img_rgb,
                     low_pct=5, high_pct=99,    # gentler stretch
                     out_lo=10, out_hi=245,     # headroom to avoid clipping
                     gamma=0.9,                 # <1 brightens midtones
                     sat_gain=1.25):            # boost color

    # 1) Stretch luminance in YCrCb
    ycc = cv.cvtColor(img_rgb, cv.COLOR_RGB2YCrCb)
    Y, Cr, Cb = cv.split(ycc)

    Yf = Y.astype(np.float32)
    p1, p2 = np.percentile(Yf, (low_pct, high_pct))
    a = (out_hi - out_lo) / max(p2 - p1, 1e-6)
    b = out_lo - a * p1
    Y1 = np.clip(a * Yf + b, 0, 255)

    # 2) Mild gamma to shape midtones (gamma<1 -> brighter)
    Y1 = 255.0 * ((Y1 / 255.0) ** gamma)
    Y1 = np.clip(Y1, 0, 255).astype(np.uint8)

    rgb = cv.cvtColor(cv.merge((Y1, Cr, Cb)), cv.COLOR_YCrCb2RGB)

    # 3) Add saturation in HSV
    hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= sat_gain
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv.cvtColor(hsv, cv.COLOR_HSV2RGB)

def grayworld_rgb(img_rgb):
    f = img_rgb.astype(np.float32)
    means = f.mean(axis=(0,1))                # R,G,B means
    gain = means.mean() / (means + 1e-6)
    out = np.clip(f * gain, 0, 255).astype(np.uint8)
    return out

# Apply contrast stretching
contrast_stretched_img = contrast_stretch_rgb(median_denoised_img)
contrast_stretched_img = boost_saturation_rgb(contrast_stretched_img, s_gain=7.25)

enh = enhance_soft_rgb(median_denoised_img, low_pct=20, high_pct=80,
                       out_lo=0, out_hi=245, gamma=0.75, sat_gain=7.25)


mg_enhanced = enhance_soft_rgb(median_and_gaussian_denoised_img, low_pct=20, high_pct=80,
                                 out_lo=0, out_hi=245, gamma=0.75, sat_gain=7.25)

mg_contrast_stretched = contrast_stretch_rgb(median_and_gaussian_denoised_img)
mg_contrast_stretched = boost_saturation_rgb(mg_contrast_stretched, s_gain=7.25)

median_and_non_local_enhanced = enhance_soft_rgb(median_and_non_local_denoised_img, low_pct=20, high_pct=80,
                                    out_lo=0, out_hi=245, gamma=0.95, sat_gain=7.25)

median_and_non_local_contrast_stretched = contrast_stretch_rgb(median_and_non_local_denoised_img)
median_and_non_local_contrast_stretched = boost_saturation_rgb(median_and_non_local_contrast_stretched, s_gain=7.25)

increased_kernel_filters = enhance_soft_rgb(increased_kernel_size, low_pct=10, high_pct=90,
                                    out_lo=0, out_hi=250, gamma=1.15, sat_gain=7.25)

increased_kernel_stretched = contrast_stretch_rgb(increased_kernel_size)
increased_kernel_stretched = boost_saturation_rgb(increased_kernel_stretched, s_gain=7.25)


# Create a figure with subplots


'''
fig, axs = plt.subplots(4, 3, figsize=(18, 6))
axs[0][0].imshow(median_denoised_img)
axs[0][0].set_title("Median Denoised Image")
axs[0][0].axis('off')

axs[0][1].imshow(contrast_stretched_img)
axs[0][1].set_title("Contrast Stretched Image")
axs[0][1].axis('off')

axs[0][2].imshow(enh)
axs[0][2].set_title("Enhanced Image")
axs[0][2].axis('off')

axs[1][0].imshow(median_and_gaussian_denoised_img)
axs[1][0].set_title("Median and Gaussian Denoised Image")
axs[1][0].axis('off')

axs[1][1].imshow(mg_contrast_stretched)
axs[1][1].set_title("Median + Gaussian Contrast Stretched")
axs[1][1].axis('off')

axs[1][2].imshow(mg_enhanced)
axs[1][2].set_title("Median + Gaussian Enhanced")
axs[1][2].axis('off')

axs[2][0].imshow(median_and_non_local_denoised_img)
axs[2][0].set_title("Median and Non-local Denoised Image")
axs[2][0].axis('off')

axs[2][1].imshow(median_and_non_local_contrast_stretched)
axs[2][1].set_title("Median + Non-local Contrast Stretched")
axs[2][1].axis('off')

axs[2][2].imshow(median_and_non_local_enhanced)
axs[2][2].set_title("Median + Non-local Enhanced")
axs[2][2].axis('off')

axs[3][0].imshow(increased_kernel_size)
axs[3][0].set_title("Increased Kernel Size Denoised Image")
axs[3][0].axis('off')

axs[3][1].imshow(increased_kernel_stretched)
axs[3][1].set_title("Increased Kernel Contrast Stretched")
axs[3][1].axis('off')

axs[3][2].imshow(increased_kernel_filters)
axs[3][2].set_title("Increased Kernel Enhanced")
axs[3][2].axis('off')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

'''

#open template images
temp_001 = cv.imread('images/pre_template_001.jpg')
temp_001 = cv.cvtColor(temp_001, cv.COLOR_BGR2RGB)

temp_002 = cv.imread('images/pre_template_002.jpg')
temp_002 = cv.cvtColor(temp_002, cv.COLOR_BGR2RGB)

temp_003 = cv.imread('images/pre_template_003.jpg')
temp_003 = cv.cvtColor(temp_003, cv.COLOR_BGR2RGB)

#apply median noise reduction
temp_001 = cv.medianBlur(temp_001, 5)
temp_002 = cv.medianBlur(temp_002, 5)
temp_003 = cv.medianBlur(temp_003, 5)


#apply gaussian to template images
temp_001 = cv.GaussianBlur(temp_001, (9, 9), 0)
temp_002 = cv.GaussianBlur(temp_002, (9, 9), 0)
temp_003 = cv.GaussianBlur(temp_003, (9, 9), 0)

#apply contrast stretching and saturation boost to template images
temp_001 = enhance_soft_rgb(temp_001, low_pct=10, high_pct=90,
                                    out_lo=0, out_hi=250, gamma=1.15, sat_gain=7.25)

temp_002 = enhance_soft_rgb(temp_002, low_pct=10, high_pct=90,
                                    out_lo=0, out_hi=250, gamma=1.15, sat_gain=7.25)

temp_003 = enhance_soft_rgb(temp_003, low_pct=10, high_pct=90,
                                    out_lo=0, out_hi=250, gamma=1.15, sat_gain=7.25)


#template matching
def template_matching(image, template):
    image_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    template_gray = cv.cvtColor(template, cv.COLOR_RGB2GRAY)

    result = cv.matchTemplate(image_gray, template_gray, cv.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv.minMaxLoc(result)

    h, w = template_gray.shape
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    coordinates = {top_left[0], top_left[1], w, h}

    matched_image = image.copy()
    cv.rectangle(matched_image, top_left, bottom_right, (255, 0, 0), 2)

    return matched_image, max_val, coordinates



matched_001, score_001, coordinates_001 = template_matching(increased_kernel_filters, temp_001)
print(score_001)
print("Coordinates (x, y, width, height):", coordinates_001)
matched_002, score_002, coordinates_002 = template_matching(increased_kernel_filters, temp_002)
print(score_002)
print("Coordinates (x, y, width, height):", coordinates_002)
matched_003, score_003, coordinates_003 = template_matching(increased_kernel_filters, temp_003)
print(score_003)
print("Coordinates (x, y, width, height):", coordinates_003)


fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(matched_001)
axs[0].set_title(f"Template 1 Matched Image - Score: {score_001:.4f}")
axs[0].axis('off')
axs[1].imshow(matched_002)
axs[1].set_title(f"Template 2 Matched Image - Score: {score_002:.4f}")
axs[1].axis('off')
axs[2].imshow(matched_003)
axs[2].set_title(f"Template 3 Matched Image - Score: {score_003:.4f}")
axs[2].axis('off')
plt.tight_layout()
plt.show()
