
import cv2 as cv
import matplotlib.pyplot as plt

IMG = "imgs/flock-of-fish.webp"
IMG2 = "imgs/fish-pattern.png"

def template_matching(img_gray, template_gray):
    result = cv.matchTemplate(img_gray, template_gray, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    # Draw a rectangle around the matched region
    top_left = max_loc
    h, w = template_gray.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    return [result, top_left, bottom_right]


if __name__ == "__main__":
    img = cv.imread(IMG)
    template = cv.imread(IMG2)

    # Convert to grayscale
    template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    levels = 3
    g_pyr = [img]
    for i in range(levels):
        g_pyr.append(cv.pyrDown(g_pyr[i]))
    num_levels = len(g_pyr)  # should be levels + 1

    # create enough columns for every pyramid image
    fig, axes = plt.subplots(2, num_levels, figsize=(15, 5))

    for i, img_ in enumerate(g_pyr):
        img_gray = cv.cvtColor(img_, cv.COLOR_BGR2GRAY)
        r, tl, br = template_matching(img_gray, template_gray)
        cv.rectangle(img_, tl, br, (0, 255, 0), 2)

        axes[0, i].imshow(cv.cvtColor(img_, cv.COLOR_BGR2RGB))
        axes[0, i].set_title(f"Gaussian Level {i}")

        axes[1, i].imshow(r, cmap="jet")
        axes[1, i].set_title(f"Template matching result Level {i}")
    
    plt.tight_layout()
    plt.show()
