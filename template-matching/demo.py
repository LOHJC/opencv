
import cv2 as cv
import matplotlib.pyplot as plt

IMG = "imgs/flock-of-fish.webp"
IMG2 = "imgs/fish-pattern.png"

# IMG = "imgs/flock-of-fish-2.webp"
# IMG2 = "imgs/fish-pattern-2.png"


if __name__ == "__main__":
    img = cv.imread(IMG)
    template = cv.imread(IMG2)

    # Convert to grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    # Template matching
    result = cv.matchTemplate(img_gray, template_gray, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    # Draw a rectangle around the matched region
    top_left = max_loc
    h, w = template_gray.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

    # Display the result
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Matched Image")
    plt.subplot(1, 2, 2)
    plt.imshow(result, cmap="jet")
    plt.colorbar()
    plt.title("Template Matching Result")
    plt.tight_layout()
    plt.show()