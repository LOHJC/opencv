
import cv2 as cv
import matplotlib.pyplot as plt

# IMG = "../imgs/old-joestar.webp"
IMG = "../imgs/jojo_512.jpg"

if __name__ == "__main__":
    img = cv.imread(IMG)
    img = cv.resize(img, (512, 512))

    # Gaussian Pyramid
    g_pyr = [img]
    for i in range(6):
        g_pyr.append(cv.pyrDown(g_pyr[i]))

    # Laplacian Pyramid
    l_pyr = [g_pyr[5]]
    for i in range(5, 0, -1):
        size = (g_pyr[i - 1].shape[1], g_pyr[i - 1].shape[0])
        l_pyr.append(cv.subtract(g_pyr[i - 1], cv.pyrUp(g_pyr[i], dstsize=size)))

    # Display the pyramids
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))

    for i in range(6):
        axes[0, i].imshow(cv.cvtColor(g_pyr[i], cv.COLOR_BGR2RGB))
        axes[0, i].set_title(f"Gaussian Level {i}")
        axes[0, i].axis("off")
        
        if i < 5:
            axes[1, i].imshow(cv.cvtColor(l_pyr[5 - i], cv.COLOR_BGR2RGB))
            axes[1, i].set_title(f"Laplacian Level {i}")
            axes[1, i].axis("off")
        else:
            axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()